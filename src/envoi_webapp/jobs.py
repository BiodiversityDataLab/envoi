"""Job manager of the web app: one worker process per extraction job.

Earth Engine keeps its credentials in one module-global state object, so the
web app runs each job in its own worker process (``python -m
envoi_webapp.worker``). One :class:`JobManager` per server process starts these
processes, reads their messages, applies the limits, and deletes the job
workspaces. :func:`get_job_manager` creates that manager and returns it to
every session.

* :meth:`JobManager.submit` checks the job limits, reserves a slot, creates a
  job workspace with a random name, and starts the worker. It returns at once.
* One channel thread per job writes the pickled request to the worker's stdin,
  then reads the worker's messages until the worker exits.
* In hosted mode, one housekeeping thread applies the run-time limit, the
  workspace size limit, the abandon time-out, and the retention time.
* :meth:`JobManager.snapshot` gives the UI the status of a job, and counts as
  the heartbeat for the abandon time-out.

Key handling: the manager builds the request with the key, pickles it once,
and the channel thread drops the pickled bytes after it wrote them to the
worker. The manager keeps only a SHA-256 hash of the key material
(:func:`key_hash`), for the per-key limit. The worker command line, the worker
environment, and the files that the manager writes contain no key material.
The manager never logs a line from the worker's message channel. When a worker
crashes, it logs at most the last 2 kB of the worker's stderr file, which the
worker redacted.

Locking: one lock protects the job records. The manager never holds it while it
waits for a process, writes to a pipe, or walks a folder. A module-level lock
makes :func:`get_job_manager` create at most one manager per server process,
and another one makes the hosted start-up clean-up run at most once per server
process and workspace root.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from envoi_webapp.job_protocol import (
    DoneMessage,
    ErrorMessage,
    JobMessage,
    JobRequest,
    JobSnapshot,
    JobState,
    ProgressMessage,
    parse_message,
)
from envoi_webapp.settings import (
    WORKSPACE_SUBFOLDER_NAME,
    WORKSPACE_VARIABLE,
    WebappSettings,
    format_bytes,
    format_minutes,
    load_settings,
)

logger = logging.getLogger(__name__)

# Job IDs and job workspace names are secrets.token_urlsafe(16): 128 random
# bits as 22 URL-safe characters. The start-up clean-up deletes only folders
# with this name format and the marker file, so it cannot delete other folders.
_TOKEN_BYTES = 16
_WORKSPACE_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{22}")
_WORKSPACE_MARKER_NAME = ".envoi-webapp-job"
_WORKER_STDERR_NAME = "worker-stderr.log"
# In hosted mode, extract() writes into this subfolder of the workspace. The
# archive is in the workspace itself, outside this folder.
_HOSTED_OUTPUT_FOLDER_NAME = "outputs"

# The manager removes these variables from the worker's environment, so that
# the worker cannot use a server credential instead of the user's key (R3).
_CREDENTIAL_VARIABLES = frozenset({"ENVOI_EE_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS"})
# In hosted mode, these variables point the worker's temporary folder to the
# job workspace. Python's tempfile module reads TMPDIR, TEMP, and TMP in this
# order, other libraries read one of them, and GDAL (rasterio) reads CPL_TMPDIR.
# Temporary files then count toward the workspace size limit, and the manager
# deletes them with the workspace.
_TEMPORARY_FOLDER_VARIABLES = ("TMPDIR", "TEMP", "TMP", "CPL_TMPDIR")

# The hosted start-up clean-up runs at most once per server process and
# workspace root. A second manager in the same process (for example the one
# that deploy/serve/smoke_check.py creates, or one in a test) would otherwise
# delete the workspaces of the first manager's running jobs. The set holds the
# resolved roots that were cleaned.
_cleaned_workspace_roots: set[Path] = set()
_cleaned_workspace_roots_lock = threading.Lock()

# Time between terminate() and kill() when the manager stops a worker.
_STOP_WAIT_S = 5.0
# The housekeeping pass stops a worker that has not exited this long after its
# job got a final state, for example a worker that sent "done" and then hangs.
# The final state does not change.
_FINAL_STATE_EXIT_GRACE_S = 30.0
# At most this much of the end of a crashed worker's stderr file goes to the
# server log.
_STDERR_TAIL_BYTES = 2048
# The housekeeping pass forgets an ended job this long after it ended, when its
# workspace is deleted. A session that asks later gets "unknown job".
_FORGET_ENDED_JOB_AFTER_S = 24 * 60 * 60
# Attempts to delete a workspace. Windows can refuse a deletion for a short
# time after the worker process exited.
_DELETE_ATTEMPTS = 3
_DELETE_RETRY_WAIT_S = 0.1

# UTC time in the names of the run log and the archive.
_FILE_TIME_FORMAT = "%Y%m%dT%H%M%SZ"

# Texts for the user.
_SESSION_BUSY_MESSAGE = (
    "An extraction of this page is still running. Wait until it ends, or cancel it, "
    "before you start a new one."
)
_SERVER_FULL_MESSAGE = (
    "The server is running the maximum number of extractions. Try again in a few minutes."
)
_DISK_FULL_MESSAGE = (
    "The server does not have enough free disk space for another extraction now. "
    "Try again in a few minutes."
)
_START_FAILED_MESSAGE = "The extraction process could not start."
_CRASH_MESSAGE = "The extraction process ended unexpectedly."
_UNEXPECTED_RESULT_MESSAGE = "The extraction process sent a result that the web app cannot use."
_CANCEL_REASON = "You cancelled the extraction."
_CANCEL_FOR_KEY_REASON = (
    "The extraction was cancelled from a page that uses the same service-account key."
)
_SHUTDOWN_REASON = "The web app stopped."
# Stop reasons of the housekeeping pass. {limit} is the limit in the user's units.
_RUN_TIME_REASON = (
    "The extraction ran longer than the limit of {limit} and was stopped. "
    "Use fewer points, data products, or window sizes."
)
_ABANDON_REASON = (
    "The extraction was stopped because its page did not ask for the job status for {limit}. "
    "Keep the page open and visible while an extraction runs."
)
_WORKSPACE_SIZE_REASON = (
    "The results of the extraction grew larger than the limit of {limit}, and the extraction "
    "was stopped. Use fewer points, smaller windows, or data products with fewer bands."
)


class JobRejected(ValueError):
    """The job manager did not start a job because a limit is reached.

    The message is for the user. It says why, and what the user can do.
    ``reason`` names the limit, so that the UI can offer an action: for
    ``"key"``, the UI offers to cancel the earlier job of the key with
    :meth:`JobManager.cancel_for_key`.
    """

    def __init__(
        self, message: str, *, reason: Literal["session", "key", "server", "disk"]
    ) -> None:
        super().__init__(message)
        # "session": this session runs a job. "key": a job with this key runs.
        # "server": the server runs the maximum number of jobs. "disk": the
        # disk budget has no room for another job.
        self.reason = reason


def key_hash(credentials_json: str | bytes) -> str:
    """Return a SHA-256 hash (hex) of the key material of a service-account key.

    The hash covers the ``client_email``, the ``private_key_id``, and the
    ``private_key`` of the key. A missing field counts as an empty text. The
    job manager keeps only this hash of a key. It uses it for the limit of one
    running job per key in hosted mode, and for :meth:`JobManager.cancel_for_key`.
    The UI computes the same hash from the key that the session uploaded.

    Because the hash includes the private key:

    * a forged key with the ``client_email`` of another user's key gives
      another hash, so it can neither cancel nor block that user's job,
    * two different keys of the same service account count as different keys.

    The hash reveals no key material: SHA-256 cannot be reversed, and a private
    key has too many possible values to guess.

    Raises:
        ValueError: when the text is not a JSON object. The message contains
            no key material.
    """
    # The exception of json.loads() keeps the whole key text in its "doc"
    # attribute. The except block therefore only sets a value, and the error
    # is raised after it, so that the new exception has no __context__ that
    # carries the key (the same reason as in envoi.auth.init_gee()).
    try:
        key = json.loads(credentials_json)
    except ValueError:
        key = None
    if not isinstance(key, dict):
        raise ValueError("The service-account key must be a JSON object.")

    # One canonical text of the three fields, so that the same key always
    # gives the same hash, whatever the layout of the key file.
    key_fields = [key.get(name, "") for name in ("client_email", "private_key_id", "private_key")]
    canonical_text = json.dumps(key_fields, ensure_ascii=True)
    return hashlib.sha256(canonical_text.encode("ascii")).hexdigest()


def _manager_error(message: str) -> ErrorMessage:
    """Return an ``error`` message that the job manager sets when the worker sent none."""
    return {"type": "error", "message": message, "warning_count": 0, "run_log_tail": []}


@dataclass(eq=False)
class _Job:
    """The job manager's record of one job. The manager's lock protects every field."""

    job_id: str
    session_token: str
    key_hash: str
    # The job workspace. submit() creates the folder after the reservation.
    workspace: Path
    # The archive path that the request gives the worker (hosted mode), or None
    # (local mode). A "done" message with another archive value gives FAILED,
    # so the UI only gets this path.
    archive_path: Path | None
    started_at: float
    # Time of the latest snapshot() call. The abandon time-out counts from it.
    last_heartbeat_at: float
    state: JobState = JobState.RUNNING
    # None until submit() has started the worker process.
    process: subprocess.Popen | None = None
    channel_thread: threading.Thread | None = None
    ended_at: float | None = None
    # The latest progress message per segment (batch_id, dataset,
    # window_size_m, mode), in the order in which the segments first reported.
    progress: dict[tuple[str, str, int, str], ProgressMessage] = field(default_factory=dict)
    result: DoneMessage | None = None
    error: ErrorMessage | None = None
    stop_reason: str | None = None
    # Time of the first mark_downloaded() call. Later calls do not move it.
    downloaded_at: float | None = None
    # True after the worker process exited (or never started). The manager
    # deletes a workspace only after this.
    process_exited: bool = False
    # Size of the workspace after the worker exited. None while the worker can
    # still write: the disk budget then counts the full workspace limit.
    workspace_bytes: int | None = None
    # True while one thread deletes the workspace, so no second thread does.
    deleting_workspace: bool = False
    workspace_deleted: bool = False
    # True after discard(), or after the same session started a new job. The
    # manager deletes the workspace when the worker exited, and then forgets
    # the job.
    discarded: bool = False

    @property
    def log_label(self) -> str:
        """Short name of the job for the server log. The full job ID is not logged."""
        return self.job_id[:8]


class JobManager:
    """Start, watch, stop, and clean up the extraction jobs of one web-app server process.

    The web app keeps one manager per server process (through
    :func:`get_job_manager`). A session keeps only the job ID that
    :meth:`submit` returns, and its own session token. In hosted mode the
    manager applies the limits of ``settings.limits``. In local mode only the
    limit of one running job per session applies, and "Cancel" works.

    All public methods are safe to call from several threads (Streamlit runs
    each session in its own thread). Job state lives in memory only. A restart
    of the server process loses it, and in hosted mode the next start deletes
    the workspaces that are left.
    """

    def __init__(
        self,
        settings: WebappSettings,
        *,
        worker_command: Sequence[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        housekeeping_interval_s: float | None = 5.0,
    ) -> None:
        """Create the workspace root and, in hosted mode, start the housekeeping thread.

        The workspace root is a folder in ``settings.workspace_parent``:

        * hosted mode: ``envoi-webapp-jobs``. The constructor creates it with
          mode ``0o700`` when it is missing. On POSIX, it refuses a folder that
          is a symbolic link, belongs to another user, or gives permissions to
          the group or to others. It does not change such a folder.
        * local mode: a new folder ``envoi-webapp-jobs-<random>`` with mode
          ``0o700`` for this manager only. :meth:`shutdown` removes it when it
          is empty.

        Args:
            settings: The web-app settings.
            worker_command: The command that starts a worker process. The
                default is ``[sys.executable, "-m", "envoi_webapp.worker"]``.
                Tests give a fake worker.
            clock: Returns the current time in seconds. All job times use it.
                Tests give a fake clock.
            housekeeping_interval_s: Time between two housekeeping passes in
                hosted mode. None starts no housekeeping thread: tests then call
                :meth:`run_housekeeping` themselves.

        Raises:
            ValueError: in hosted mode on POSIX, when the workspace root is not
                safe to use. The message names the folder and the fix.

        In hosted mode, the first manager of a server process for a workspace
        root deletes the workspaces that an earlier server process left: the
        folders in the root with the workspace name format and the marker file.
        It deletes nothing else.
        """
        self._settings = settings
        self._limits = settings.limits
        self._worker_command = (
            [sys.executable, "-m", "envoi_webapp.worker"]
            if worker_command is None
            else list(worker_command)
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._shutdown_event = threading.Event()
        self._housekeeping_thread: threading.Thread | None = None

        # Workspace root. Hosted mode uses a fixed folder, so that the next
        # server process finds the workspaces that this one leaves. Local mode
        # uses a new private folder: on a shared computer, another user could
        # otherwise create a fixed folder first and replace a new workspace,
        # which is the worker's working folder, before the worker starts.
        if settings.is_hosted:
            self._workspace_root = settings.workspace_parent / WORKSPACE_SUBFOLDER_NAME
            try:
                self._workspace_root.mkdir(mode=0o700, parents=True)
            except FileExistsError:
                pass
            _check_hosted_workspace_root(self._workspace_root)
        else:
            settings.workspace_parent.mkdir(parents=True, exist_ok=True)
            self._workspace_root = Path(
                tempfile.mkdtemp(
                    prefix=f"{WORKSPACE_SUBFOLDER_NAME}-", dir=settings.workspace_parent
                )
            )

        # Hosted mode: the start-up clean-up, once per server process and root,
        # and the housekeeping thread. The lock makes a second manager wait
        # until the clean-up is done, so the clean-up cannot delete that
        # manager's new workspaces.
        if settings.is_hosted:
            resolved_root = self._workspace_root.resolve()
            with _cleaned_workspace_roots_lock:
                if resolved_root not in _cleaned_workspace_roots:
                    self._delete_leftover_workspaces()
                    _cleaned_workspace_roots.add(resolved_root)
            if housekeeping_interval_s is not None:
                self._housekeeping_thread = threading.Thread(
                    target=self._run_housekeeping_loop,
                    args=(housekeeping_interval_s,),
                    name="envoi-webapp-housekeeping",
                    daemon=True,
                )
                self._housekeeping_thread.start()

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def submit(
        self,
        *,
        session_token: str,
        points: pd.DataFrame,
        run_configs: list[dict[str, Any]],
        input_crs: str,
        credentials_json: str | bytes,
        output_dir: Path | None = None,
    ) -> str:
        """Check the job limits, start a worker process for the job, and return the job ID.

        The method returns when the worker process has started. It does not
        wait for the worker to read the request.

        Admission (under the lock, with a reserved slot, so two quick
        submissions cannot both pass a limit):

        * both modes: one running job per session,
        * hosted mode: one running job per key, the maximum number of
          concurrent jobs, and the disk budget: (the sizes of the finished
          workspaces) + (running jobs + 1) x the workspace limit must not be
          larger than the budget.

        When the job starts, the earlier jobs of the same session are discarded
        (in hosted mode, their results are deleted). In local mode, the method
        also tries again to delete the workspaces whose deletion failed before.

        Args:
            session_token: The random token of the Streamlit session.
            points: The validated input points with the Darwin Core columns.
            run_configs: The run configurations for ``envoi.extract()``.
            input_crs: The CRS of the input coordinates.
            credentials_json: The service-account key as JSON text (``str``,
                or ``bytes`` in UTF-8). The manager sends it to the worker as
                ``str`` and keeps no reference to it.
            output_dir: Local mode only: the absolute path of the folder that
                the user chose. In hosted mode the manager chooses the folder,
                and this must be None.

        Returns:
            The job ID: a random text that the session uses in the other calls.
            When the worker process cannot start, the job has the state FAILED.

        Raises:
            JobRejected: when a limit is reached. The message is for the user.
            ValueError: when ``output_dir`` does not fit the mode, or the key
                is not UTF-8 text or not a JSON object. The message contains
                no key material.
        """
        if self._settings.is_hosted:
            if output_dir is not None:
                raise ValueError(
                    "In hosted mode the job manager chooses the output folder. "
                    "Do not pass output_dir."
                )
        elif output_dir is None or not Path(output_dir).is_absolute():
            raise ValueError("In local mode, output_dir must be an absolute folder path.")

        # The key as text, so that the hash and the worker get the same value.
        # A UnicodeDecodeError keeps the whole key in its "object" attribute,
        # so the except block only sets a value, and the error is raised after
        # it, without a __context__ that carries the key.
        if isinstance(credentials_json, bytes):
            try:
                key_text: str | None = credentials_json.decode("utf-8")
            except UnicodeDecodeError:
                key_text = None
            if key_text is None:
                raise ValueError("The service-account key must be UTF-8 text.")
        else:
            key_text = credentials_json
        job_key_hash = key_hash(key_text)

        # Local mode has no housekeeping thread: retry the workspace deletions
        # that failed earlier, for example on Windows while a file was open.
        if not self._settings.is_hosted:
            self._delete_due_workspaces()

        # The run log and the archive are named after the UTC start time. Both
        # names are fixed templates with UTC digits only and contain no user
        # data, so they meet the rules of envoi._filenames without a check.
        file_time = datetime.now(timezone.utc).strftime(_FILE_TIME_FORMAT)
        run_log_name = f"envoi-run-log-{file_time}.txt"
        archive_name = f"envoi-results-{file_time}.zip"

        # The workspace gets a random name in the workspace root. In hosted
        # mode, extract() writes into its outputs folder, and the archive is
        # next to that folder.
        workspace = self._workspace_root / secrets.token_urlsafe(_TOKEN_BYTES)
        if self._settings.is_hosted:
            job_output_dir = workspace / _HOSTED_OUTPUT_FOLDER_NAME
            archive_path: Path | None = workspace / archive_name
        else:
            job_output_dir = Path(output_dir)
            archive_path = None

        # Admission: check the limits and reserve the slot in one step.
        now = self._clock()
        job = _Job(
            job_id=secrets.token_urlsafe(_TOKEN_BYTES),
            session_token=session_token,
            key_hash=job_key_hash,
            workspace=workspace,
            archive_path=archive_path,
            started_at=now,
            last_heartbeat_at=now,
        )
        with self._lock:
            self._check_admission_locked(session_token, job_key_hash)
            self._jobs[job.job_id] = job
            replaced_jobs = [
                other
                for other in self._jobs.values()
                if other is not job and other.session_token == session_token
            ]
            for replaced_job in replaced_jobs:
                replaced_job.discarded = True
            jobs_to_delete = [
                replaced_job
                for replaced_job in replaced_jobs
                if self._claim_workspace_deletion_locked(replaced_job, now)
            ]
            for replaced_job in replaced_jobs:
                self._forget_if_released_locked(replaced_job)
        for replaced_job in jobs_to_delete:
            self._delete_workspace(replaced_job)

        # Request, workspace, and worker process. The request is pickled once.
        # The channel thread drops the bytes after it wrote them to the worker,
        # so the manager then holds no key. The workspace is readable only by
        # the server account, and has the marker that the start-up clean-up
        # looks for. The worker command and environment contain no key.
        try:
            request_holder = [
                pickle.dumps(
                    JobRequest(
                        points=points,
                        run_configs=run_configs,
                        input_crs=input_crs,
                        output_dir=job_output_dir,
                        archive_path=archive_path,
                        run_log_name=run_log_name,
                        credentials_json=key_text,
                    )
                )
            ]
            workspace.mkdir(mode=0o700)
            (workspace / _WORKSPACE_MARKER_NAME).touch()
            with open(workspace / _WORKER_STDERR_NAME, "wb") as stderr_file:
                process = subprocess.Popen(
                    self._worker_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    cwd=workspace,
                    env=self._worker_environment(workspace),
                )
        # The log line has no traceback: a log record that keeps the traceback
        # would keep this frame, and with it the pickled request.
        except OSError as error:
            logger.error("Job %s: could not start the worker process: %s", job.log_label, error)
            self._end_failed_start(job)
            return job.job_id
        # Broad catch: a request that cannot be pickled is a programming error.
        # End the job, so that it does not keep its slot, and raise the error.
        except Exception:
            self._end_failed_start(job)
            raise

        # Channel thread: it sends the request and reads the messages.
        channel_thread = threading.Thread(
            target=self._run_channel,
            args=(job, process, request_holder),
            name=f"envoi-webapp-job-{job.log_label}",
            daemon=True,
        )
        with self._lock:
            job.process = process
            job.channel_thread = channel_thread
            stopped_before_start = job.state.is_final
        channel_thread.start()

        # A cancel or a limit can come between the reservation and the start of
        # the process. It set the final state, but it had no process to stop.
        if stopped_before_start:
            _stop_process(process)
        return job.job_id

    def snapshot(self, job_id: str) -> JobSnapshot | None:
        """Return the status of a job, and record the call as the job's heartbeat.

        In hosted mode, the manager stops a running job when no session asked
        for its status during the abandon time-out.

        For a succeeded hosted job, ``result["archive"]`` is the archive path
        that the manager chose: a ``done`` message with another archive value
        gives the state FAILED. ``archive_available`` also comes from that path.

        Returns:
            A copy of the job status, or None when the manager does not know
            the job (for example after a server restart or after
            :meth:`discard`).
        """
        now = self._clock()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.discarded:
                return None
            job.last_heartbeat_at = now
            archive_available = (
                job.state is JobState.SUCCEEDED
                and job.archive_path is not None
                and not job.deleting_workspace
                and not job.workspace_deleted
            )
            return JobSnapshot(
                job_id=job.job_id,
                state=job.state,
                started_at=job.started_at,
                ended_at=job.ended_at,
                progress=tuple(job.progress.values()),
                result=job.result,
                error=job.error,
                stop_reason=job.stop_reason,
                archive_available=archive_available,
            )

    def cancel(self, job_id: str) -> bool:
        """Stop a running job. The job gets the state CANCELLED.

        The method returns after the worker process exited (at most about
        5 seconds after ``terminate()``, then ``kill()``). In local mode, the
        files that the job wrote before the cancel stay in the output folder.

        Returns:
            True when the job was running and is now cancelled. False when the
            job is unknown or had already ended.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        return self._stop_job(job, JobState.CANCELLED, _CANCEL_REASON)

    def cancel_for_key(self, key_hash: str) -> bool:
        """Cancel the running jobs of a key, for example after a page reload.

        The caller computes ``key_hash`` with :func:`key_hash` from the key that
        its session uploaded. The hash covers the private key, so a key that
        only has the same ``client_email`` does not match.

        Returns:
            True when at least one running job of the key was cancelled.
        """
        with self._lock:
            key_jobs = [
                job
                for job in self._jobs.values()
                if job.key_hash == key_hash and job.state is JobState.RUNNING
            ]
        cancelled = [
            self._stop_job(job, JobState.CANCELLED, _CANCEL_FOR_KEY_REASON) for job in key_jobs
        ]
        return any(cancelled)

    def discard(self, job_id: str) -> None:
        """Forget a job and delete its workspace ("Clear results").

        A running job is cancelled first. The manager deletes the workspace
        when the worker process has exited. In local mode the files in the
        user's output folder stay.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.discarded = True
        self._stop_job(job, JobState.CANCELLED, _CANCEL_REASON)

        with self._lock:
            delete_now = self._claim_workspace_deletion_locked(job, self._clock())
            self._forget_if_released_locked(job)
        if delete_now:
            self._delete_workspace(job)

    def mark_downloaded(self, job_id: str) -> None:
        """Record the first click on "Download results" of a succeeded job.

        In hosted mode, the manager deletes the results a retention time after
        the first click. Later clicks do not move that time.
        """
        now = self._clock()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.state is JobState.SUCCEEDED and job.downloaded_at is None:
                job.downloaded_at = now

    def run_housekeeping(self) -> None:
        """Run one housekeeping pass.

        In hosted mode, the pass stops (state STOPPED) each running job that
        exceeds the run-time limit, the workspace size limit, or the abandon
        time-out. In both modes, it stops the worker process of a job that has
        not exited 30 seconds after the job got its final state (the state does
        not change), deletes the workspaces that are due (R9), retries
        deletions that failed, and forgets jobs that ended long ago.

        The housekeeping thread calls this at each interval, in hosted mode
        only. Local mode has no housekeeping thread, so there a worker that
        does not exit after its final message keeps running. Tests call this
        method with a fake clock.
        """
        now = self._clock()
        limits = self._limits
        jobs_to_stop: list[tuple[_Job, str]] = []
        jobs_to_measure: list[_Job] = []
        jobs_to_delete: list[_Job] = []
        lingering_jobs: list[_Job] = []
        with self._lock:
            for job in list(self._jobs.values()):
                if job.state.is_final:
                    if (
                        job.process is not None
                        and not job.process_exited
                        and now - job.ended_at >= _FINAL_STATE_EXIT_GRACE_S
                    ):
                        lingering_jobs.append(job)
                    elif self._claim_workspace_deletion_locked(job, now):
                        jobs_to_delete.append(job)
                    elif job.workspace_deleted and now - job.ended_at > _FORGET_ENDED_JOB_AFTER_S:
                        del self._jobs[job.job_id]
                elif limits is not None:
                    if now - job.started_at > limits.max_run_time_s:
                        limit_text = format_minutes(limits.max_run_time_s)
                        jobs_to_stop.append((job, _RUN_TIME_REASON.format(limit=limit_text)))
                    elif now - job.last_heartbeat_at > limits.abandon_timeout_s:
                        limit_text = format_minutes(limits.abandon_timeout_s)
                        jobs_to_stop.append((job, _ABANDON_REASON.format(limit=limit_text)))
                    else:
                        jobs_to_measure.append(job)

        # Walk the workspaces outside the lock. The worker can delete files
        # while the walk runs, so the walk skips files that disappear.
        for job in jobs_to_measure:
            if _folder_size_bytes(job.workspace) > limits.max_workspace_bytes:
                limit_text = format_bytes(limits.max_workspace_bytes)
                jobs_to_stop.append((job, _WORKSPACE_SIZE_REASON.format(limit=limit_text)))

        for job, stop_reason in jobs_to_stop:
            self._stop_job(job, JobState.STOPPED, stop_reason)
        # A worker that does not exit after its final state: stop the process
        # and keep the state. The channel thread then handles the exit.
        for job in lingering_jobs:
            logger.warning(
                "Job %s: the worker process did not exit within %g seconds after the "
                "job ended. Stopping it.",
                job.log_label,
                _FINAL_STATE_EXIT_GRACE_S,
            )
            _stop_process(job.process)
        for job in jobs_to_delete:
            self._delete_workspace(job)

    def shutdown(self) -> None:
        """Stop the housekeeping thread, cancel every running job, and stop every worker.

        The method also stops a worker process whose job already has a final
        state but that has not exited, for example a worker that hangs after
        its ``done`` message. It waits until the channel threads have handled
        the end of their workers, and then deletes the workspaces that are due,
        also those whose deletion failed before. In local mode it then removes
        the manager's workspace root when the root is empty. The manager
        accepts no further use after it.
        """
        self._shutdown_event.set()
        if self._housekeeping_thread is not None:
            self._housekeeping_thread.join()
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            self._stop_job(job, JobState.CANCELLED, _SHUTDOWN_REASON)
            # The worker of a job with a final state can still run. The join
            # below waits until the worker closed its channel, so stop it now.
            with self._lock:
                process = job.process
            if process is not None:
                _stop_process(process)
        for job in jobs:
            if job.channel_thread is not None:
                job.channel_thread.join()

        self._delete_due_workspaces()
        # Best effort: a root that still holds a workspace stays.
        if not self._settings.is_hosted:
            try:
                os.rmdir(self._workspace_root)
            except OSError:
                pass

    # -----------------------------------------------------------------------
    # Admission
    # -----------------------------------------------------------------------

    def _check_admission_locked(self, session_token: str, job_key_hash: str) -> None:
        """Raise JobRejected when a job limit allows no further job. Call it under the lock."""
        running_jobs = [job for job in self._jobs.values() if job.state is JobState.RUNNING]
        if any(job.session_token == session_token for job in running_jobs):
            raise JobRejected(_SESSION_BUSY_MESSAGE, reason="session")

        limits = self._limits
        if limits is None:
            return
        if any(job.key_hash == job_key_hash for job in running_jobs):
            raise JobRejected(
                "An extraction with this service-account key is already running, for "
                "example on a page that you reloaded or closed. That extraction stops "
                f"within {format_minutes(limits.abandon_timeout_s)} after its page closed. "
                "You can also cancel it now.",
                reason="key",
            )
        if len(running_jobs) >= limits.max_concurrent_jobs:
            raise JobRejected(_SERVER_FULL_MESSAGE, reason="server")

        # Disk budget. A job whose worker can still write counts with the full
        # workspace limit, because it can grow to that size. The finished
        # workspaces of this session do not count: the new job deletes them.
        reserved_bytes = 0
        for job in self._jobs.values():
            if job.workspace_deleted:
                continue
            if job.workspace_bytes is None:
                reserved_bytes += limits.max_workspace_bytes
            elif job.session_token != session_token:
                reserved_bytes += job.workspace_bytes
        if reserved_bytes + limits.max_workspace_bytes > limits.disk_budget_bytes:
            raise JobRejected(_DISK_FULL_MESSAGE, reason="disk")

    def _worker_environment(self, workspace: Path) -> dict[str, str]:
        """Return the environment of a worker: the server's, without credential variables.

        ``PYTHONSAFEPATH`` is ``1``. In hosted mode, ``HOME`` is the job
        workspace, so that the worker finds no credential file in the server
        account's home folder. ``TMPDIR``, ``TEMP``, and ``TMP`` are the job
        workspace too, so that temporary files of libraries count toward the
        workspace size limit and are deleted with the workspace. In local mode
        ``HOME`` stays, so that user installations of pip packages stay
        importable, and the temporary folder stays too.
        """
        # Windows variable names are not case-sensitive.
        environment = {
            name: value
            for name, value in os.environ.items()
            if name.upper() not in _CREDENTIAL_VARIABLES
        }
        # "python -m" puts the working folder (the job workspace) first on
        # sys.path, so a module file in the workspace would replace an import
        # of the worker. PYTHONSAFEPATH stops this on Python 3.11 and later.
        # Python 3.10 ignores it: there, only the private workspace root
        # (mode 0o700) keeps other users from putting files in the workspace.
        environment["PYTHONSAFEPATH"] = "1"
        if self._settings.is_hosted:
            environment["HOME"] = str(workspace)
            for variable_name in _TEMPORARY_FOLDER_VARIABLES:
                environment[variable_name] = str(workspace)
        return environment

    # -----------------------------------------------------------------------
    # Job state
    # -----------------------------------------------------------------------

    def _set_final_locked(
        self,
        job: _Job,
        state: JobState,
        *,
        stop_reason: str | None = None,
        result: DoneMessage | None = None,
        error: ErrorMessage | None = None,
    ) -> bool:
        """Give the job its final state, unless it has one. Call it under the lock.

        The first final state wins (plan decision D12).

        Returns:
            True when the job got the state now.
        """
        if job.state.is_final:
            return False
        job.state = state
        job.ended_at = self._clock()
        job.stop_reason = stop_reason
        job.result = result
        job.error = error
        return True

    def _stop_job(self, job: _Job, state: JobState, stop_reason: str) -> bool:
        """Give a running job a final state, then stop its worker process.

        The state is set first, so a ``done`` message that comes during the
        stop does not change it.

        Returns:
            True when the job was running.
        """
        with self._lock:
            state_changed = self._set_final_locked(job, state, stop_reason=stop_reason)
            process = job.process
        if state_changed and process is not None:
            _stop_process(process)
        return state_changed

    def _record_message(self, job: _Job, message: JobMessage) -> None:
        """Apply one message from the worker. Messages after a final state are ignored.

        A ``done`` message whose archive value is not the archive path of the
        request (also an archive when the request asked for none, or none when
        it asked for one) gives FAILED with a general message. The UI then
        never gets a path that the worker chose.
        """
        unexpected_archive = False
        with self._lock:
            if job.state.is_final:
                return
            if message["type"] == "progress":
                segment_key = (
                    message["batch_id"],
                    message["dataset"],
                    message["window_size_m"],
                    message["mode"],
                )
                job.progress[segment_key] = message
            elif message["type"] == "done":
                expected_archive = None if job.archive_path is None else str(job.archive_path)
                if message["archive"] == expected_archive:
                    self._set_final_locked(job, JobState.SUCCEEDED, result=message)
                else:
                    unexpected_archive = True
                    self._set_final_locked(
                        job, JobState.FAILED, error=_manager_error(_UNEXPECTED_RESULT_MESSAGE)
                    )
            else:
                self._set_final_locked(job, JobState.FAILED, error=message)

        # The log line does not show the archive value: the manager logs no
        # content from the message channel.
        if unexpected_archive:
            logger.warning(
                "Job %s: the worker reported an archive path other than the one in its "
                "request. The job failed.",
                job.log_label,
            )

    def _end_failed_start(self, job: _Job) -> None:
        """End a job whose workspace or worker process could not be created."""
        with self._lock:
            self._set_final_locked(
                job, JobState.FAILED, error=_manager_error(_START_FAILED_MESSAGE)
            )
            job.process_exited = True
            delete_now = self._claim_workspace_deletion_locked(job, self._clock())
        if delete_now:
            self._delete_workspace(job)

    # -----------------------------------------------------------------------
    # Channel thread
    # -----------------------------------------------------------------------

    def _run_channel(
        self, job: _Job, process: subprocess.Popen, request_holder: list[bytes]
    ) -> None:
        """Send the request to the worker, read its messages, and handle its exit.

        One thread per job runs this. The request goes through a list that this
        thread empties, because the thread object keeps its arguments until the
        thread ends: the list then no longer holds the key.
        """
        invalid_line_count = 0
        try:
            # Send the request. The worker imports its libraries before it reads
            # stdin, so this write can block for some seconds.
            request_bytes = request_holder.pop()
            try:
                process.stdin.write(request_bytes)
                process.stdin.close()
                request_sent = True
            # BrokenPipeError on POSIX, OSError(EINVAL) on Windows: the worker
            # exited before it read the request.
            except OSError:
                request_sent = False
                # Close the pipe anyway: close() releases the write buffer, which
                # can hold part of the request, and then fails again on its flush.
                try:
                    process.stdin.close()
                except OSError:
                    pass
            del request_bytes
            if not request_sent:
                with self._lock:
                    self._set_final_locked(
                        job, JobState.FAILED, error=_manager_error(_START_FAILED_MESSAGE)
                    )
                _stop_process(process)

            # Read the messages until the worker closes the channel. Lines that
            # are not valid messages are skipped and never logged.
            for line in process.stdout:
                message = parse_message(line)
                if message is None:
                    invalid_line_count += 1
                    continue
                self._record_message(job, message)

        # Broad catch: the thread must always reach the exit handling below,
        # so that no job stays RUNNING and no process is left behind.
        except Exception:
            logger.exception("Job %s: reading the worker messages failed.", job.log_label)
            with self._lock:
                self._set_final_locked(job, JobState.FAILED, error=_manager_error(_CRASH_MESSAGE))
            _stop_process(process)
        finally:
            process.stdout.close()
            process.wait()
            self._handle_worker_exit(job, process, invalid_line_count)

    def _handle_worker_exit(
        self, job: _Job, process: subprocess.Popen, invalid_line_count: int
    ) -> None:
        """Set the final state of a worker that ended without one, and release its workspace."""
        if invalid_line_count:
            logger.warning(
                "Job %s: skipped %d line(s) on the message channel that were not valid messages.",
                job.log_label,
                invalid_line_count,
            )

        # A worker that ended without "done" or "error" crashed. Log the end of
        # its stderr file (the worker redacted it) before the workspace goes.
        with self._lock:
            crashed = self._set_final_locked(
                job, JobState.FAILED, error=_manager_error(_CRASH_MESSAGE)
            )
        if crashed:
            logger.warning(
                "Job %s: the worker process ended without a final message (exit code %s). "
                "End of its stderr file:\n%s",
                job.log_label,
                process.returncode,
                _read_file_tail(job.workspace / _WORKER_STDERR_NAME, _STDERR_TAIL_BYTES),
            )

        with self._lock:
            job.process_exited = True
            delete_now = self._claim_workspace_deletion_locked(job, self._clock())
        if delete_now:
            self._delete_workspace(job)
            return

        # The workspace stays (a succeeded hosted job). Its size counts in the
        # disk budget from now on, instead of the full workspace limit.
        workspace_bytes = _folder_size_bytes(job.workspace)
        with self._lock:
            if not job.workspace_deleted:
                job.workspace_bytes = workspace_bytes

    # -----------------------------------------------------------------------
    # Workspaces
    # -----------------------------------------------------------------------

    def _claim_workspace_deletion_locked(self, job: _Job, now: float) -> bool:
        """Return True when the caller must delete the job's workspace now. Call it under the lock.

        Deletion rules (R9). A workspace is deleted only after its worker exited:

        * local mode: always (it holds only the worker's stderr file),
        * hosted mode, a job that did not succeed: always (no partial results),
        * hosted mode, a succeeded job: after :meth:`discard` or a new job of
          the same session, a retention time after the first download click,
          or at the latest the maximum retention time after the job ended.

        A True result claims the deletion, so that no other thread starts it too.
        """
        if job.workspace_deleted or job.deleting_workspace or not job.process_exited:
            return False
        limits = self._limits
        deletion_due = (
            limits is None
            or job.discarded
            or job.state is not JobState.SUCCEEDED
            or (
                job.downloaded_at is not None
                and now - job.downloaded_at >= limits.retention_after_download_s
            )
            or now - job.ended_at >= limits.max_retention_s
        )
        if deletion_due:
            job.deleting_workspace = True
        return deletion_due

    def _forget_if_released_locked(self, job: _Job) -> None:
        """Forget a discarded job when its workspace is deleted. Call it under the lock."""
        if job.discarded and job.workspace_deleted:
            self._jobs.pop(job.job_id, None)

    def _delete_workspace(self, job: _Job) -> None:
        """Delete the workspace of a job whose worker exited. The caller claimed the deletion."""
        workspace_deleted = False
        for attempt in range(_DELETE_ATTEMPTS):
            if attempt:
                time.sleep(_DELETE_RETRY_WAIT_S)
            shutil.rmtree(job.workspace, ignore_errors=True)
            workspace_deleted = not os.path.lexists(job.workspace)
            if workspace_deleted:
                break

        with self._lock:
            job.deleting_workspace = False
            if workspace_deleted:
                job.workspace_deleted = True
                job.workspace_bytes = 0
                self._forget_if_released_locked(job)
        if not workspace_deleted:
            logger.warning(
                "Job %s: could not delete the job workspace. The manager tries again at the "
                "next housekeeping pass (hosted mode), at the next submission (local mode), "
                "and at shutdown.",
                job.log_label,
            )

    def _delete_due_workspaces(self) -> None:
        """Delete each due, unclaimed workspace, also one whose deletion failed before."""
        now = self._clock()
        with self._lock:
            jobs_to_delete = [
                job
                for job in self._jobs.values()
                if self._claim_workspace_deletion_locked(job, now)
            ]
        for job in jobs_to_delete:
            self._delete_workspace(job)

    def _delete_leftover_workspaces(self) -> None:
        """Delete the job workspaces that an earlier server process left (hosted mode).

        Only folders with the workspace name format and the marker file are
        deleted. Symbolic links are not followed.
        """
        with os.scandir(self._workspace_root) as entries:
            for entry in entries:
                if not _WORKSPACE_NAME_PATTERN.fullmatch(entry.name):
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not os.path.isfile(os.path.join(entry.path, _WORKSPACE_MARKER_NAME)):
                    continue
                shutil.rmtree(entry.path, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Housekeeping thread
    # -----------------------------------------------------------------------

    def _run_housekeeping_loop(self, interval_s: float) -> None:
        """Run a housekeeping pass at each interval until shutdown()."""
        while not self._shutdown_event.wait(interval_s):
            # Broad catch: one failed pass (for example a file-system error)
            # must not end the thread, or no limit would apply any more.
            try:
                self.run_housekeeping()
            except Exception:
                logger.exception("A housekeeping pass of the job manager failed.")


# ---------------------------------------------------------------------------
# The job manager of the server process
# ---------------------------------------------------------------------------

# The one job manager of this server process, created by the first call of
# get_job_manager(). It lives in this module, not in the Streamlit resource
# cache: any browser client can clear that cache for all users, and a new
# manager would not know the running jobs, so the hosted limits would start
# again from zero. The lock makes concurrent first calls create one manager.
_process_job_manager: JobManager | None = None
_process_job_manager_lock = threading.Lock()


def get_job_manager(
    settings_factory: Callable[[], WebappSettings] = load_settings,
) -> JobManager:
    """Return the one job manager of this server process. The first call creates it.

    Every session and every rerun gets the same manager, so all sessions share
    one set of job limits, and a job outlives a rerun of the page. The manager
    lives until the process ends. The web app never shuts it down.

    Args:
        settings_factory: Returns the web-app settings for a new manager. The
            function is called only when no manager exists yet. The default
            reads the environment variables (:func:`load_settings`).

    Returns:
        The job manager of this process.

    Raises:
        ValueError: when the settings are invalid, or when the hosted workspace
            folder is not safe to use. No manager is stored then, so the next
            call tries again.
    """
    global _process_job_manager
    with _process_job_manager_lock:
        if _process_job_manager is None:
            _process_job_manager = JobManager(settings_factory())
        return _process_job_manager


def _reset_job_manager_for_tests() -> None:
    """Shut down and forget the job manager of this process. Tests only."""
    global _process_job_manager
    with _process_job_manager_lock:
        manager = _process_job_manager
        _process_job_manager = None
    if manager is not None:
        manager.shutdown()


# ---------------------------------------------------------------------------
# Process and file helpers
# ---------------------------------------------------------------------------


def _stop_process(process: subprocess.Popen) -> None:
    """Stop a worker process: terminate(), wait up to 5 seconds, then kill()."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        # A process cannot ignore kill(), so this wait ends.
        process.kill()
        process.wait()


def _check_hosted_workspace_root(root: Path) -> None:
    """Raise ValueError when the hosted workspace root is not safe to use (POSIX only).

    The root is not safe when it is a symbolic link or not a folder, belongs to
    another user, or gives any permission to the group or to others. Another
    user could then read the results, or put a module file in a workspace that
    the worker imports. The function changes nothing: a folder that the web app
    did not create keeps its permissions. Windows has no POSIX owner and mode,
    and ``load_settings()`` refuses hosted mode there.

    Raises:
        ValueError: the message names the folder, the problem, and the fix.
    """
    if sys.platform == "win32":
        return
    root_stat = os.lstat(root)
    if stat.S_ISLNK(root_stat.st_mode):
        problem = "is a symbolic link"
    elif not stat.S_ISDIR(root_stat.st_mode):
        problem = "is not a folder"
    elif root_stat.st_uid != os.geteuid():
        problem = "belongs to another user"
    elif stat.S_IMODE(root_stat.st_mode) & 0o077:
        problem = (
            f"gives permissions to the group or to others "
            f"(mode {stat.S_IMODE(root_stat.st_mode):o})"
        )
    else:
        return
    raise ValueError(
        f"The job workspace folder {root} {problem}. The web app uses only a folder that "
        "the server account owns and that only the server account can use (mode 700). "
        f"Delete the folder, or set {WORKSPACE_VARIABLE} to another parent folder. The web "
        "app then creates the folder with the right owner and permissions."
    )


def _folder_size_bytes(folder: Path) -> int:
    """Return the total size of the files in a folder and its subfolders.

    Files and folders that disappear during the walk are skipped. Symbolic
    links are not followed.
    """
    total_bytes = 0
    # os.walk() skips folders that it cannot list (onerror is None).
    for current_folder, _subfolders, file_names in os.walk(folder):
        for file_name in file_names:
            try:
                total_bytes += os.lstat(os.path.join(current_folder, file_name)).st_size
            except OSError:
                continue
    return total_bytes


def _read_file_tail(path: Path, max_bytes: int) -> str:
    """Return the last ``max_bytes`` bytes of a text file, or "" when it cannot be read."""
    try:
        with open(path, "rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(0, file.tell() - max_bytes))
            tail_bytes = file.read()
    except OSError:
        return ""
    return tail_bytes.decode("utf-8", errors="replace")
