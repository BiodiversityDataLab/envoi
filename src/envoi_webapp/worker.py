"""Worker process that runs one extraction job of the web app.

The web app's job manager starts this module as a separate process for each
job (``python -m envoi_webapp.worker``). Earth Engine keeps its credentials in
one module-global state object, so only a process boundary keeps the key of one
user apart from the keys of other users.

The worker process:

1. reads one pickled :class:`~envoi_webapp.job_protocol.JobRequest` from stdin,
2. initializes Earth Engine with the key in the request,
3. calls :func:`envoi.extract` and reports its progress,
4. writes the run log into the output folder, also when the job fails,
5. moves the files of the output folder into a ZIP archive, when the request
   asks for one,
6. sends one ``done`` or ``error`` message last, and exits with code 0 or 1.

The messages are JSON lines (:func:`~envoi_webapp.job_protocol.encode_message`)
on the message channel: a copy of the original stdout of the process.

Key material must never leave the process. The worker removes it with
:func:`~envoi_webapp.helpers.redact_credential_secrets` from each run-log
record, from the ``error`` message, and from each write to ``sys.stderr``.
An uncaught exception writes one redacted line to stderr, never a traceback.
``progress`` and ``done`` messages carry only names and paths that the job
manager supplied, so the worker does not redact them.

:func:`run_job` holds the job steps without process set-up, so tests can call it
with fake ``init_gee`` and ``extract`` functions. :func:`main` holds the process
set-up: the message channel, the redaction of stderr, and the exit code.
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import sys
import threading
import time
import warnings
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

import envoi
from envoi import extract, init_gee
from envoi._filenames import describe_unsafe_path_component, is_safe_path_component
from envoi.progress import ProgressEvent
from envoi_webapp.helpers import redact_credential_secrets
from envoi_webapp.job_protocol import (
    RUN_LOG_TAIL_LINES,
    JobMessage,
    JobRequest,
    encode_message,
)

# Python warnings whose source file is inside this folder go into the run log.
# Warnings from other libraries (pandas, urllib3, google-auth) go to stderr only,
# so that the warning count shows only warnings about the run.
_ENVOI_PACKAGE_DIR = Path(envoi.__file__).resolve().parent

# Time format of the run-log records. The times are UTC, like the time in the
# run-log file name that the job manager chooses.
_RUN_LOG_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# First line of the run log. It tells a researcher who opens the file what the
# file lists, also when the job had no warnings.
_RUN_LOG_HEADER = (
    "envoi run log: the warnings and errors that envoi reported during this job. Times are UTC."
)

# The worker stops writing to stderr after this many bytes. The job manager
# sends stderr to a file in the job workspace, so a flood of library warnings
# could otherwise fill the disk.
_STDERR_LIMIT_BYTES = 1_000_000


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class _KeyRedactor:
    """Remove key material from text with ``redact_credential_secrets()``.

    Key material is the full key text, the private key and each of its base64
    lines, the ``private_key_id``, and the ``client_email``. Before the redactor
    has a key, it returns text unchanged: :func:`main` installs it on stderr
    before it reads the request, and gives it the key right after.
    """

    def __init__(self, credentials_json: str | None = None) -> None:
        self._raw_credential_bytes: bytes | None = None
        if credentials_json is not None:
            self.set_key(credentials_json)

    def set_key(self, credentials_json: str) -> None:
        """Remove the material of this key from all later text."""
        self._raw_credential_bytes = credentials_json.encode("utf-8")

    def __call__(self, text: str) -> str:
        return redact_credential_secrets(text, self._raw_credential_bytes)


class _RedactingStderr(io.TextIOBase):
    """Text stream for ``sys.stderr`` that removes key material from each write.

    It writes at most ``_STDERR_LIMIT_BYTES`` bytes and drops the rest. Writes to
    file descriptor 2 that do not go through ``sys.stderr`` (for example from a
    C library) are not redacted. This relies on C libraries never printing key
    material. ``fileno()`` is not available, so that Python code cannot bypass
    the redaction through the file descriptor.
    """

    def __init__(self, target: TextIO, redact: Callable[[str], str]) -> None:
        super().__init__()
        self._target = target
        self._redact = redact
        self._remaining_bytes = _STDERR_LIMIT_BYTES
        self._lock = threading.Lock()

    @property
    def encoding(self) -> str:
        return self._target.encoding

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        # Redact the whole text of the write before the size limit cuts it, so
        # that the cut cannot split a secret into parts that no longer match.
        redacted_bytes = self._redact(text).encode("utf-8", errors="replace")
        with self._lock:
            if self._remaining_bytes <= 0:
                return len(text)
            kept_bytes = redacted_bytes[: self._remaining_bytes]
            self._remaining_bytes -= len(kept_bytes)
            self._target.write(kept_bytes.decode("utf-8", errors="ignore"))
            if self._remaining_bytes <= 0:
                self._target.write(
                    "\nenvoi worker: stderr reached its size limit. Later output is dropped.\n"
                )
            self._target.flush()
        return len(text)

    def flush(self) -> None:
        self._target.flush()


def _install_exception_hooks(redact: Callable[[str], str]) -> None:
    """Make uncaught and ignored exceptions write one redacted line to stderr, never a traceback.

    A traceback can show the text of the exception at each level of the chain,
    and the text of an exception from the key parser can contain the key.
    """

    def write_one_line(exception_type: type[BaseException], exception: Any) -> None:
        # str() of an unusual exception can itself fail. The line then names
        # only the exception type.
        try:
            text = redact(f"{exception_type.__name__}: {exception}")
        except Exception:
            text = exception_type.__name__
        sys.stderr.write(f"envoi worker: uncaught {' '.join(text.split())}\n")

    def handle_uncaught(exception_type, exception, exception_traceback) -> None:
        write_one_line(exception_type, exception)

    def handle_uncaught_in_thread(hook_args: threading.ExceptHookArgs) -> None:
        # The default thread hook also ignores SystemExit.
        if hook_args.exc_type is SystemExit:
            return
        write_one_line(hook_args.exc_type, hook_args.exc_value)

    def handle_unraisable(unraisable: Any) -> None:
        # "Exception ignored in ..." output, for example from a __del__ method.
        write_one_line(unraisable.exc_type, unraisable.exc_value)

    sys.excepthook = handle_uncaught
    threading.excepthook = handle_uncaught_in_thread
    sys.unraisablehook = handle_unraisable
    # A log call whose arguments do not match its format string makes logging
    # print a traceback (Handler.handleError). The worker prints none.
    logging.raiseExceptions = False


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------


class _RunLog(logging.Handler):
    """The run log of one job: the records that envoi reports while the job runs.

    It collects two kinds of record, redacted, in the order in which they come:

    * ``WARNING`` and higher records of the ``envoi`` logger (this class is the
      logging handler),
    * Python warnings whose source file is inside the installed ``envoi`` package.

    Records can come from the worker threads of the Earth Engine adapter. The
    records stay in memory until :meth:`write`, so that the run log also exists
    when a step fails before the outputs are written.
    """

    def __init__(self, redact: Callable[[str], str]) -> None:
        super().__init__(level=logging.WARNING)
        self._redact = redact
        self._records: list[str] = []
        self._records_lock = threading.Lock()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt=_RUN_LOG_TIME_FORMAT
        )
        formatter.converter = time.gmtime
        self.setFormatter(formatter)

    @property
    def record_count(self) -> int:
        """Number of records in the run log."""
        with self._records_lock:
            return len(self._records)

    def emit(self, record: logging.LogRecord) -> None:
        """Add one record of the ``envoi`` logger (called by the logging module)."""
        # The standard handler pattern: a record that cannot be formatted goes
        # to handleError(), which writes to the redacted stderr.
        try:
            text = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self._add(text)

    def _add_warning(
        self, message: Warning | str, category: type[Warning], source_path: Path, lineno: int
    ) -> None:
        timestamp = time.strftime(_RUN_LOG_TIME_FORMAT, time.gmtime())
        # A path inside the package ("envoi/extract.py") is enough to find the
        # source, and does not show the folder layout of the server.
        package_path = source_path.relative_to(_ENVOI_PACKAGE_DIR.parent).as_posix()
        self._add(f"{timestamp} {category.__name__} {package_path}:{lineno}: {message}")

    def _add(self, text: str) -> None:
        redacted_text = self._redact(text)
        with self._records_lock:
            self._records.append(redacted_text)

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Collect the run-log records while the ``with`` block runs.

        Python warnings from other libraries go on to the warning display that
        was active before (normally stderr). The block restores the warning
        filters, the warning display, and the ``envoi`` logger at its end.
        """
        envoi_logger = logging.getLogger("envoi")
        envoi_logger.addHandler(self)
        try:
            with warnings.catch_warnings():
                # Show each envoi warning each time it occurs. Python's default
                # filter shows the same warning text from the same source line
                # only once, so the same warning for a second dataset or a
                # second point would be missing from the run log.
                warnings.filterwarnings("always", module=r"envoi(\.|$)")
                show_other_warning = warnings.showwarning

                # Same signature as warnings.showwarning, which this replaces.
                def show_warning(message, category, filename, lineno, file=None, line=None):
                    source_path = Path(filename).resolve()
                    if source_path.is_relative_to(_ENVOI_PACKAGE_DIR):
                        self._add_warning(message, category, source_path, lineno)
                    else:
                        show_other_warning(message, category, filename, lineno, file, line)

                warnings.showwarning = show_warning
                yield
        finally:
            envoi_logger.removeHandler(self)

    def tail(self) -> list[str]:
        """Return the last ``RUN_LOG_TAIL_LINES`` lines of the records.

        A record can have more than one line. The header line of the file is
        not part of the tail.
        """
        with self._records_lock:
            record_lines = "\n".join(self._records).splitlines()
        return record_lines[-RUN_LOG_TAIL_LINES:]

    def write(self, path: Path) -> None:
        """Write the run log as a UTF-8 text file. Create the parent folder when it is missing."""
        with self._records_lock:
            records = list(self._records)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([_RUN_LOG_HEADER, *records]) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def _move_outputs_to_archive(output_dir: Path, archive_path: Path) -> None:
    """Move every file of ``output_dir`` into a new ZIP archive at ``archive_path``.

    The archive names are the paths relative to ``output_dir``, with ``/``
    separators, so the archive has the layout that ``extract()`` wrote. The
    function deletes each file right after it is in the archive. The disk use
    then stays near the size of the outputs, and the workspace size limit of the
    job manager does not stop the job during packaging. Then it removes the empty
    subfolders. ``output_dir`` itself stays, empty.

    ``archive_path`` must not be inside a subfolder of ``output_dir``: that
    subfolder is then not empty, and its removal raises ``OSError``.
    """
    # List the files before the archive exists, so that the archive never
    # contains itself.
    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for output_file in output_files:
            archive.write(output_file, arcname=output_file.relative_to(output_dir).as_posix())
            output_file.unlink()

    # Remove the empty subfolders, the deepest first.
    output_folders = [path for path in output_dir.rglob("*") if path.is_dir()]
    for output_folder in sorted(output_folders, key=lambda path: len(path.parts), reverse=True):
        output_folder.rmdir()


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


def run_job(
    request: JobRequest,
    emit: Callable[[JobMessage], None],
    *,
    init_gee_func: Callable[..., Any] = init_gee,
    extract_func: Callable[..., Any] = extract,
) -> bool:
    """Run one extraction job, and report its progress and its result through ``emit``.

    The function writes the run log ``request.run_log_name`` into
    ``request.output_dir``, also when the job fails. When the job succeeds and
    ``request.archive_path`` is set, it moves every file of ``request.output_dir``
    into that ZIP archive. A failed job gives no archive.

    Args:
        request: The job.
        emit: Sends one message on the message channel. ``run_job`` calls it once
            for each ``progress`` event, and last for one ``done`` or ``error``
            message.
        init_gee_func: Initializes Earth Engine. Called as
            ``init_gee_func(credentials_json=...)``. Tests give a fake.
        extract_func: Runs the extraction, with the arguments of
            :func:`envoi.extract`. Tests give a fake.

    Returns:
        True when the job succeeded and ``done`` was sent. False when the job
        failed and ``error`` was sent.

    Raises:
        Exception: only an exception from ``emit``. Each failure of the job
            itself becomes an ``error`` message, with the key material removed.
    """
    redact = _KeyRedactor(request.credentials_json)
    run_log = _RunLog(redact)
    run_log_path: Path | None = None

    def report_progress(event: ProgressEvent) -> None:
        # int() and str(): encode_message() accepts only plain Python values,
        # and an adapter can count with numpy integers.
        emit(
            {
                "type": "progress",
                "batch_id": str(event.batch_id),
                "dataset": str(event.dataset),
                "window_size_m": int(event.window_size_m),
                "mode": event.mode,
                "completed": int(event.completed),
                "total": int(event.total),
                "unit": str(event.unit),
            }
        )

    try:
        # Check the run-log name before any work. An unsafe name could put the
        # file outside the output folder.
        if not is_safe_path_component(request.run_log_name):
            raise ValueError(
                f"The run-log file name {request.run_log_name!r} cannot be used: "
                f"{describe_unsafe_path_component(request.run_log_name)}."
            )
        run_log_path = request.output_dir / request.run_log_name

        # Initialize Earth Engine with the key of this job, and run the
        # extraction with the arguments that the web app always uses: the
        # Darwin Core column names that it requires, and no progress bars.
        with run_log.capture():
            init_gee_func(credentials_json=request.credentials_json)
            output_paths = extract_func(
                request.points,
                request.run_configs,
                output_dir=request.output_dir,
                input_crs=request.input_crs,
                id_column="occurrenceID",
                latitude_column="decimalLatitude",
                longitude_column="decimalLongitude",
                date_column="eventDate",
                quiet=True,
                progress_callback=report_progress,
            )

        # Give each output path as text relative to the output folder. The web
        # app always writes CSV files, so each value is a path. Do this before
        # the archive step moves the files.
        resolved_output_dir = request.output_dir.resolve()
        relative_outputs = {
            str(output_key): Path(output_path).resolve().relative_to(resolved_output_dir).as_posix()
            for output_key, output_path in output_paths.items()
        }

        # Write the run log next to the outputs, so that the archive contains it.
        run_log.write(run_log_path)

        if request.archive_path is not None:
            _move_outputs_to_archive(request.output_dir, request.archive_path)

    # Broad catch: each failure of the job ends as an "error" message, which is
    # how the job manager learns about it. The traceback is not sent, because
    # the text of a chained exception can contain the key.
    except Exception as error:
        # Write the run log of the failed job too, when its name is safe. If the
        # write fails, the error message still carries the last records.
        if run_log_path is not None:
            try:
                run_log.write(run_log_path)
            except OSError:
                pass
        emit(
            {
                "type": "error",
                "message": redact(f"{type(error).__name__}: {error}"),
                "warning_count": run_log.record_count,
                "run_log_tail": run_log.tail(),
            }
        )
        return False

    emit(
        {
            "type": "done",
            "outputs": relative_outputs,
            "archive": None if request.archive_path is None else str(request.archive_path),
            "warning_count": run_log.record_count,
            "run_log": request.run_log_name,
        }
    )
    return True


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------


def _disable_core_files() -> None:
    """Stop the operating system from writing a core file if the process crashes.

    A core file is a copy of the process memory, so it would contain the key.
    Windows has no ``RLIMIT_CORE`` and no ``resource`` module.
    """
    if sys.platform == "win32":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main() -> int:
    """Run the job that the job manager sends on stdin.

    Returns:
        The exit code of the process: 0 after a ``done`` message, 1 after an
        ``error`` message.
    """
    # Message channel: keep a copy of the original stdout for the JSON lines.
    # Then point file descriptor 1 and sys.stdout to stderr, so that a print()
    # in envoi or a write by a C library cannot put a line on the channel.
    channel_fd = os.dup(1)
    os.dup2(2, 1)
    channel = os.fdopen(channel_fd, "w", encoding="ascii", newline="\n")
    channel_lock = threading.Lock()

    def emit(message: JobMessage) -> None:
        # The lock keeps lines from different threads apart. The flush sends
        # each line at once, so the job manager sees progress as it happens.
        line = encode_message(message)
        with channel_lock:
            channel.write(line)
            channel.flush()

    # Redaction: install it on stderr and for uncaught exceptions before the key
    # enters the process. The redactor gets the key when the request is read.
    key_redactor = _KeyRedactor()
    sys.stderr = _RedactingStderr(sys.stderr, key_redactor)
    sys.stdout = sys.stderr
    _install_exception_hooks(key_redactor)

    # Read the request. On a failure, the worker does not know the key, so the
    # message names only the type of the error.
    try:
        _disable_core_files()
        request = pickle.load(sys.stdin.buffer)
        if not isinstance(request, JobRequest):
            raise TypeError(f"Expected a JobRequest, not {type(request).__name__}.")
        key_redactor.set_key(request.credentials_json)
    except Exception as error:
        emit(
            {
                "type": "error",
                "message": f"The worker process could not start the job ({type(error).__name__}).",
                "warning_count": 0,
                "run_log_tail": [],
            }
        )
        return 1

    # Run the job. run_job() sends the last message.
    job_succeeded = run_job(request, emit)
    return 0 if job_succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
