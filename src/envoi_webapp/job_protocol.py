"""Data that the web app's job manager and its worker process exchange.

The web app runs each extraction job in a separate worker process
(``python -m envoi_webapp.worker``). The job manager in the Streamlit server
starts that process and reads its replies. This module is the one definition
of the data that crosses the process boundary, so both sides import the same
classes:

* :class:`JobRequest`: the job. The job manager pickles it once to the
  worker's stdin. It holds the service-account key.
* Messages: the worker's replies, one JSON object per line on its message
  channel. The message types are ``progress``, ``done``, and ``error``.
  :func:`encode_message` writes one line and :func:`parse_message` reads one.
* :class:`JobState` and :class:`JobSnapshot`: the job status that the job
  manager gives to the UI.

The module starts no processes, reads or writes no files, and makes no Earth
Engine calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd

# The ``error`` message carries at most this many lines from the end of the
# run log. In hosted mode the web app deletes a failed job's workspace, so
# these lines are all that the user sees of the run log.
RUN_LOG_TAIL_LINES = 20


# ---------------------------------------------------------------------------
# Request: job manager -> worker
# ---------------------------------------------------------------------------


# eq=False: DataFrame comparison is element-wise, so a generated __eq__ would
# raise "truth value of a DataFrame is ambiguous" instead of returning a bool.
@dataclass(frozen=True, eq=False)
class JobRequest:
    """One extraction job, as the job manager sends it to the worker process.

    The job manager pickles the request to the worker's stdin once, and then
    drops its reference to it. ``repr()`` does not show ``credentials_json``,
    so a log line or a traceback that shows the request does not show the key.
    Pickle is safe here because both ends run the same envoi installation, and
    the worker never unpickles data that a user supplied.
    """

    # Input points with the Darwin Core column names that the web app requires.
    points: pd.DataFrame
    # Run configurations for envoi.extract(), one per data-product row.
    run_configs: list[dict[str, Any]]
    # CRS of the input coordinates, for example "EPSG:4326".
    input_crs: str
    # Absolute folder that extract() writes to.
    output_dir: Path
    # Where the worker writes the ZIP archive of output_dir. None: no archive.
    archive_path: Path | None
    # File name of the run log that the worker writes into output_dir.
    run_log_name: str
    # Service-account key as JSON text. Excluded from repr() so it never shows.
    credentials_json: str = field(repr=False)


# ---------------------------------------------------------------------------
# Messages: worker -> job manager
# ---------------------------------------------------------------------------


class ProgressMessage(TypedDict):
    """Progress of one segment of the job.

    A segment is one (``batch_id``, ``dataset``, ``window_size_m``, ``mode``)
    unit of work. The fields after ``type`` are the fields of
    :class:`envoi.ProgressEvent`.
    """

    type: Literal["progress"]
    batch_id: str
    dataset: str
    window_size_m: int
    mode: Literal["tabular", "raster"]
    completed: int
    total: int
    unit: str


class DoneMessage(TypedDict):
    """The job succeeded. The worker sends this message last."""

    type: Literal["done"]
    # The return value of extract(): each key with its path as text, relative
    # to JobRequest.output_dir.
    outputs: dict[str, str]
    # Path of the ZIP archive as text, or None when the request asked for none.
    archive: str | None
    # Number of records in the run log.
    warning_count: int
    # File name of the run log in JobRequest.output_dir.
    run_log: str


class ErrorMessage(TypedDict):
    """The job failed. The worker sends this message last."""

    type: Literal["error"]
    # Error text for the user. The worker removes key material before it sends it.
    message: str
    # Number of records in the run log.
    warning_count: int
    # Last lines of the run log (at most RUN_LOG_TAIL_LINES), without key material.
    run_log_tail: list[str]


JobMessage = ProgressMessage | DoneMessage | ErrorMessage

# The fields of each message type after "type", with the Python type of each
# JSON value. A message is valid only with exactly these fields, so a change on
# one side of the channel that the other side does not know fails at once.
_MESSAGE_FIELDS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "progress": {
        "batch_id": str,
        "dataset": str,
        "window_size_m": int,
        "mode": str,
        "completed": int,
        "total": int,
        "unit": str,
    },
    "done": {
        "outputs": dict,
        "archive": (str, type(None)),
        "warning_count": int,
        "run_log": str,
    },
    "error": {
        "message": str,
        "warning_count": int,
        "run_log_tail": list,
    },
}


def _is_valid_message(message: Any) -> bool:
    """Return True when ``message`` has exactly the fields of its type, with valid values."""

    if not isinstance(message, dict):
        return False
    message_type = message.get("type")
    if not isinstance(message_type, str) or message_type not in _MESSAGE_FIELDS:
        return False
    expected_fields = _MESSAGE_FIELDS[message_type]
    if set(message) != {"type", *expected_fields}:
        return False

    for field_name, expected_type in expected_fields.items():
        value = message[field_name]
        # JSON true and false parse as bool, and bool is a subclass of int.
        if isinstance(value, bool) or not isinstance(value, expected_type):
            return False

    # Check the values inside the fields that the loop above checks only by type.
    if message_type == "progress":
        return message["mode"] in ("tabular", "raster")
    if message_type == "done":
        return all(
            isinstance(output_key, str) and isinstance(output_path, str)
            for output_key, output_path in message["outputs"].items()
        )
    return all(isinstance(line, str) for line in message["run_log_tail"])


def encode_message(message: JobMessage) -> str:
    """Return ``message`` as one line of ASCII JSON text, with a newline at the end.

    ``ensure_ascii=True`` escapes every non-ASCII character, and JSON escapes
    line breaks inside strings. The result is therefore always exactly one
    line, whatever text encoding the channel uses.

    Raises:
        ValueError: when ``message`` is not a valid ``progress``, ``done``, or
            ``error`` message. The worker then finds a protocol mistake at once,
            instead of sending a line that the job manager skips. The error
            message does not repeat the content of ``message``.
    """

    if not _is_valid_message(message):
        raise ValueError(
            "Not a valid job message. A message needs a 'type' of 'progress', 'done', or "
            "'error', and exactly the fields of that type in envoi_webapp.job_protocol."
        )
    return json.dumps(message, ensure_ascii=True) + "\n"


def parse_message(line: str | bytes) -> JobMessage | None:
    """Return the message in one line of the message channel.

    Returns:
        The message as a dict, or None when the line is not valid JSON or not
        a valid ``progress``, ``done``, or ``error`` message. The caller skips
        such a line.
    """

    try:
        message = json.loads(line)
    # JSONDecodeError and UnicodeDecodeError are both subclasses of ValueError.
    except ValueError:
        return None
    if not _is_valid_message(message):
        return None
    return message


# ---------------------------------------------------------------------------
# Job status: job manager -> UI
# ---------------------------------------------------------------------------


class JobState(Enum):
    """State of a job.

    A job starts as ``RUNNING`` and ends in one final state. The first final
    state wins: when a job already has a final state, the job manager does not
    replace it.
    """

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    # The user clicked "Cancel".
    CANCELLED = "cancelled"
    # The job manager stopped the job at a limit, for example the run time.
    STOPPED = "stopped"

    @property
    def is_final(self) -> bool:
        """True when the job has ended."""
        return self is not JobState.RUNNING


@dataclass(frozen=True)
class JobSnapshot:
    """The status of one job at one moment, as the job manager gives it to the UI.

    A snapshot is a copy. Later messages from the worker do not change it.
    Times are seconds on the job manager's clock (``time.monotonic`` unless a
    test supplies another clock). Compare them only with each other and with
    that clock.
    """

    job_id: str
    state: JobState
    # Time at which the job manager accepted the job.
    started_at: float
    # Time at which the job got its final state. None while it runs.
    ended_at: float | None = None
    # The latest progress message of each segment, in the order in which the
    # segments first reported progress.
    progress: tuple[ProgressMessage, ...] = ()
    # The worker's "done" message. Set only when the state is SUCCEEDED.
    result: DoneMessage | None = None
    # The worker's "error" message. When the worker process ended or could not
    # start without an "error" message, the job manager sets one with a general
    # text and no run-log lines. None when the job did not fail.
    error: ErrorMessage | None = None
    # Text for the user that says why the job manager ended the job (a cancel
    # or a limit). None when the worker ended by itself.
    stop_reason: str | None = None
    # True while the ZIP archive of a succeeded hosted job can be downloaded.
    # False in local mode (no archive), and after the job manager deleted the
    # job workspace at the end of the retention time.
    archive_available: bool = False
