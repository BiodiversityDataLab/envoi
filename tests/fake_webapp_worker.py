"""Fake worker process for the job manager tests in ``tests/test_webapp_jobs.py``.

The job manager starts this script as ``[sys.executable, <this file>]`` instead
of the real worker. Like the real worker, it reads one pickled ``JobRequest``
from stdin and writes JSON lines on stdout. It makes no Earth Engine calls.

The test puts an instruction in ``request.run_configs[0]["fake_worker"]``:

* ``report``: when set, the path of a JSON file, outside the job workspace,
  where the worker writes what it received: the key's ``client_email`` and the
  Python type of the key in the request, its command line, its environment,
  its working folder, its temporary folder (``tempfile.gettempdir()``),
  ``sys.flags.safe_path`` (None before Python 3.11), the paths of the request,
  and its process ID. The file appears only after the request was read.
* ``import_envoi``: when true, the report also says whether ``import envoi``
  works.
* ``write_file``: when set, ``{"path": <path relative to the working folder>,
  "size_bytes": <n>}``. The worker writes a file of that size before it acts.
* ``action``: what the worker does after the report:

  * ``done``: send a ``done`` message, exit with 0. In hosted mode, it first
    writes ``archive_bytes`` bytes (default 10) to the archive path. When
    ``archive_override`` is in the instruction, the message gives its value
    (a path text or None) as the archive, instead of the request's path.
  * ``error``: send an ``error`` message, exit with 1.
  * ``wait``: send a ``progress`` message, then wait until the file ``release``
    exists (at most 60 seconds), then act as ``done``. Tests stop it, or
    create the file.
  * ``done_then_wait``: send a ``done`` message, then keep running (and keep
    stdout open) until the file ``release`` exists, at most 60 seconds.
  * ``crash``: write a line to stderr, exit with 3 without a message.
  * ``not_json``: write a line that is not JSON, then act as ``done``.
  * ``progress``: send each message of ``messages`` as is, then act as ``done``.
  * ``ignore_terminate``: POSIX only. Catch SIGTERM: send a ``done`` message on
    the first one when ``done_on_terminate`` is true, and go on waiting.
"""

from __future__ import annotations

import json
import os
import pickle
import signal
import sys
import tempfile
import time
from pathlib import Path

_CREDENTIAL_VARIABLES = {"ENVOI_EE_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS"}

# A test that forgets to stop a waiting worker still ends: the worker gives up.
_MAX_WAIT_S = 60.0


def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _progress_message() -> dict:
    return {
        "type": "progress",
        "batch_id": "extract_01_fake",
        "dataset": "fake_dataset",
        "window_size_m": 100,
        "mode": "tabular",
        "completed": 1,
        "total": 2,
        "unit": "points",
    }


def _send_done(request, instruction: dict) -> int:
    archive = None
    if request.archive_path is not None:
        Path(request.archive_path).write_bytes(b"A" * instruction.get("archive_bytes", 10))
        archive = str(request.archive_path)
    if "archive_override" in instruction:
        archive = instruction["archive_override"]
    _send(
        {
            "type": "done",
            "outputs": {"extract_01_fake": "extract_01_fake.csv"},
            "archive": archive,
            "warning_count": 1,
            "run_log": request.run_log_name,
        }
    )
    return 0


def _write_report(request, instruction: dict) -> None:
    key = json.loads(request.credentials_json)
    report = {
        "client_email": key["client_email"],
        "credentials_json_type": type(request.credentials_json).__name__,
        "safe_path": getattr(sys.flags, "safe_path", None),
        "argv": sys.argv,
        "environment": dict(os.environ),
        "credential_variables": sorted(
            name for name in os.environ if name.upper() in _CREDENTIAL_VARIABLES
        ),
        "home": os.environ.get("HOME"),
        "cwd": os.getcwd(),
        "temporary_folder": tempfile.gettempdir(),
        "output_dir": str(request.output_dir),
        "archive_path": None if request.archive_path is None else str(request.archive_path),
        "run_log_name": request.run_log_name,
        "pid": os.getpid(),
    }
    if instruction.get("import_envoi"):
        try:
            import envoi  # noqa: F401

            report["import_envoi"] = True
        except Exception:
            report["import_envoi"] = False

    # Write to a temporary name, then rename, so the test never reads half a file.
    report_path = Path(instruction["report"])
    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(report), encoding="utf-8")
    os.replace(temporary_path, report_path)


def _wait_for_release(instruction: dict) -> None:
    release_path = Path(instruction["release"]) if instruction.get("release") else None
    deadline = time.monotonic() + _MAX_WAIT_S
    while time.monotonic() < deadline:
        if release_path is not None and release_path.exists():
            return
        time.sleep(0.02)


def main() -> int:
    request = pickle.load(sys.stdin.buffer)
    instruction = request.run_configs[0]["fake_worker"]
    action = instruction["action"]

    # Install the handler before the report exists: the test stops the worker
    # as soon as it sees the report.
    if action == "ignore_terminate":
        done_sent = []

        def handle_terminate(signal_number, frame):
            if instruction.get("done_on_terminate") and not done_sent:
                done_sent.append(True)
                _send_done(request, instruction)

        signal.signal(signal.SIGTERM, handle_terminate)

    if instruction.get("write_file"):
        file_path = Path(instruction["write_file"]["path"])
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"F" * instruction["write_file"]["size_bytes"])
    if instruction.get("report"):
        _write_report(request, instruction)

    if action == "done":
        return _send_done(request, instruction)
    if action == "error":
        _send(
            {
                "type": "error",
                "message": "Fake failure.",
                "warning_count": 2,
                "run_log_tail": ["WARNING fake run-log line"],
            }
        )
        return 1
    if action == "wait":
        _send(_progress_message())
        _wait_for_release(instruction)
        return _send_done(request, instruction)
    if action == "done_then_wait":
        _send_done(request, instruction)
        _wait_for_release(instruction)
        return 0
    if action == "crash":
        sys.stderr.write("fake worker crashed on purpose\n")
        sys.stderr.flush()
        return 3
    if action == "not_json":
        sys.stdout.write("this line is not JSON\n")
        return _send_done(request, instruction)
    if action == "progress":
        for message in instruction["messages"]:
            _send(message)
        return _send_done(request, instruction)
    if action == "ignore_terminate":
        _send(_progress_message())
        _wait_for_release(instruction)
        return 0
    raise ValueError(f"Unknown fake worker action: {action!r}")


if __name__ == "__main__":
    sys.exit(main())
