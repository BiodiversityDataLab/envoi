from __future__ import annotations

import io
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import threading
import types
import urllib.parse
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import envoi.qc
from envoi import extract, update_catalog
from envoi.progress import ProgressEvent
from envoi_webapp.job_protocol import (
    RUN_LOG_TAIL_LINES,
    JobRequest,
    encode_message,
    parse_message,
)
from envoi_webapp.worker import (
    _STDERR_LIMIT_BYTES,
    _install_exception_hooks,
    _KeyRedactor,
    _RedactingStderr,
    run_job,
)

# Static parts of a fake key. The body lines take the place of the base64 lines
# of a real private key. The text between BEGIN and END must stay shorter than
# 64 characters. Otherwise the gitleaks pre-commit hook reports it as a key.
_FAKE_PRIVATE_KEY = (
    "-----BEGIN PRIVATE KEY-----\nWORKERKEYLINE1\nWORKERKEYLINE2\n-----END PRIVATE KEY-----\n"
)
_FAKE_PRIVATE_KEY_ID = "fake-private-key-id-for-worker-test"
_FAKE_CLIENT_EMAIL = "worker-test@fake-project.iam.gserviceaccount.com"
_FAKE_CERT_URL = (
    "https://www.googleapis.com/robot/v1/metadata/x509/"
    "worker-test%40fake-project.iam.gserviceaccount.com"
)

_RUN_LOG_NAME = "envoi-run-log-20261008T120000Z.txt"

# A run-log record starts with its UTC time.
_RECORD_TIME_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"

# The run configurations that the web app builds for one tabular row and one
# raster row of the local test dataset.
_TABULAR_CONFIG = {
    "batch_id": "extract_01_dem_local",
    "datasets": ["dem_local"],
    "settings": {
        "output_type": "tabular",
        "window_size_m": 100,
        "statistics": ["mean"],
        "output_file_format": "csv",
    },
}
_RASTER_CONFIG = {
    "batch_id": "extract_02_dem_local",
    "datasets": ["dem_local"],
    "settings": {"output_type": "raster", "window_size_m": 200, "resample_m": 10},
}


def _key_text(private_key: str = _FAKE_PRIVATE_KEY, *, with_token_uri: bool = True) -> str:
    """Return a service-account key as JSON text, laid out like a key file from Google."""
    key = {
        "type": "service_account",
        "project_id": "fake-project",
        "private_key_id": _FAKE_PRIVATE_KEY_ID,
        "private_key": private_key,
        "client_email": _FAKE_CLIENT_EMAIL,
        "client_id": "123456789",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_x509_cert_url": _FAKE_CERT_URL,
    }
    if not with_token_uri:
        del key["token_uri"]
    return json.dumps(key, indent=2)


def _key_material(key_text: str) -> list[str]:
    """Return each part of a key that no output may contain.

    The parts are the full key text, the private key, each body line of the
    private key, the private key ID, the client email (also URL-encoded), and
    the certificate URL.
    """
    key = json.loads(key_text)
    private_key = key["private_key"]
    material = [
        key_text,
        private_key,
        key["private_key_id"],
        key["client_email"],
        urllib.parse.quote(key["client_email"], safe=""),
        key["client_x509_cert_url"],
    ]
    material.extend(
        line.strip()
        for line in private_key.splitlines()
        if line.strip() and not line.startswith("-----")
    )
    return material


def _assert_no_key_material(text: str, key_text: str) -> None:
    leaked_parts = [part for part in _key_material(key_text) if part in text]
    assert leaked_parts == []


def _points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "occurrenceID": ["P1", "P2"],
            "decimalLatitude": [62.976878, 62.981296],
            "decimalLongitude": [18.026823, 18.030991],
        }
    )


def _request(
    tmp_path: Path,
    *,
    archive: bool = True,
    key_text: str | None = None,
    run_log_name: str = _RUN_LOG_NAME,
    points: pd.DataFrame | None = None,
) -> JobRequest:
    """Return a request with the folder layout of a hosted job workspace."""
    workspace = tmp_path / "workspace"
    return JobRequest(
        points=_points() if points is None else points,
        run_configs=[_TABULAR_CONFIG, _RASTER_CONFIG],
        input_crs="EPSG:4326",
        output_dir=workspace / "outputs",
        archive_path=workspace / "results.zip" if archive else None,
        run_log_name=run_log_name,
        credentials_json=_key_text() if key_text is None else key_text,
    )


class _Channel:
    """Collect the messages of run_job() as the job manager reads them: encoded, then parsed."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def __call__(self, message: dict) -> None:
        parsed_message = parse_message(encode_message(message))
        assert parsed_message is not None
        self.messages.append(parsed_message)

    @property
    def types(self) -> list[str]:
        return [message["type"] for message in self.messages]


class _FakeInitGee:
    """Record the calls of init_gee(), and optionally log a warning and raise."""

    def __init__(self, *, warning: str | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._warning = warning
        self._error = error

    def __call__(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self._warning is not None:
            logging.getLogger("envoi.auth").warning(self._warning)
        if self._error is not None:
            raise self._error


def _fake_extract_writing_outputs(points, run_configs, *, output_dir, progress_callback, **options):
    """Write the files of one tabular and one raster output, as extract() does."""
    output_dir = Path(output_dir)
    tiles_dir = output_dir / "extract_02_dem" / "dem"
    tiles_dir.mkdir(parents=True)
    (output_dir / "extract_01_dem.csv").write_text(
        "occurrenceID,dem_mean_100m\nP1,12.5\nP2,13.0\n", encoding="utf-8"
    )
    (output_dir / "extract_01_dem_qc.csv").write_text("occurrenceID,in_extent\n", encoding="utf-8")
    (output_dir / "extract_01_dem_metadata.json").write_text("{}", encoding="utf-8")
    (tiles_dir / "P1.tif").write_bytes(b"II*\x00fake tile")
    (output_dir / "extract_02_dem" / "tiles_manifest.csv").write_text("id,path\n", encoding="utf-8")
    (output_dir / "extract_02_dem" / "extract_02_dem_metadata.json").write_text(
        "{}", encoding="utf-8"
    )

    # Adapters can count with numpy integers. The worker must still send plain ints.
    for completed in (1, 2):
        progress_callback(
            ProgressEvent(
                batch_id="extract_01_dem",
                dataset="dem",
                window_size_m=np.int64(100),
                mode="tabular",
                completed=np.int64(completed),
                total=np.int64(2),
                unit="points",
            )
        )
    return {"extract_01_dem": output_dir / "extract_01_dem.csv", "extract_02_dem:dem": tiles_dir}


_FAKE_OUTPUT_FILES = [
    "extract_01_dem.csv",
    "extract_01_dem_metadata.json",
    "extract_01_dem_qc.csv",
    "extract_02_dem/dem/P1.tif",
    "extract_02_dem/extract_02_dem_metadata.json",
    "extract_02_dem/tiles_manifest.csv",
]


def _run_log_records(run_log_path: Path) -> list[str]:
    """Return the lines of a run-log file after its header line."""
    return run_log_path.read_text(encoding="utf-8").splitlines()[1:]


def _files_below(folder: Path) -> list[str]:
    return sorted(
        path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
    )


class TestSuccessfulJob:
    def test_sends_progress_then_done(self, tmp_path):
        """Each progress event becomes a progress message with plain ints, and done comes last."""
        channel = _Channel()

        job_succeeded = run_job(
            _request(tmp_path),
            channel,
            init_gee_func=_FakeInitGee(),
            extract_func=_fake_extract_writing_outputs,
        )

        assert job_succeeded is True
        assert channel.types == ["progress", "progress", "done"]
        assert channel.messages[0] == {
            "type": "progress",
            "batch_id": "extract_01_dem",
            "dataset": "dem",
            "window_size_m": 100,
            "mode": "tabular",
            "completed": 1,
            "total": 2,
            "unit": "points",
        }
        assert channel.messages[1]["completed"] == 2

    def test_done_lists_relative_outputs_archive_and_run_log(self, tmp_path):
        """The done message gives the outputs relative to the output folder, with / separators."""
        request = _request(tmp_path)
        channel = _Channel()

        run_job(
            request,
            channel,
            init_gee_func=_FakeInitGee(),
            extract_func=_fake_extract_writing_outputs,
        )

        assert channel.messages[-1] == {
            "type": "done",
            "outputs": {
                "extract_01_dem": "extract_01_dem.csv",
                "extract_02_dem:dem": "extract_02_dem/dem",
            },
            "archive": str(request.archive_path),
            "warning_count": 0,
            "run_log": _RUN_LOG_NAME,
        }

    def test_archive_holds_outputs_and_run_log_and_empties_output_folder(self, tmp_path):
        """The archive holds the written files and the run log. The output folder is left empty."""
        request = _request(tmp_path)

        run_job(
            request,
            _Channel(),
            init_gee_func=_FakeInitGee(),
            extract_func=_fake_extract_writing_outputs,
        )

        with zipfile.ZipFile(request.archive_path) as archive:
            assert sorted(archive.namelist()) == sorted([*_FAKE_OUTPUT_FILES, _RUN_LOG_NAME])
            assert (
                archive.read("extract_01_dem.csv")
                == b"occurrenceID,dem_mean_100m\nP1,12.5\nP2,13.0\n"
            )
            assert archive.read("extract_02_dem/dem/P1.tif") == b"II*\x00fake tile"
            assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())
        assert list(request.output_dir.iterdir()) == []

    def test_without_archive_outputs_stay_in_output_folder(self, tmp_path):
        """Without an archive path, the outputs and the run log stay in the output folder."""
        request = _request(tmp_path, archive=False)
        channel = _Channel()

        run_job(
            request,
            channel,
            init_gee_func=_FakeInitGee(),
            extract_func=_fake_extract_writing_outputs,
        )

        assert channel.messages[-1]["archive"] is None
        assert _files_below(request.output_dir) == sorted([*_FAKE_OUTPUT_FILES, _RUN_LOG_NAME])
        assert not (tmp_path / "workspace" / "results.zip").exists()

    def test_calls_init_gee_and_extract_with_web_app_arguments(self, tmp_path):
        """init_gee() gets only the key text, and extract() gets the web app's fixed arguments."""
        request = _request(tmp_path)
        fake_init_gee = _FakeInitGee()
        extract_calls = []

        def recording_extract(points, run_configs, **options):
            extract_calls.append({"points": points, "run_configs": run_configs, **options})
            return {}

        run_job(request, _Channel(), init_gee_func=fake_init_gee, extract_func=recording_extract)

        assert fake_init_gee.calls == [{"credentials_json": request.credentials_json}]
        [extract_call] = extract_calls
        assert extract_call["points"] is request.points
        assert extract_call["run_configs"] == [_TABULAR_CONFIG, _RASTER_CONFIG]
        assert extract_call["output_dir"] == request.output_dir
        assert extract_call["input_crs"] == "EPSG:4326"
        assert extract_call["id_column"] == "occurrenceID"
        assert extract_call["latitude_column"] == "decimalLatitude"
        assert extract_call["longitude_column"] == "decimalLongitude"
        assert extract_call["date_column"] == "eventDate"
        assert extract_call["quiet"] is True
        assert callable(extract_call["progress_callback"])

    def test_archive_matches_direct_extract_call(self, tmp_path, dem_tif, sample_df):
        """With the real extract(), the archive holds the files of a direct call plus the run log."""
        update_catalog(
            {"datasets": {"dem_local": {"data_source": "local", "path": str(dem_tif), "bands": 1}}}
        )
        # Two year-only dates give the same envoi warning twice, from one source line.
        points = sample_df.copy()
        points.loc[[0, 1], "eventDate"] = "2002"
        request = _request(tmp_path, points=points)
        channel = _Channel()

        run_job(request, channel, init_gee_func=_FakeInitGee())

        direct_dir = tmp_path / "direct"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct_outputs = extract(
                points,
                [_TABULAR_CONFIG, _RASTER_CONFIG],
                output_dir=direct_dir,
                input_crs="EPSG:4326",
                quiet=True,
            )

        done_message = channel.messages[-1]
        assert done_message["type"] == "done"
        assert done_message["outputs"] == {
            output_key: Path(output_path).relative_to(direct_dir).as_posix()
            for output_key, output_path in direct_outputs.items()
        }
        assert done_message["warning_count"] == 2
        with zipfile.ZipFile(request.archive_path) as archive:
            assert sorted(archive.namelist()) == sorted([*_files_below(direct_dir), _RUN_LOG_NAME])
            assert (
                archive.read("extract_01_dem_local.csv")
                == (direct_dir / "extract_01_dem_local.csv").read_bytes()
            )
            run_log_records = archive.read(_RUN_LOG_NAME).decode("utf-8").splitlines()[1:]
        assert len(run_log_records) == 2
        for record in run_log_records:
            assert re.fullmatch(
                rf"{_RECORD_TIME_PATTERN} UserWarning envoi/extract\.py:\d+: "
                r"Date '2002' interpreted as 2002-01-01\..*",
                record,
            )
        assert list(request.output_dir.iterdir()) == []


class TestRunLog:
    def test_holds_envoi_logger_warnings_and_envoi_python_warnings_only(self, tmp_path):
        """The run log counts envoi warnings. Other records and other warnings stay out."""
        request = _request(tmp_path, archive=False)
        channel = _Channel()

        def extract_with_warnings(points, run_configs, **options):
            logging.getLogger("envoi.adapters.earth_engine.adapter").warning(
                "GEE stats fetch failed for point %d: %s", 3, "quota exceeded"
            )
            logging.getLogger("envoi.catalog").info("An info record is not a warning.")
            logging.getLogger("urllib3").warning("A warning record of another library.")
            # A warning whose source line is in envoi, as Python reports it for
            # warnings.warn(..., stacklevel=2) in envoi code.
            warnings.warn_explicit(
                "Low coverage for 2 sample(s) (<50%).",
                UserWarning,
                filename=envoi.qc.__file__,
                lineno=73,
                module="envoi.qc",
            )
            warnings.warn("A Python warning from another module.", UserWarning, stacklevel=1)
            return {}

        with pytest.warns(UserWarning) as shown_warnings:
            run_job(
                request,
                channel,
                init_gee_func=_FakeInitGee(),
                extract_func=extract_with_warnings,
            )

        # The other module's warning goes on to the earlier warning display.
        # The envoi warning goes only into the run log.
        assert [str(shown.message) for shown in shown_warnings] == [
            "A Python warning from another module."
        ]
        assert channel.messages[-1]["warning_count"] == 2
        records = _run_log_records(request.output_dir / _RUN_LOG_NAME)
        assert len(records) == 2
        assert re.fullmatch(
            rf"{_RECORD_TIME_PATTERN} WARNING envoi\.adapters\.earth_engine\.adapter: "
            r"GEE stats fetch failed for point 3: quota exceeded",
            records[0],
        )
        assert re.fullmatch(
            rf"{_RECORD_TIME_PATTERN} UserWarning envoi/qc\.py:73: "
            r"Low coverage for 2 sample\(s\) \(<50%\)\.",
            records[1],
        )

    def test_records_a_repeated_envoi_warning_each_time(self, tmp_path):
        """The same envoi warning from the same source line is recorded each time it occurs."""
        request = _request(tmp_path, archive=False)
        channel = _Channel()

        def extract_with_repeated_warning(points, run_configs, **options):
            # A shared registry, as one module's __warningregistry__: with
            # Python's default filter, the second warning would not show.
            warning_registry: dict = {}
            for _dataset in ("first dataset", "second dataset"):
                warnings.warn_explicit(
                    "Low coverage for 1 sample(s) (<50%).",
                    UserWarning,
                    filename=envoi.qc.__file__,
                    lineno=73,
                    module="envoi.qc",
                    registry=warning_registry,
                )
            return {}

        run_job(
            request,
            channel,
            init_gee_func=_FakeInitGee(),
            extract_func=extract_with_repeated_warning,
        )

        assert channel.messages[-1]["warning_count"] == 2
        assert len(_run_log_records(request.output_dir / _RUN_LOG_NAME)) == 2

    def test_restores_logging_and_warning_display_after_the_job(self, tmp_path):
        """After the job, the envoi logger has no run-log handler and showwarning is restored."""
        showwarning_before = warnings.showwarning
        handlers_before = list(logging.getLogger("envoi").handlers)

        run_job(
            _request(tmp_path),
            _Channel(),
            init_gee_func=_FakeInitGee(error=RuntimeError("refused")),
        )

        assert warnings.showwarning is showwarning_before
        assert logging.getLogger("envoi").handlers == handlers_before


class TestFailedJob:
    def test_init_gee_failure_sends_error_and_writes_run_log(self, tmp_path):
        """A failed init_gee() ends with an error message, a run log, no extract() call, and no archive."""
        request = _request(tmp_path)
        channel = _Channel()
        extract_calls = []

        job_succeeded = run_job(
            request,
            channel,
            init_gee_func=_FakeInitGee(
                warning="Earth Engine answered slowly.",
                error=RuntimeError("Google Earth Engine authentication failed."),
            ),
            extract_func=lambda *args, **kwargs: extract_calls.append(args),
        )

        assert job_succeeded is False
        assert extract_calls == []
        assert channel.types == ["error"]
        error_message = channel.messages[0]
        assert error_message["message"] == (
            "RuntimeError: Google Earth Engine authentication failed."
        )
        assert error_message["warning_count"] == 1
        [tail_line] = error_message["run_log_tail"]
        assert re.fullmatch(
            rf"{_RECORD_TIME_PATTERN} WARNING envoi\.auth: Earth Engine answered slowly\.",
            tail_line,
        )
        assert _run_log_records(request.output_dir / _RUN_LOG_NAME) == [tail_line]
        assert not request.archive_path.exists()

    def test_extract_failure_sends_error_with_run_log_tail(self, tmp_path):
        """A failed extract() ends with an error message that carries the last run-log lines."""
        request = _request(tmp_path)
        channel = _Channel()
        warning_total = RUN_LOG_TAIL_LINES + 5

        def failing_extract(points, run_configs, *, output_dir, **options):
            Path(output_dir).mkdir(parents=True)
            (Path(output_dir) / "extract_01_dem.csv").write_text("partial\n", encoding="utf-8")
            for point_index in range(warning_total):
                logging.getLogger("envoi.adapters.earth_engine.adapter").warning(
                    "GEE stats fetch failed for point %d: %s", point_index, "timeout"
                )
            raise ValueError("Output 'extract_01_dem': no image covers the sample dates.")

        job_succeeded = run_job(
            request, channel, init_gee_func=_FakeInitGee(), extract_func=failing_extract
        )

        assert job_succeeded is False
        assert channel.types == ["error"]
        error_message = channel.messages[0]
        assert error_message["message"] == (
            "ValueError: Output 'extract_01_dem': no image covers the sample dates."
        )
        assert error_message["warning_count"] == warning_total
        records = _run_log_records(request.output_dir / _RUN_LOG_NAME)
        assert len(records) == warning_total
        assert error_message["run_log_tail"] == records[-RUN_LOG_TAIL_LINES:]
        assert error_message["run_log_tail"][-1].endswith(
            f"GEE stats fetch failed for point {warning_total - 1}: timeout"
        )
        # A failed job gives no archive. The job manager deletes the workspace.
        assert not request.archive_path.exists()

    @pytest.mark.parametrize("run_log_name", ["../escape.txt", "", "run log.txt"])
    def test_unsafe_run_log_name_fails_before_init_gee(self, tmp_path, run_log_name):
        """An unsafe run-log name ends the job with an error before Earth Engine is initialized."""
        request = _request(tmp_path, run_log_name=run_log_name)
        channel = _Channel()
        fake_init_gee = _FakeInitGee()

        job_succeeded = run_job(request, channel, init_gee_func=fake_init_gee)

        assert job_succeeded is False
        assert fake_init_gee.calls == []
        assert channel.types == ["error"]
        assert "The run-log file name" in channel.messages[0]["message"]
        assert _files_below(tmp_path) == []

    def test_key_material_is_redacted_from_error_and_run_log(self, tmp_path):
        """Key material in an error and in a warning reaches neither the message nor the run log."""
        key_text = _key_text()
        request = _request(tmp_path, key_text=key_text)
        channel = _Channel()
        # An error text like the one google-auth gives for a malformed key: it
        # repeats the parsed key, private key included.
        leaking_error = ValueError(
            f"Could not deserialize key data: {json.loads(key_text)!r}. Raw key: {key_text}"
        )
        fake_init_gee = _FakeInitGee(
            warning=f"Key of {_FAKE_CLIENT_EMAIL} ({_FAKE_PRIVATE_KEY_ID}):\n{_FAKE_PRIVATE_KEY}",
            error=leaking_error,
        )

        run_job(request, channel, init_gee_func=fake_init_gee)

        error_message = channel.messages[-1]
        run_log_text = (request.output_dir / _RUN_LOG_NAME).read_text(encoding="utf-8")
        assert error_message["type"] == "error"
        assert "[redacted]" in error_message["message"]
        assert "[redacted]" in run_log_text
        for text in [error_message["message"], *error_message["run_log_tail"], run_log_text]:
            _assert_no_key_material(text, key_text)


@pytest.fixture(scope="module")
def generated_private_key() -> str:
    """A new 2048-bit RSA private key in PKCS#8 PEM, the format of Google's keys."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


def _run_worker_process(stdin_bytes: bytes, cwd: Path) -> subprocess.CompletedProcess:
    """Run the real worker module, as the job manager starts it, without credential variables."""
    worker_environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"ENVOI_EE_CREDENTIALS", "GOOGLE_APPLICATION_CREDENTIALS"}
    }
    return subprocess.run(
        [sys.executable, "-m", "envoi_webapp.worker"],
        input=stdin_bytes,
        capture_output=True,
        cwd=cwd,
        env=worker_environment,
        timeout=300,
        check=False,
    )


class TestExceptionHooks:
    """Ignored exceptions and bad log calls write no traceback in the worker."""

    @pytest.fixture(autouse=True)
    def _restore_process_hooks(self, monkeypatch):
        # The hooks are process-wide. monkeypatch puts the originals back after each test.
        monkeypatch.setattr(sys, "excepthook", sys.excepthook)
        monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)
        monkeypatch.setattr(threading, "excepthook", threading.excepthook)
        monkeypatch.setattr(logging, "raiseExceptions", logging.raiseExceptions)

    @staticmethod
    def _redact(text: str) -> str:
        return text.replace("KEYSECRET", "[redacted]")

    def test_ignored_exception_writes_one_redacted_line(self, capsys):
        """An ignored exception (sys.unraisablehook) writes one redacted line, no traceback."""
        _install_exception_hooks(self._redact)
        unraisable = types.SimpleNamespace(
            exc_type=ValueError,
            exc_value=ValueError("bad KEYSECRET\nsecond line"),
            exc_traceback=None,
            err_msg=None,
            object=None,
        )

        sys.unraisablehook(unraisable)

        assert capsys.readouterr().err == (
            "envoi worker: uncaught ValueError: bad [redacted] second line\n"
        )

    def test_bad_log_call_prints_no_traceback(self, capsys):
        """A log call whose arguments do not fit its format prints no traceback."""
        _install_exception_hooks(self._redact)
        logger = logging.getLogger("envoi.test_worker_hooks")
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
        try:
            # Two placeholders and one argument: formatting the record fails on purpose.
            logger.warning("two values: %s %s", "only one")  # noqa: PLE1206
        finally:
            logger.removeHandler(handler)

        assert logging.raiseExceptions is False
        assert "Traceback" not in capsys.readouterr().err

    def test_uncaught_exception_writes_one_redacted_line(self, capsys):
        """An uncaught exception (sys.excepthook) writes one line without key material."""
        key_text = _key_text()
        _install_exception_hooks(_KeyRedactor(key_text))
        try:
            raise ValueError(f"Could not parse the key:\n{key_text}")
        except ValueError as error:
            caught_error = error

        sys.excepthook(type(caught_error), caught_error, caught_error.__traceback__)

        stderr_text = capsys.readouterr().err
        assert stderr_text.startswith(
            "envoi worker: uncaught ValueError: Could not parse the key: "
        )
        assert stderr_text.count("\n") == 1
        assert stderr_text.endswith("\n")
        assert "[redacted]" in stderr_text
        assert "Traceback" not in stderr_text
        _assert_no_key_material(stderr_text, key_text)

    def test_uncaught_exception_in_a_thread_writes_one_redacted_line(self, capsys):
        """An uncaught exception in a thread (threading.excepthook) writes one redacted line."""
        key_text = _key_text()
        _install_exception_hooks(_KeyRedactor(key_text))

        def fail_with_key_material() -> None:
            raise RuntimeError(f"Key {_FAKE_PRIVATE_KEY_ID} of {_FAKE_CLIENT_EMAIL} failed.")

        thread = threading.Thread(target=fail_with_key_material)
        thread.start()
        thread.join()

        stderr_text = capsys.readouterr().err
        assert (
            stderr_text
            == "envoi worker: uncaught RuntimeError: Key [redacted] of [redacted] failed.\n"
        )


class TestRedactingStderr:
    def test_removes_each_kind_of_key_material(self):
        """Each write loses the full key, the base64 lines, the ID, the email, and the URL."""
        key_text = _key_text()
        target = io.StringIO()
        redacting_stderr = _RedactingStderr(target, _KeyRedactor(key_text))
        quoted_email = urllib.parse.quote(_FAKE_CLIENT_EMAIL, safe="")

        for text in [
            f"Full key: {key_text}\n",
            "One base64 line: WORKERKEYLINE2\n",
            f"Email: {_FAKE_CLIENT_EMAIL}\n",
            f"Key ID: {_FAKE_PRIVATE_KEY_ID}\n",
            f"Encoded email: {quoted_email}\n",
            f"Certificate URL: {_FAKE_CERT_URL}\n",
        ]:
            assert redacting_stderr.write(text) == len(text)

        written_text = target.getvalue()
        assert written_text.count("[redacted]") >= 6
        _assert_no_key_material(written_text, key_text)
        assert "Email: [redacted]\n" in written_text

    def test_stops_after_the_size_limit_with_one_notice(self):
        """After 1 MB the writer adds one notice and drops all later output."""
        target = io.StringIO()
        redacting_stderr = _RedactingStderr(target, lambda text: text)

        redacting_stderr.write("a" * (_STDERR_LIMIT_BYTES - 3))
        redacting_stderr.write("bcdefg")
        redacting_stderr.write("dropped later output\n")

        written_text = target.getvalue()
        notice = "\nenvoi worker: stderr reached its size limit. Later output is dropped.\n"
        assert written_text == "a" * (_STDERR_LIMIT_BYTES - 3) + "bcd" + notice
        assert _STDERR_LIMIT_BYTES == 1_000_000


class TestWorkerProcess:
    def test_invalid_key_ends_with_error_and_no_key_material(self, tmp_path, generated_private_key):
        """A real worker with a key without token_uri sends an error, exits with 1, and leaks no key."""
        # Without token_uri, ee.ServiceAccountCredentials gives an error text
        # that contains the whole key. The check fails before any network call.
        key_text = _key_text(generated_private_key, with_token_uri=False)
        request = _request(tmp_path, key_text=key_text)

        completed = _run_worker_process(pickle.dumps(request), cwd=tmp_path)

        stdout_text = completed.stdout.decode("ascii")
        stderr_text = completed.stderr.decode("utf-8", errors="replace")
        messages = [parse_message(line) for line in stdout_text.splitlines()]
        assert completed.returncode == 1, stderr_text
        assert len(messages) == 1
        error_message = messages[0]
        assert error_message is not None
        assert error_message["type"] == "error"
        assert "not a valid Google service account key" in error_message["message"]
        run_log_path = request.output_dir / _RUN_LOG_NAME
        assert run_log_path.is_file()
        assert not request.archive_path.exists()

        checked_texts = [stdout_text, error_message["message"], *error_message["run_log_tail"]]
        checked_texts.append(stderr_text)
        checked_texts.extend(
            path.read_text(encoding="utf-8", errors="replace")
            for path in tmp_path.rglob("*")
            if path.is_file()
        )
        for text in checked_texts:
            _assert_no_key_material(text, key_text)

    def test_unreadable_request_ends_with_error(self, tmp_path):
        """A worker that cannot unpickle its request sends an error message and exits with 1."""
        completed = _run_worker_process(b"this is not a pickled job request", cwd=tmp_path)

        messages = [parse_message(line) for line in completed.stdout.decode("ascii").splitlines()]
        assert completed.returncode == 1
        assert messages == [
            {
                "type": "error",
                "message": "The worker process could not start the job (UnpicklingError).",
                "warning_count": 0,
                "run_log_tail": [],
            }
        ]
