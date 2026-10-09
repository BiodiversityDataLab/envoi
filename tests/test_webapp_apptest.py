"""Tests of the whole web-app page with Streamlit's ``AppTest``.

Each test runs ``render_app()`` through a two-line app script, as the hosted
container does, with a fake job manager in place of ``get_job_manager()``. No
test starts a worker process or calls Earth Engine: ``ee.Initialize`` is
replaced with a function that fails the test, because the Streamlit server
process must never call it (spec R1).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import ee
import pytest
from streamlit.testing.v1 import AppTest

from envoi_webapp import app
from envoi_webapp.app import (
    JOB_ID_KEY,
    JOB_LOST_MESSAGE,
    JOB_OUTPUT_DIR_KEY,
    LOCAL_INSTALL_URL,
    RESULTS_DELETED_MESSAGE,
)
from envoi_webapp.job_protocol import JobSnapshot, JobState
from envoi_webapp.jobs import JobRejected, key_hash
from envoi_webapp.settings import (
    MODE_VARIABLE,
    WORKSPACE_VARIABLE,
    HostedLimits,
    WebappSettings,
)

# Hosted mode runs only on Linux and macOS (load_settings refuses Windows).
hosted_only = pytest.mark.skipif(
    sys.platform == "win32", reason="Hosted mode does not run on Windows."
)

# AppTest can drive st.file_uploader only from Streamlit 1.56. The app itself
# needs 1.52 (the webapp extra), so older versions skip these tests.
needs_upload_driver = pytest.mark.skipif(
    not hasattr(AppTest, "file_uploader"),
    reason="AppTest.file_uploader needs Streamlit 1.56 or later.",
)

_APP_SCRIPT = "from envoi_webapp.app import render_app\n\nrender_app()\n"
_POINTS_CSV = b"occurrenceID,decimalLatitude,decimalLongitude\na,59.1,18.1\nb,59.2,18.2\n"
_KEY_BYTES = json.dumps(
    {
        "type": "service_account",
        "project_id": "demo",
        "private_key_id": "fake-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nFAKEKEY\n-----END PRIVATE KEY-----\n",
        "client_email": "svc@demo.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
).encode("utf-8")
_JOB_ID = "job-1"


class _FakeJobManager:
    """Stand-in for ``JobManager``: it returns set snapshots and records each call."""

    def __init__(self, snapshot: JobSnapshot | None = None, submit_error: Exception | None = None):
        self.snapshots: dict[str, JobSnapshot] = {} if snapshot is None else {_JOB_ID: snapshot}
        self.submit_error = submit_error
        self.submissions: list[dict] = []
        self.cancelled: list[str] = []
        self.cancelled_key_hashes: list[str] = []
        self.discarded: list[str] = []

    def snapshot(self, job_id: str) -> JobSnapshot | None:
        return self.snapshots.get(job_id)

    def submit(self, **kwargs) -> str:
        self.submissions.append(kwargs)
        if self.submit_error is not None:
            raise self.submit_error
        self.snapshots[_JOB_ID] = JobSnapshot(
            job_id=_JOB_ID, state=JobState.RUNNING, started_at=0.0
        )
        return _JOB_ID

    def cancel(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        self.snapshots[job_id] = replace(
            self.snapshots[job_id],
            state=JobState.CANCELLED,
            stop_reason="You cancelled the extraction.",
        )
        return True

    def cancel_for_key(self, job_key_hash: str) -> bool:
        self.cancelled_key_hashes.append(job_key_hash)
        return True

    def discard(self, job_id: str) -> None:
        self.discarded.append(job_id)
        self.snapshots.pop(job_id, None)

    def mark_downloaded(self, job_id: str) -> None:
        raise AssertionError("AppTest never clicks the download button.")


@pytest.fixture
def run_app(monkeypatch, tmp_path):
    """Return a function that runs the page in a mode, with a fake job manager."""

    def refuse_initialize(*args, **kwargs):
        raise AssertionError("The Streamlit server process must not call ee.Initialize().")

    monkeypatch.setattr(ee, "Initialize", refuse_initialize)
    workspace_parent = tmp_path / "workspace"
    workspace_parent.mkdir()
    monkeypatch.setenv(WORKSPACE_VARIABLE, str(workspace_parent))
    script_path = tmp_path / "app_script.py"
    script_path.write_text(_APP_SCRIPT, encoding="utf-8")

    def run(mode: str, manager: _FakeJobManager, session_state: dict | None = None) -> AppTest:
        monkeypatch.setenv(MODE_VARIABLE, mode)
        monkeypatch.setattr(app, "get_job_manager", lambda: manager)
        app_test = AppTest.from_file(str(script_path), default_timeout=60)
        for key, value in (session_state or {}).items():
            app_test.session_state[key] = value
        return app_test.run()

    return run


def _texts(elements) -> list[str]:
    return [element.value for element in elements]


def _button(app_test: AppTest, label: str):
    return next(button for button in app_test.button if button.label == label)


def _click_and_settle(app_test: AppTest, label: str) -> AppTest:
    """Click a button, then run the page once more.

    A click that calls ``st.rerun()`` reruns the script inside the same
    ``AppTest.run()``, and AppTest then keeps some elements of the first pass.
    The second run gives the page as the browser shows it after the rerun.
    """
    _button(app_test, label).click().run()
    return app_test.run()


def _download_buttons(app_test: AppTest) -> list:
    return list(app_test.get("download_button"))


def _succeeded(*, archive_path, archive_available=True, warning_count=0) -> JobSnapshot:
    return JobSnapshot(
        job_id=_JOB_ID,
        state=JobState.SUCCEEDED,
        started_at=0.0,
        ended_at=1.0,
        result={
            "type": "done",
            "outputs": {"extract_01_dem": "extract_01_dem.csv"},
            "archive": None if archive_path is None else str(archive_path),
            "warning_count": warning_count,
            "run_log": "envoi-run-log-20261008T120000Z.txt",
        },
        archive_available=archive_available,
    )


def _upload_form(app_test: AppTest) -> AppTest:
    """Upload the points and the key, and fill one raster row with a 200 m window."""
    app_test.file_uploader(key="location_csv").upload("points.csv", _POINTS_CSV, "text/csv")
    app_test.file_uploader(key="credentials_json").upload(
        "key.json", _KEY_BYTES, "application/json"
    )
    app_test.selectbox(key="output_type_select_0_0").set_value("raster")
    app_test.run()
    app_test.selectbox(key="dataset_select_0_0").set_value("dem_copernicus_glo30")
    app_test.text_input(key="windows_input_0_0").input("200")
    return app_test.run()


class TestModes:
    def test_local_mode_shows_the_output_folder_controls(self, run_app):
        """Local mode renders the output-folder field and "Browse...", and no hosted notice."""
        app_test = run_app("local", _FakeJobManager())

        assert not app_test.exception
        assert [text_input.label for text_input in app_test.text_input] == ["Output directory"]
        assert any(button.label == "Browse..." for button in app_test.button)
        assert not any(
            "How this service handles your data" in text for text in _texts(app_test.info)
        )

    @hosted_only
    def test_hosted_mode_hides_the_output_folder_and_shows_the_notice(self, run_app):
        """Hosted mode has no widget for a server path, and its notice states the data rules."""
        app_test = run_app("hosted", _FakeJobManager())

        assert not app_test.exception
        assert not any(text_input.label == "Output directory" for text_input in app_test.text_input)
        assert not any(button.label == "Browse..." for button in app_test.button)
        notice = next(
            text for text in _texts(app_test.info) if "How this service handles your data" in text
        )
        assert "never written to disk" in notice
        assert "deleted 10 minutes after the first download" in notice
        assert "at the latest 30 minutes after the extraction ends" in notice
        assert "the extraction stops after 10 minutes" in notice
        assert "A server restart" in notice
        assert "at most 50 MB and 10,000 rows" in notice
        assert LOCAL_INSTALL_URL in notice
        assert "ask where to save each download" in notice

    @needs_upload_driver
    def test_invalid_mode_shows_a_configuration_error_and_no_form(self, run_app):
        """An unknown ENVOI_WEBAPP_MODE stops the page before any form field appears."""
        app_test = run_app("public", _FakeJobManager())

        assert not app_test.exception
        assert "ENVOI_WEBAPP_MODE must be 'local' or 'hosted'" in app_test.error[0].value
        assert not list(app_test.file_uploader)
        assert not list(app_test.text_input)

    def test_job_manager_error_replaces_the_run_section(self, run_app, monkeypatch):
        """When the job manager cannot start, the page shows why and offers no run button."""
        app_test = run_app("local", _FakeJobManager())

        def failing_factory():
            raise ValueError("The workspace folder is not safe.")

        monkeypatch.setattr(app, "get_job_manager", failing_factory)
        app_test.run()

        assert not app_test.exception
        assert _texts(app_test.error) == [
            "The web app cannot run extractions. The workspace folder is not safe."
        ]
        assert not any(button.label == "Extract selected data" for button in app_test.button)


@needs_upload_driver
class TestSubmit:
    def test_local_submit_passes_the_chosen_folder_and_disables_the_run_button(
        self, run_app, tmp_path
    ):
        """A valid local form starts a job that writes to the chosen folder, then shows progress."""
        manager = _FakeJobManager()
        app_test = run_app("local", manager)
        output_dir = tmp_path / "my outputs"
        app_test.text_input(key="output_dir").input(str(output_dir))
        _upload_form(app_test)

        _click_and_settle(app_test, "Extract selected data")

        assert not app_test.exception
        assert len(manager.submissions) == 1
        submission = manager.submissions[0]
        assert submission["output_dir"] == output_dir.resolve()
        assert output_dir.is_dir()
        assert submission["credentials_json"] == _KEY_BYTES
        assert submission["input_crs"] == "EPSG:4326"
        assert list(submission["points"]["occurrenceID"]) == ["a", "b"]
        assert submission["run_configs"] == [
            {
                "batch_id": "extract_01_dem_copernicus_glo30",
                "datasets": ["dem_copernicus_glo30"],
                "settings": {"output_type": "raster", "window_size_m": 200, "resample_m": 10},
            }
        ]
        assert app_test.session_state[JOB_ID_KEY] == _JOB_ID
        assert _button(app_test, "Extract selected data").disabled
        assert any(button.label == "Cancel" for button in app_test.button)
        assert app_test.get("progress")[0].proto.text == "Starting extraction"

    @hosted_only
    def test_hosted_submit_passes_no_output_folder(self, run_app):
        """In hosted mode the job manager chooses the folder, so the page passes none."""
        manager = _FakeJobManager()
        app_test = _upload_form(run_app("hosted", manager))

        _click_and_settle(app_test, "Extract selected data")

        assert not app_test.exception
        assert manager.submissions[0]["output_dir"] is None
        assert app_test.session_state[JOB_ID_KEY] == _JOB_ID

    @hosted_only
    def test_hosted_form_limits_stop_the_submission(self, run_app):
        """A raster window above the hosted limit is a form error, and no job starts."""
        manager = _FakeJobManager()
        app_test = _upload_form(run_app("hosted", manager))
        app_test.text_input(key="windows_input_0_0").input("5000").run()

        _button(app_test, "Extract selected data").click().run()

        assert manager.submissions == []
        assert "Server limit — Data-product row 1 (dem_copernicus_glo30)" in (
            app_test.error[0].value
        )

    @hosted_only
    def test_key_rejection_offers_to_cancel_the_earlier_job(self, run_app):
        """A rejection by key shows the message and a button that cancels the job of that key."""
        rejection = JobRejected("A job with this key runs.", reason="key")
        manager = _FakeJobManager(submit_error=rejection)
        app_test = _upload_form(run_app("hosted", manager))

        _button(app_test, "Extract selected data").click().run()

        assert _texts(app_test.warning) == ["A job with this key runs."]
        _button(app_test, "Cancel the earlier job").click().run()

        assert manager.cancelled_key_hashes == [key_hash(_KEY_BYTES)]
        assert any("The earlier extraction was cancelled" in text for text in _texts(app_test.info))

        # The notice shows once. The rejection and its button are gone.
        app_test.run()
        assert not app_test.warning
        assert not any("The earlier extraction" in text for text in _texts(app_test.info))
        assert not any(button.label == "Cancel the earlier job" for button in app_test.button)

    @hosted_only
    def test_other_rejections_show_only_the_message(self, run_app):
        """A full server gives its message without a cancel button, until the next run click."""
        manager = _FakeJobManager(submit_error=JobRejected("The server is full.", reason="server"))
        app_test = _upload_form(run_app("hosted", manager))

        _button(app_test, "Extract selected data").click().run()
        assert _texts(app_test.warning) == ["The server is full."]
        assert not any(button.label == "Cancel the earlier job" for button in app_test.button)

        app_test.run()
        assert _texts(app_test.warning) == ["The server is full."]

        manager.submit_error = None
        _click_and_settle(app_test, "Extract selected data")
        assert not app_test.warning
        assert app_test.session_state[JOB_ID_KEY] == _JOB_ID


class TestJobStates:
    def test_running_job_shows_progress_and_cancel(self, run_app):
        """A running job shows its progress and "Cancel", which ends in the cancelled state."""
        snapshot = JobSnapshot(
            job_id=_JOB_ID,
            state=JobState.RUNNING,
            started_at=0.0,
            progress=(
                {
                    "type": "progress",
                    "batch_id": "extract_01_dem",
                    "dataset": "dem",
                    "window_size_m": 200,
                    "mode": "raster",
                    "completed": 1,
                    "total": 2,
                    "unit": "points",
                },
            ),
        )
        manager = _FakeJobManager(snapshot)
        app_test = run_app(
            "local",
            manager,
            {
                JOB_ID_KEY: _JOB_ID,
                app.JOB_SEGMENTS_KEY: {("extract_01_dem", "dem", 200, "raster"): 2},
            },
        )

        progress = app_test.get("progress")[0].proto
        assert progress.value == 50
        assert progress.text == "dem | 200 m | 1/2 points"
        assert _button(app_test, "Extract selected data").disabled

        _click_and_settle(app_test, "Cancel")

        assert manager.cancelled == [_JOB_ID]
        assert _texts(app_test.warning) == ["You cancelled the extraction."]
        assert not _button(app_test, "Extract selected data").disabled

    @hosted_only
    def test_hosted_success_keeps_the_download_and_the_warning_count_after_a_rerun(
        self, run_app, tmp_path
    ):
        """The download button, the warning count, and the retention text stay after a rerun."""
        archive_path = tmp_path / "envoi-results-20261008T120000Z.zip"
        archive_path.write_bytes(b"zip")
        manager = _FakeJobManager(_succeeded(archive_path=archive_path, warning_count=1234))
        app_test = run_app("hosted", manager, {JOB_ID_KEY: _JOB_ID})

        for _ in range(2):
            assert not app_test.exception
            download_buttons = _download_buttons(app_test)
            assert len(download_buttons) == 1
            assert download_buttons[0].proto.label == "Download results"
            assert download_buttons[0].proto.ignore_rerun
            assert download_buttons[0].proto.deferred_file_id
            assert "1,234 warnings in the run log." in _texts(app_test.info)
            assert any(
                "Results are deleted 10 minutes after the first download, or 30 minutes after "
                "the job ended." in text
                for text in _texts(app_test.caption)
            )
            app_test.run()

    @hosted_only
    def test_hosted_success_after_retention_shows_that_results_were_deleted(self, run_app):
        """When the archive is gone, the page says so and shows no download button."""
        manager = _FakeJobManager(_succeeded(archive_path="/gone.zip", archive_available=False))
        app_test = run_app("hosted", manager, {JOB_ID_KEY: _JOB_ID})

        assert not _download_buttons(app_test)
        assert RESULTS_DELETED_MESSAGE in _texts(app_test.info)

    def test_local_success_lists_the_output_paths_in_the_chosen_folder(self, run_app, tmp_path):
        """Local mode shows each output and the run log as a path in the chosen output folder."""
        output_dir = tmp_path / "outputs"
        manager = _FakeJobManager(_succeeded(archive_path=None, warning_count=1))
        app_test = run_app("local", manager, {JOB_ID_KEY: _JOB_ID, JOB_OUTPUT_DIR_KEY: output_dir})

        assert not _download_buttons(app_test)
        assert "1 warning in the run log." in _texts(app_test.info)
        assert _texts(app_test.code) == [
            f"extract_01_dem: {output_dir / 'extract_01_dem.csv'}",
            f"run log: {output_dir / 'envoi-run-log-20261008T120000Z.txt'}",
        ]

    def test_failed_job_shows_the_message_and_the_run_log_tail(self, run_app):
        """A failed job shows its message, the warning count, and the last run-log lines."""
        snapshot = JobSnapshot(
            job_id=_JOB_ID,
            state=JobState.FAILED,
            started_at=0.0,
            ended_at=1.0,
            error={
                "type": "error",
                "message": "RuntimeError: Earth Engine refused the key.",
                "warning_count": 2,
                "run_log_tail": ["first warning", "second warning"],
            },
        )
        app_test = run_app("local", _FakeJobManager(snapshot), {JOB_ID_KEY: _JOB_ID})

        assert _texts(app_test.error) == [
            "The extraction failed. RuntimeError: Earth Engine refused the key."
        ]
        assert "2 warnings in the run log." in _texts(app_test.info)
        assert _texts(app_test.code) == ["first warning\nsecond warning"]

    @pytest.mark.parametrize(
        ("state", "reason"),
        [
            (JobState.CANCELLED, "You cancelled the extraction."),
            (JobState.STOPPED, "The extraction ran longer than the limit of 60 minutes."),
        ],
    )
    def test_cancelled_or_stopped_job_shows_only_the_reason(self, run_app, state, reason):
        """A cancelled or stopped job shows its reason, and no warning count or run-log lines."""
        snapshot = JobSnapshot(
            job_id=_JOB_ID, state=state, started_at=0.0, ended_at=1.0, stop_reason=reason
        )
        app_test = run_app("local", _FakeJobManager(snapshot), {JOB_ID_KEY: _JOB_ID})

        assert _texts(app_test.warning) == [reason]
        assert not any("in the run log" in text for text in _texts(app_test.info))
        assert not app_test.code

    def test_unknown_job_shows_that_the_state_is_lost_until_cleared(self, run_app):
        """A job that the manager does not know shows "state is lost" and "Clear results"."""
        manager = _FakeJobManager()
        app_test = run_app("local", manager, {JOB_ID_KEY: "forgotten-job"})

        assert _texts(app_test.warning) == [JOB_LOST_MESSAGE]
        _click_and_settle(app_test, "Clear results")

        assert manager.discarded == ["forgotten-job"]
        assert JOB_ID_KEY not in app_test.session_state
        assert not app_test.warning

    @hosted_only
    def test_clear_results_discards_the_job(self, run_app, tmp_path):
        """ "Clear results" discards a succeeded job and removes the download button."""
        manager = _FakeJobManager(_succeeded(archive_path=tmp_path / "results.zip"))
        app_test = run_app("hosted", manager, {JOB_ID_KEY: _JOB_ID})

        _click_and_settle(app_test, "Clear results")

        assert manager.discarded == [_JOB_ID]
        assert not _download_buttons(app_test)


@needs_upload_driver
class TestUpload:
    @hosted_only
    def test_hosted_upload_above_the_line_limit_is_not_parsed(self, run_app, monkeypatch, tmp_path):
        """A file above the hosted line limit gets the limit message, and pandas never reads it."""
        limits = HostedLimits(max_upload_lines=2)
        monkeypatch.setattr(
            app,
            "load_settings",
            lambda: WebappSettings(mode="hosted", workspace_parent=tmp_path, limits=limits),
        )

        def refuse_parse(source):
            raise AssertionError("A file above the upload limits must not be parsed.")

        monkeypatch.setattr(app, "read_points_csv", refuse_parse)
        app_test = run_app("hosted", _FakeJobManager())

        app_test.file_uploader(key="location_csv").upload("points.csv", _POINTS_CSV, "text/csv")
        app_test.run()

        assert not app_test.exception
        assert "The file has 3 lines. The limit is 2 lines" in app_test.error[0].value
        assert not app_test.dataframe
