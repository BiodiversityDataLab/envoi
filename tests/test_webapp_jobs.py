from __future__ import annotations

import dataclasses
import gc
import json
import logging
import os
import pickle
import re
import stat
import sys
import threading
import time
import types
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest

from envoi._filenames import is_safe_path_component
from envoi_webapp import jobs
from envoi_webapp.job_protocol import JobState
from envoi_webapp.jobs import JobManager, JobRejected, key_hash
from envoi_webapp.settings import HostedLimits, WebappSettings, load_settings

# The job manager starts this script instead of the real worker. Its module
# docstring lists the instructions that a test can give.
_FAKE_WORKER_PATH = Path(__file__).with_name("fake_webapp_worker.py")
_FAKE_WORKER_COMMAND = [sys.executable, str(_FAKE_WORKER_PATH)]

# Real worker processes start in this time, also on slow CI machines.
_WAIT_TIMEOUT_S = 60.0

_WORKSPACE_NAME_PATTERN = r"[A-Za-z0-9_-]{22}"
_IS_POSIX = os.name == "posix"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _key_text(name: str) -> str:
    """Return a fake service-account key whose key material contains ``name``.

    The text between BEGIN and END stays shorter than 64 characters, so that
    the gitleaks pre-commit hook does not report it as a key.
    """
    line_stem = f"JOBS{name.upper()}LINE"
    body = f"{line_stem}1\n{line_stem}2\n"
    private_key = "-----BEGIN PRIVATE KEY-----\n" + body + "-----END PRIVATE KEY-----\n"
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "fake-project",
            "private_key_id": f"{name}-fake-private-key-id",
            "private_key": private_key,
            "client_email": f"{name}@fake-project.iam.gserviceaccount.com",
            "client_id": "123456789",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_x509_cert_url": (
                "https://www.googleapis.com/robot/v1/metadata/x509/"
                f"{name}%40fake-project.iam.gserviceaccount.com"
            ),
        },
        indent=2,
    )


def _key_material(key_text: str) -> list[str]:
    """Return the parts of a key that must not leave the session or the worker.

    The parts are the full key text, the private key and each of its body
    lines, the private key ID, the client email (also URL-encoded), and the
    certificate URL.
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


def _leaked_key_material(text: str | bytes, key_text: str) -> list[str]:
    if isinstance(text, (bytes, bytearray)):
        return [part for part in _key_material(key_text) if part.encode("utf-8") in text]
    return [part for part in _key_material(key_text) if part in text]


# ---------------------------------------------------------------------------
# Manager, jobs, and waiting
# ---------------------------------------------------------------------------


class _FakeClock:
    """A clock that moves only when the test calls advance()."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> _FakeClock:
    return _FakeClock()


@pytest.fixture
def workspace_parent(tmp_path) -> Path:
    """The parent folder of the workspace roots (ENVOI_WEBAPP_WORKSPACE)."""
    return tmp_path / "parent"


@pytest.fixture
def make_manager(workspace_parent, clock):
    """Return a function that creates a job manager with the fake worker.

    Hosted mode by default, with the D8 limits except the ones that the test
    changes. No housekeeping thread unless the test asks: the tests call
    run_housekeeping() with the fake clock. Each manager is shut down at the
    end of the test, so no worker process stays.
    """
    managers: list[JobManager] = []

    def make(
        mode: str = "hosted",
        *,
        worker_command: list[str] | None = None,
        housekeeping_interval_s: float | None = None,
        **limit_changes,
    ) -> JobManager:
        limits = dataclasses.replace(HostedLimits(), **limit_changes) if mode == "hosted" else None
        settings = WebappSettings(mode=mode, workspace_parent=workspace_parent, limits=limits)
        manager = JobManager(
            settings,
            worker_command=_FAKE_WORKER_COMMAND if worker_command is None else worker_command,
            clock=clock,
            housekeeping_interval_s=housekeeping_interval_s,
        )
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.shutdown()


def _points(row_count: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "occurrenceID": [f"P{index}" for index in range(row_count)],
            "decimalLatitude": [59.0] * row_count,
            "decimalLongitude": [18.0] * row_count,
        }
    )


def _submit(
    manager: JobManager,
    *,
    session: str = "session-a",
    key_name: str = "alice",
    key_text: str | bytes | None = None,
    output_dir: Path | None = None,
    points: pd.DataFrame | None = None,
    **instruction,
) -> str:
    """Submit a job whose fake worker follows ``instruction`` (default: send done).

    The key is ``_key_text(key_name)``, or ``key_text`` (text or bytes) when the
    test gives one.
    """
    instruction.setdefault("action", "done")
    return manager.submit(
        session_token=session,
        points=_points() if points is None else points,
        run_configs=[
            {
                "batch_id": "extract_01_fake",
                "datasets": ["fake_dataset"],
                "settings": {},
                "fake_worker": instruction,
            }
        ],
        input_crs="EPSG:4326",
        credentials_json=_key_text(key_name) if key_text is None else key_text,
        output_dir=output_dir,
    )


def _wait_until(condition, description: str) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting until {description}.")


def _read_report(path: Path) -> dict:
    """Wait for the fake worker's report, which it writes after it read the request."""
    _wait_until(path.exists, f"{path.name} exists")
    return json.loads(path.read_text(encoding="utf-8"))


def _job_record(manager: JobManager, job_id: str):
    # Private access: the tests wait for the channel thread and check the
    # worker process, which the public interface does not show.
    return manager._jobs[job_id]


def _wait_for_worker_end(record) -> None:
    """Wait until the channel thread handled the end of the worker (state, reaping, clean-up)."""
    record.channel_thread.join(_WAIT_TIMEOUT_S)
    assert not record.channel_thread.is_alive()


def _assert_reaped(process) -> None:
    """The worker process has exited and the manager collected its exit status (no zombie)."""
    assert process.returncode is not None
    if _IS_POSIX:
        # A process that the manager reaped is no longer a child of this process.
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)


def _ended_snapshot(manager: JobManager, job_id: str):
    record = _job_record(manager, job_id)
    _wait_for_worker_end(record)
    _assert_reaped(record.process)
    return manager.snapshot(job_id)


def _folders(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_dir())


def _reachable_texts(roots: list[object]) -> list[str | bytes]:
    """Return every str and bytes object that the roots reach through containers and instances."""
    skipped_types = (
        types.ModuleType,
        type,
        types.FunctionType,
        types.BuiltinFunctionType,
        types.MethodType,
        types.CodeType,
        types.FrameType,
    )
    texts: list[str | bytes] = []
    seen_ids: set[int] = set()
    pending = list(roots)
    while pending:
        current = pending.pop()
        if id(current) in seen_ids:
            continue
        seen_ids.add(id(current))
        if isinstance(current, (str, bytes, bytearray)):
            texts.append(current)
        elif not isinstance(current, skipped_types):
            pending.extend(gc.get_referents(current))
    return texts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKeyHash:
    def test_same_key_gives_the_same_hash_for_text_and_bytes(self):
        """The hash depends on the key material, not on the text layout or type."""
        alice_key = _key_text("alice")
        alice_hash = key_hash(alice_key)
        compact_key = json.dumps(json.loads(alice_key), separators=(",", ":"), sort_keys=True)

        assert re.fullmatch(r"[0-9a-f]{64}", alice_hash)
        assert key_hash(alice_key.encode("utf-8")) == alice_hash
        assert key_hash(compact_key) == alice_hash
        assert key_hash(_key_text("bob")) != alice_hash
        assert _leaked_key_material(alice_hash, alice_key) == []

    @pytest.mark.parametrize("changed_field", ["private_key", "private_key_id", "client_email"])
    def test_each_key_material_field_changes_the_hash(self, changed_field):
        """A key with the same client_email but another private key is another key."""
        alice_key = json.loads(_key_text("alice"))
        other_key = {**alice_key, changed_field: alice_key[changed_field] + "X"}

        assert key_hash(json.dumps(other_key)) != key_hash(json.dumps(alice_key))

    def test_missing_field_counts_as_empty_text(self):
        """A missing key-material field gives the same hash as an empty one."""
        partial_key = {"type": "service_account", "client_email": "alice@example.com"}
        empty_fields_key = {**partial_key, "private_key_id": "", "private_key": ""}

        assert key_hash(json.dumps(partial_key)) == key_hash(json.dumps(empty_fields_key))

    @pytest.mark.parametrize(
        "credentials_json",
        [
            "not json " + _key_text("alice"),
            _key_text("alice")[:-5],
            json.dumps(["alice@fake-project.iam.gserviceaccount.com"]),
            json.dumps("alice@fake-project.iam.gserviceaccount.com"),
        ],
    )
    def test_text_that_is_not_a_json_object_raises_without_key_material(self, credentials_json):
        """Text that is not a JSON object raises ValueError without chained errors."""
        with pytest.raises(ValueError, match="must be a JSON object") as error_info:
            key_hash(credentials_json)

        assert error_info.value.__context__ is None
        assert error_info.value.__cause__ is None
        assert _leaked_key_material(str(error_info.value), _key_text("alice")) == []


class TestIsolation:
    def test_two_concurrent_jobs_receive_only_their_own_key(self, make_manager, tmp_path):
        """Two running jobs with different keys each get only their own key and workspace (AC1)."""
        manager = make_manager()
        release_path = tmp_path / "release"

        job_a = _submit(
            manager,
            session="session-a",
            key_name="alice",
            action="wait",
            release=str(release_path),
            report=str(tmp_path / "report-a.json"),
        )
        job_b = _submit(
            manager,
            session="session-b",
            key_name="bob",
            action="wait",
            release=str(release_path),
            report=str(tmp_path / "report-b.json"),
        )
        report_a = _read_report(tmp_path / "report-a.json")
        report_b = _read_report(tmp_path / "report-b.json")

        assert report_a["client_email"] == "alice@fake-project.iam.gserviceaccount.com"
        assert report_b["client_email"] == "bob@fake-project.iam.gserviceaccount.com"
        assert _leaked_key_material(json.dumps(report_a), _key_text("bob")) == []
        assert _leaked_key_material(json.dumps(report_b), _key_text("alice")) == []
        workspace_a = Path(report_a["cwd"]).resolve()
        workspace_b = Path(report_b["cwd"]).resolve()
        assert workspace_a != workspace_b
        assert workspace_a.parent == workspace_b.parent == manager._workspace_root.resolve()
        assert re.fullmatch(_WORKSPACE_NAME_PATTERN, workspace_a.name)
        assert re.fullmatch(_WORKSPACE_NAME_PATTERN, workspace_b.name)

        release_path.touch()
        assert _ended_snapshot(manager, job_a).state is JobState.SUCCEEDED
        assert _ended_snapshot(manager, job_b).state is JobState.SUCCEEDED

    def test_hosted_worker_gets_workspace_paths_and_no_key_outside_the_request(
        self, make_manager, tmp_path, monkeypatch
    ):
        """The worker command line, environment, and workspace files hold no key (AC3, R3)."""
        monkeypatch.setenv("ENVOI_EE_CREDENTIALS", str(tmp_path / "server-key.json"))
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "server-adc.json"))
        manager = make_manager()
        key_text = _key_text("alice")

        job_id = _submit(manager, report=str(tmp_path / "report.json"), action="done")
        report = _read_report(tmp_path / "report.json")
        snapshot = _ended_snapshot(manager, job_id)

        # The worker has no credential variable, and HOME and cwd are the workspace.
        assert report["credential_variables"] == []
        workspace = Path(report["home"])
        assert workspace.resolve() == Path(report["cwd"]).resolve()
        assert workspace.parent == manager._workspace_root
        assert (workspace / ".envoi-webapp-job").is_file()
        # The temporary folder is the workspace too, so temporary files count
        # toward the size limit and are deleted with the workspace.
        for variable_name in ("TMPDIR", "TEMP", "TMP", "CPL_TMPDIR"):
            assert report["environment"][variable_name] == str(workspace)
        assert Path(report["temporary_folder"]).resolve() == workspace.resolve()

        # The request names the outputs folder and an archive outside it.
        assert Path(report["output_dir"]) == workspace / "outputs"
        archive_path = Path(report["archive_path"])
        assert archive_path.parent == workspace
        assert re.fullmatch(r"envoi-results-\d{8}T\d{6}Z\.zip", archive_path.name)
        assert re.fullmatch(r"envoi-run-log-\d{8}T\d{6}Z\.txt", report["run_log_name"])
        assert is_safe_path_component(report["run_log_name"])
        assert snapshot.state is JobState.SUCCEEDED
        assert snapshot.result["archive"] == str(archive_path)
        assert snapshot.archive_available is True

        # No key material in the command line, the environment, or the workspace files.
        checked_texts = [*report["argv"], *report["environment"], *report["environment"].values()]
        checked_texts.extend(path.read_bytes() for path in workspace.rglob("*") if path.is_file())
        for text in checked_texts:
            assert _leaked_key_material(text, key_text) == []

    @pytest.mark.parametrize("mode", ["local", "hosted"])
    def test_worker_does_not_import_from_its_working_folder(
        self, make_manager, tmp_path, monkeypatch, mode
    ):
        """The worker environment has PYTHONSAFEPATH=1, so the workspace is not on sys.path (R6)."""
        monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
        manager = make_manager(mode)

        job_id = _submit(
            manager,
            output_dir=tmp_path / "user-outputs" if mode == "local" else None,
            report=str(tmp_path / "report.json"),
        )
        report = _read_report(tmp_path / "report.json")

        assert report["environment"]["PYTHONSAFEPATH"] == "1"
        # Python 3.10 ignores the variable. There, the private root protects the workspace.
        if sys.version_info >= (3, 11):
            assert report["safe_path"] is True
        assert _ended_snapshot(manager, job_id).state is JobState.SUCCEEDED

    def test_manager_keeps_no_key_after_the_worker_read_the_request(self, make_manager, tmp_path):
        """No object that the manager or its channel thread holds contains key material (R4)."""
        manager = make_manager()
        key_text = _key_text("alice")
        job_id = _submit(
            manager,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        _read_report(tmp_path / "report.json")
        channel_thread = _job_record(manager, job_id).channel_thread

        def reachable_texts() -> list[str | bytes]:
            # The manager, and the local variables of each frame of the channel
            # thread. The walk does not enter modules, classes, or functions.
            roots: list[object] = [manager]
            frame = sys._current_frames().get(channel_thread.ident)
            while frame is not None:
                roots.append(dict(frame.f_locals))
                frame = frame.f_back
            return _reachable_texts(roots)

        def holds_key() -> bool:
            return any(_leaked_key_material(text, key_text) for text in reachable_texts())

        # The walk finds key material when an object holds it.
        assert any(_leaked_key_material(text, key_text) for text in _reachable_texts([[key_text]]))
        _wait_until(lambda: not holds_key(), "the manager dropped the request")
        assert _job_record(manager, job_id).state is JobState.RUNNING

        (tmp_path / "release").touch()
        assert _ended_snapshot(manager, job_id).state is JobState.SUCCEEDED
        assert not holds_key()


class TestLocalMode:
    def test_local_worker_keeps_home_and_writes_to_the_user_folder(self, make_manager, tmp_path):
        """Local mode keeps HOME and TMPDIR, imports envoi, deletes only the workspace (R3, R10)."""
        manager = make_manager("local")
        output_dir = tmp_path / "user-outputs"
        output_dir.mkdir()
        (output_dir / "earlier-file.txt").write_text("keep", encoding="utf-8")

        job_id = _submit(
            manager,
            output_dir=output_dir,
            report=str(tmp_path / "report.json"),
            import_envoi=True,
            write_file={"path": str(output_dir / "result.csv"), "size_bytes": 5},
        )
        report = _read_report(tmp_path / "report.json")
        snapshot = _ended_snapshot(manager, job_id)

        assert report["home"] == os.environ.get("HOME")
        for variable_name in ("TMPDIR", "TEMP", "TMP", "CPL_TMPDIR"):
            assert report["environment"].get(variable_name) == os.environ.get(variable_name)
        assert report["import_envoi"] is True
        workspace = Path(report["cwd"])
        assert workspace.resolve().parent == manager._workspace_root.resolve()
        assert report["output_dir"] == str(output_dir)
        assert report["archive_path"] is None
        assert snapshot.state is JobState.SUCCEEDED
        assert snapshot.archive_available is False
        # The workspace held only the stderr file and is gone. The user's files stay.
        assert not workspace.exists()
        assert (output_dir / "earlier-file.txt").is_file()
        assert (output_dir / "result.csv").is_file()

    def test_local_cancel_keeps_the_files_written_before_it(self, make_manager, tmp_path):
        """Cancel stops a local job. The files that it wrote stay in the output folder (R10)."""
        manager = make_manager("local")
        output_dir = tmp_path / "user-outputs"
        output_dir.mkdir()
        job_id = _submit(
            manager,
            output_dir=output_dir,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
            write_file={"path": str(output_dir / "partial.csv"), "size_bytes": 5},
        )
        report = _read_report(tmp_path / "report.json")

        assert manager.cancel(job_id) is True
        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.CANCELLED
        assert snapshot.stop_reason == "You cancelled the extraction."
        assert not Path(report["cwd"]).exists()
        assert (output_dir / "partial.csv").is_file()

    def test_local_mode_applies_no_key_server_or_time_limit(self, make_manager, tmp_path, clock):
        """Local mode runs two jobs with one key, and no time-out stops them (R12)."""
        manager = make_manager("local")
        output_dir = tmp_path / "user-outputs"
        release_path = tmp_path / "release"
        job_ids = [
            _submit(
                manager,
                session=session,
                key_name="alice",
                output_dir=output_dir,
                action="wait",
                release=str(release_path),
                report=str(tmp_path / f"{session}.json"),
                write_file={"path": "big.bin", "size_bytes": 5_000},
            )
            for session in ("session-a", "session-b", "session-c")
        ]
        for session in ("session-a", "session-b", "session-c"):
            _read_report(tmp_path / f"{session}.json")

        clock.advance(10 * 24 * 60 * 60)
        manager.run_housekeeping()

        assert [manager.snapshot(job_id).state for job_id in job_ids] == [JobState.RUNNING] * 3
        release_path.touch()
        for job_id in job_ids:
            assert _ended_snapshot(manager, job_id).state is JobState.SUCCEEDED

    def test_local_mode_needs_an_absolute_output_folder(self, make_manager):
        """A local job without an absolute output folder is a programming error."""
        manager = make_manager("local")

        with pytest.raises(ValueError, match="absolute folder path"):
            _submit(manager, output_dir=None)
        with pytest.raises(ValueError, match="absolute folder path"):
            _submit(manager, output_dir=Path("relative-folder"))

    def test_hosted_mode_refuses_an_output_folder(self, make_manager, tmp_path):
        """In hosted mode the manager chooses the output folder."""
        manager = make_manager()

        with pytest.raises(ValueError, match="chooses the output folder"):
            _submit(manager, output_dir=tmp_path)

    @pytest.mark.parametrize("retry_at", ["submit", "shutdown"])
    def test_failed_deletion_is_tried_again(self, make_manager, tmp_path, monkeypatch, retry_at):
        """A local workspace that was not deleted is deleted at the next submit or at shutdown."""
        monkeypatch.setattr(jobs, "_DELETE_ATTEMPTS", 1)
        delete_folder = jobs.shutil.rmtree
        failures_left = [1]

        def rmtree_that_fails_once(path, ignore_errors=False):
            # Like rmtree(ignore_errors=True) when a file is locked: no error, no deletion.
            if failures_left[0]:
                failures_left[0] -= 1
                return
            delete_folder(path, ignore_errors=ignore_errors)

        monkeypatch.setattr(jobs.shutil, "rmtree", rmtree_that_fails_once)
        manager = make_manager("local")
        output_dir = tmp_path / "user-outputs"
        job_id = _submit(manager, output_dir=output_dir, report=str(tmp_path / "report.json"))
        workspace = Path(_read_report(tmp_path / "report.json")["cwd"])
        assert _ended_snapshot(manager, job_id).state is JobState.SUCCEEDED
        assert workspace.is_dir()

        if retry_at == "submit":
            _submit(manager, session="session-b", key_name="bob", output_dir=output_dir)
            assert not workspace.exists()
        else:
            manager.shutdown()
            assert not workspace.exists()
            assert not manager._workspace_root.exists()


class TestKeyInput:
    def test_key_as_bytes_reaches_the_worker_as_text(self, make_manager, tmp_path):
        """A key given as UTF-8 bytes reaches the worker as text, and has the hash of the text."""
        manager = make_manager()
        key_text = _key_text("alice")

        job_id = _submit(
            manager,
            key_text=key_text.encode("utf-8"),
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        report = _read_report(tmp_path / "report.json")

        assert report["client_email"] == "alice@fake-project.iam.gserviceaccount.com"
        assert report["credentials_json_type"] == "str"
        assert manager.cancel_for_key(key_hash(key_text)) is True
        assert _ended_snapshot(manager, job_id).state is JobState.CANCELLED

    def test_key_bytes_that_are_not_utf8_raise_without_key_material(self, make_manager):
        """Key bytes that are not UTF-8 raise ValueError without chained errors or key material."""
        manager = make_manager()
        key_text = _key_text("alice")

        with pytest.raises(ValueError, match="must be UTF-8 text") as error_info:
            _submit(manager, key_text=b"\xff" + key_text.encode("utf-8"))

        assert error_info.value.__context__ is None
        assert error_info.value.__cause__ is None
        assert _leaked_key_material(str(error_info.value), key_text) == []
        assert manager._jobs == {}
        assert _folders(manager._workspace_root) == []


class TestWorkspaceRoot:
    def test_local_roots_are_private_and_unique_per_manager(self, make_manager, workspace_parent):
        """Each local manager gets a new root (mode 0o700). shutdown() removes it when empty."""
        first_manager = make_manager("local")
        second_manager = make_manager("local")
        roots = [first_manager._workspace_root, second_manager._workspace_root]

        assert roots[0] != roots[1]
        for root in roots:
            assert root.parent == workspace_parent
            assert root.name.startswith("envoi-webapp-jobs-")
            assert root.is_dir()
            if _IS_POSIX:
                assert stat.S_IMODE(os.lstat(root).st_mode) == 0o700
        assert not (workspace_parent / "envoi-webapp-jobs").exists()

        first_manager.shutdown()
        assert not roots[0].exists()
        assert roots[1].is_dir()

    def test_hosted_root_is_the_fixed_private_subfolder(self, make_manager, workspace_parent):
        """Hosted mode uses <parent>/envoi-webapp-jobs with mode 0o700, and keeps it at shutdown."""
        manager = make_manager()
        root = workspace_parent / "envoi-webapp-jobs"

        assert manager._workspace_root == root
        if _IS_POSIX:
            assert stat.S_IMODE(os.lstat(root).st_mode) == 0o700
        manager.shutdown()
        assert root.is_dir()

    @pytest.mark.skipif(not _IS_POSIX, reason="The owner and mode check runs on POSIX only.")
    @pytest.mark.parametrize(
        ("problem", "expected_message"),
        [
            ("mode_777", "gives permissions to the group or to others"),
            ("mode_750", "gives permissions to the group or to others"),
            ("symbolic_link", "is a symbolic link"),
            ("other_owner", "belongs to another user"),
        ],
    )
    def test_hosted_mode_refuses_an_unsafe_root(
        self, make_manager, workspace_parent, monkeypatch, problem, expected_message
    ):
        """Hosted mode refuses a root that is a symlink, has another owner, or is open (R6)."""
        root = workspace_parent / "envoi-webapp-jobs"
        workspace_parent.mkdir()
        if problem == "symbolic_link":
            target = workspace_parent / "target"
            target.mkdir(mode=0o700)
            root.symlink_to(target, target_is_directory=True)
        else:
            root.mkdir(mode=0o700)
        if problem == "mode_777":
            os.chmod(root, 0o777)
        elif problem == "mode_750":
            os.chmod(root, 0o750)
        elif problem == "other_owner":
            server_uid = os.geteuid()
            monkeypatch.setattr(os, "geteuid", lambda: server_uid + 1)
        mode_before = os.lstat(root).st_mode

        with pytest.raises(ValueError, match=expected_message) as error_info:
            make_manager()

        message = str(error_info.value)
        assert str(root) in message
        assert "ENVOI_WEBAPP_WORKSPACE" in message
        # The manager does not change a folder that it did not create.
        assert os.lstat(root).st_mode == mode_before


class TestAdmission:
    @pytest.mark.parametrize("mode", ["local", "hosted"])
    def test_one_running_job_per_session(self, make_manager, tmp_path, mode):
        """A session with a running job cannot start a second one, in both modes."""
        manager = make_manager(mode)
        output_dir = tmp_path / "user-outputs" if mode == "local" else None
        job_id = _submit(
            manager,
            output_dir=output_dir,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        _read_report(tmp_path / "report.json")
        workspaces_before = _folders(manager._workspace_root)

        with pytest.raises(JobRejected, match="still running") as error_info:
            _submit(manager, output_dir=output_dir, key_name="bob")

        assert error_info.value.reason == "session"

        assert _folders(manager._workspace_root) == workspaces_before
        assert manager.cancel(job_id) is True

    def test_one_running_job_per_key(self, make_manager, tmp_path):
        """A second session with the same key is rejected and told that it can cancel."""
        manager = make_manager()
        job_id = _submit(
            manager,
            session="session-a",
            key_name="alice",
            action="wait",
            release=str(tmp_path / "release"),
        )
        workspaces_before = _folders(manager._workspace_root)

        with pytest.raises(JobRejected) as error_info:
            _submit(manager, session="session-b", key_name="alice")

        message = str(error_info.value)
        assert error_info.value.reason == "key"
        assert "already running" in message
        assert "within 10 minutes after its page closed" in message
        assert "cancel it now" in message
        assert _leaked_key_material(message, _key_text("alice")) == []
        assert _folders(manager._workspace_root) == workspaces_before
        # Another key is not affected.
        assert _submit(manager, session="session-c", key_name="carol")
        assert manager.cancel(job_id) is True

    def test_maximum_concurrent_jobs(self, make_manager, tmp_path):
        """The server runs at most max_concurrent_jobs jobs. A job that ended frees its slot."""
        manager = make_manager(max_concurrent_jobs=2)
        release_path = tmp_path / "release"
        running_ids = [
            _submit(manager, session=name, key_name=name, action="wait", release=str(release_path))
            for name in ("alice", "bob")
        ]
        workspaces_before = _folders(manager._workspace_root)

        with pytest.raises(JobRejected, match="maximum number of extractions") as error_info:
            _submit(manager, session="carol", key_name="carol")
        assert error_info.value.reason == "server"
        assert _folders(manager._workspace_root) == workspaces_before

        assert manager.cancel(running_ids[0]) is True
        assert _submit(manager, session="carol", key_name="carol", action="error")
        manager.cancel(running_ids[1])

    @pytest.mark.parametrize(("disk_budget_bytes", "admitted_count"), [(2_000, 2), (1_999, 1)])
    def test_quick_submissions_reserve_the_disk_budget(
        self, make_manager, tmp_path, disk_budget_bytes, admitted_count
    ):
        """Each admitted job reserves the full workspace limit at once (AC4, D8)."""
        manager = make_manager(
            max_workspace_bytes=1_000, disk_budget_bytes=disk_budget_bytes, max_concurrent_jobs=5
        )
        start_barrier = threading.Barrier(2)
        outcomes: dict[str, object] = {}

        def submit(name: str) -> None:
            start_barrier.wait()
            try:
                outcomes[name] = _submit(
                    manager,
                    session=name,
                    key_name=name,
                    action="wait",
                    release=str(tmp_path / "release"),
                )
            except JobRejected as rejection:
                outcomes[name] = rejection

        threads = [threading.Thread(target=submit, args=(name,)) for name in ("alice", "bob")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        job_ids = [outcome for outcome in outcomes.values() if isinstance(outcome, str)]
        rejections = [outcome for outcome in outcomes.values() if isinstance(outcome, JobRejected)]
        assert len(job_ids) == admitted_count
        assert all("disk space" in str(rejection) for rejection in rejections)
        assert all(rejection.reason == "disk" for rejection in rejections)
        for job_id in job_ids:
            manager.cancel(job_id)

    def test_disk_budget_counts_finished_workspaces_of_other_sessions(self, make_manager):
        """A kept result counts with its size. The same session's new job deletes it instead."""
        manager = make_manager(
            max_workspace_bytes=1_000, disk_budget_bytes=1_500, max_concurrent_jobs=5
        )
        finished_id = _submit(manager, session="session-a", key_name="alice", archive_bytes=600)
        assert _ended_snapshot(manager, finished_id).archive_available is True

        # 600 bytes kept + 1,000 bytes for the new job > 1,500 bytes.
        with pytest.raises(JobRejected, match="disk space"):
            _submit(manager, session="session-b", key_name="bob")

        new_id = _submit(manager, session="session-a", key_name="alice")
        assert manager.snapshot(finished_id) is None
        assert _ended_snapshot(manager, new_id).state is JobState.SUCCEEDED


class TestStops:
    @pytest.mark.parametrize("mode", ["local", "hosted"])
    def test_cancel_stops_the_worker(self, make_manager, tmp_path, mode):
        """Cancel ends the worker, gives CANCELLED, and deletes the workspace (AC5)."""
        manager = make_manager(mode)
        job_id = _submit(
            manager,
            output_dir=tmp_path / "user-outputs" if mode == "local" else None,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        report = _read_report(tmp_path / "report.json")

        assert manager.cancel(job_id) is True
        assert manager.cancel(job_id) is False
        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.CANCELLED
        assert snapshot.stop_reason == "You cancelled the extraction."
        assert snapshot.result is None
        assert snapshot.ended_at is not None
        assert not Path(report["cwd"]).exists()

    def test_run_time_limit_stops_the_job(self, make_manager, tmp_path, clock):
        """A job that runs longer than the run-time limit gets STOPPED (AC5)."""
        manager = make_manager(max_run_time_s=120)
        job_id = _submit(
            manager,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        report = _read_report(tmp_path / "report.json")

        clock.advance(120)
        manager.run_housekeeping()
        assert manager.snapshot(job_id).state is JobState.RUNNING

        clock.advance(1)
        manager.run_housekeeping()
        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.STOPPED
        assert "longer than the limit of 2 minutes" in snapshot.stop_reason
        assert not Path(report["cwd"]).exists()

    def test_abandon_time_out_stops_the_job(self, make_manager, tmp_path, clock):
        """A job without a status request during the abandon time-out gets STOPPED (AC5)."""
        manager = make_manager(abandon_timeout_s=60)
        job_id = _submit(
            manager,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        report = _read_report(tmp_path / "report.json")

        # A status request restarts the time-out.
        clock.advance(50)
        manager.snapshot(job_id)
        clock.advance(60)
        manager.run_housekeeping()
        assert _job_record(manager, job_id).state is JobState.RUNNING

        clock.advance(1)
        manager.run_housekeeping()
        record = _job_record(manager, job_id)
        _wait_for_worker_end(record)
        _assert_reaped(record.process)
        snapshot = manager.snapshot(job_id)

        assert snapshot.state is JobState.STOPPED
        assert "did not ask for the job status for 1 minute" in snapshot.stop_reason
        assert not Path(report["cwd"]).exists()

    def test_workspace_size_limit_stops_the_job(self, make_manager, tmp_path):
        """A workspace larger than its limit stops the job (AC5). The limit itself is allowed."""
        manager = make_manager(max_workspace_bytes=10_000)
        job_id = _submit(
            manager,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
            write_file={"path": "outputs/tiles/big.tif", "size_bytes": 5_000},
        )
        workspace = Path(_read_report(tmp_path / "report.json")["cwd"])

        # Fill the workspace up to exactly the limit.
        current_bytes = sum(path.stat().st_size for path in workspace.rglob("*") if path.is_file())
        filler_path = workspace / "outputs" / "filler.bin"
        filler_path.write_bytes(b"x" * (10_000 - current_bytes))
        manager.run_housekeeping()
        assert manager.snapshot(job_id).state is JobState.RUNNING

        with open(filler_path, "ab") as filler:
            filler.write(b"x")
        manager.run_housekeeping()
        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.STOPPED
        assert "larger than the limit of" in snapshot.stop_reason
        assert not workspace.exists()

    @pytest.mark.skipif(not _IS_POSIX, reason="terminate() on Windows cannot be caught.")
    def test_first_final_state_wins_and_kill_follows_terminate(
        self, make_manager, tmp_path, monkeypatch
    ):
        """A done message after a cancel keeps CANCELLED. kill() ends a stuck worker (D12)."""
        monkeypatch.setattr(jobs, "_STOP_WAIT_S", 0.5)
        manager = make_manager()
        job_id = _submit(
            manager,
            action="ignore_terminate",
            done_on_terminate=True,
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        report = _read_report(tmp_path / "report.json")
        process = _job_record(manager, job_id).process

        assert manager.cancel(job_id) is True
        snapshot = _ended_snapshot(manager, job_id)

        assert process.returncode == -9
        assert snapshot.state is JobState.CANCELLED
        assert snapshot.result is None
        assert not Path(report["cwd"]).exists()

    def test_cancel_during_the_start_stops_the_new_worker(self, make_manager, monkeypatch):
        """A cancel between the reservation and the process start still stops the worker."""
        manager = make_manager()
        start_process = jobs.subprocess.Popen

        def cancel_then_start(*args, **kwargs):
            for job_id in list(manager._jobs):
                assert manager.cancel(job_id) is True
            return start_process(*args, **kwargs)

        monkeypatch.setattr(jobs.subprocess, "Popen", cancel_then_start)
        job_id = _submit(manager, action="wait")
        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.CANCELLED
        assert _folders(manager._workspace_root) == []

    def test_cancel_for_key_cancels_only_that_key(self, make_manager, tmp_path):
        """cancel_for_key stops the running job of one key and leaves other jobs (R12)."""
        manager = make_manager()
        release_path = tmp_path / "release"
        alice_job = _submit(
            manager, session="session-a", key_name="alice", action="wait", release=str(release_path)
        )
        bob_job = _submit(
            manager, session="session-b", key_name="bob", action="wait", release=str(release_path)
        )

        assert manager.cancel_for_key(key_hash(_key_text("alice"))) is True
        alice_snapshot = _ended_snapshot(manager, alice_job)

        assert alice_snapshot.state is JobState.CANCELLED
        assert "same service-account key" in alice_snapshot.stop_reason
        assert manager.snapshot(bob_job).state is JobState.RUNNING
        assert manager.cancel_for_key(key_hash(_key_text("alice"))) is False
        assert manager.cancel_for_key(key_hash(_key_text("carol"))) is False
        # The key slot is free again.
        assert _submit(manager, session="session-c", key_name="alice")
        release_path.touch()
        assert _ended_snapshot(manager, bob_job).state is JobState.SUCCEEDED

    def test_forged_key_with_the_same_client_email_cannot_cancel_or_block(
        self, make_manager, tmp_path
    ):
        """Only the same key material matches a running job, not the same client_email."""
        manager = make_manager()
        release_path = tmp_path / "release"
        alice_job = _submit(
            manager, session="session-a", key_name="alice", action="wait", release=str(release_path)
        )
        forged_key = json.loads(_key_text("alice"))
        forged_key["private_key"] = forged_key["private_key"].replace("LINE1", "FORGED")
        forged_key_text = json.dumps(forged_key)

        # The forged key cancels nothing.
        assert manager.cancel_for_key(key_hash(forged_key_text)) is False
        assert manager.snapshot(alice_job).state is JobState.RUNNING

        # The forged key does not hit the per-key limit of the real key.
        forged_job = _submit(
            manager,
            session="session-b",
            key_text=forged_key_text,
            action="wait",
            release=str(release_path),
        )
        assert manager.snapshot(forged_job).state is JobState.RUNNING
        assert manager.cancel(forged_job) is True
        assert manager.snapshot(alice_job).state is JobState.RUNNING

        # The real key, uploaded again in another session, can cancel the job.
        assert manager.cancel_for_key(key_hash(_key_text("alice"))) is True
        assert _ended_snapshot(manager, alice_job).state is JobState.CANCELLED

    def test_worker_that_does_not_exit_after_done_is_stopped(self, make_manager, tmp_path, clock):
        """A worker that runs on 30 s after its done message is stopped. SUCCEEDED stays."""
        manager = make_manager()
        job_id = _submit(manager, action="done_then_wait", release=str(tmp_path / "release"))
        _wait_until(
            lambda: manager.snapshot(job_id).state is JobState.SUCCEEDED, "the job succeeded"
        )
        process = _job_record(manager, job_id).process

        clock.advance(29)
        manager.run_housekeeping()
        assert process.poll() is None

        clock.advance(1)
        manager.run_housekeeping()
        snapshot = _ended_snapshot(manager, job_id)

        assert process.returncode != 0
        assert snapshot.state is JobState.SUCCEEDED
        assert snapshot.stop_reason is None
        assert snapshot.archive_available is True

    @pytest.mark.parametrize("mode", ["local", "hosted"])
    def test_shutdown_stops_a_worker_that_does_not_exit_after_done(
        self, make_manager, tmp_path, mode
    ):
        """shutdown() stops a worker that runs on after its done message, and returns at once."""
        manager = make_manager(mode)
        job_id = _submit(
            manager,
            output_dir=tmp_path / "user-outputs" if mode == "local" else None,
            action="done_then_wait",
            release=str(tmp_path / "release"),
        )
        _wait_until(
            lambda: manager.snapshot(job_id).state is JobState.SUCCEEDED, "the job succeeded"
        )
        process = _job_record(manager, job_id).process

        shutdown_started_at = time.monotonic()
        manager.shutdown()

        # The fake worker would wait 60 seconds for the release file.
        assert time.monotonic() - shutdown_started_at < 10
        _assert_reaped(process)
        assert manager.snapshot(job_id).state is JobState.SUCCEEDED

    def test_housekeeping_thread_applies_the_limits(self, make_manager, tmp_path, clock):
        """The housekeeping thread runs passes on its own until shutdown."""
        manager = make_manager(max_run_time_s=60, housekeeping_interval_s=0.05)
        job_id = _submit(
            manager,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        _read_report(tmp_path / "report.json")

        clock.advance(61)

        _wait_until(
            lambda: manager.snapshot(job_id).state is JobState.STOPPED, "the thread stopped the job"
        )
        manager.shutdown()
        assert not manager._housekeeping_thread.is_alive()


class TestWorkspaceDeletion:
    def test_result_is_deleted_at_the_latest_after_the_maximum_retention(self, make_manager, clock):
        """Without a download click, a result is deleted 30 minutes after the job end (AC7)."""
        manager = make_manager()
        job_id = _submit(manager)
        snapshot = _ended_snapshot(manager, job_id)
        archive_path = Path(snapshot.result["archive"])
        assert archive_path.is_file()

        clock.advance(30 * 60 - 1)
        manager.run_housekeeping()
        assert archive_path.is_file()
        assert manager.snapshot(job_id).archive_available is True

        clock.advance(1)
        manager.run_housekeeping()
        snapshot = manager.snapshot(job_id)
        assert not archive_path.parent.exists()
        assert snapshot.state is JobState.SUCCEEDED
        assert snapshot.archive_available is False

    def test_first_download_click_starts_a_shorter_retention(self, make_manager, clock):
        """A result is deleted 10 minutes after the first click. Later clicks do not count (AC7)."""
        manager = make_manager()
        job_id = _submit(manager)
        archive_path = Path(_ended_snapshot(manager, job_id).result["archive"])

        clock.advance(100)
        manager.mark_downloaded(job_id)
        clock.advance(300)
        manager.mark_downloaded(job_id)
        clock.advance(299)
        manager.run_housekeeping()
        assert archive_path.is_file()

        clock.advance(1)
        manager.run_housekeeping()
        assert not archive_path.parent.exists()
        assert manager.snapshot(job_id).archive_available is False

    def test_new_job_of_the_same_session_deletes_the_earlier_result(self, make_manager, tmp_path):
        """Starting a new job deletes the session's earlier result and forgets that job (AC7)."""
        manager = make_manager()
        first_id = _submit(manager, session="session-a")
        archive_path = Path(_ended_snapshot(manager, first_id).result["archive"])
        other_session_id = _submit(manager, session="session-b", key_name="bob")
        other_archive_path = Path(_ended_snapshot(manager, other_session_id).result["archive"])

        second_id = _submit(manager, session="session-a", action="error")

        assert not archive_path.parent.exists()
        assert manager.snapshot(first_id) is None
        assert other_archive_path.is_file()
        assert _ended_snapshot(manager, second_id).state is JobState.FAILED

    def test_discard_deletes_a_succeeded_result(self, make_manager):
        """ "Clear results" deletes the workspace and forgets the job (AC7)."""
        manager = make_manager()
        job_id = _submit(manager)
        archive_path = Path(_ended_snapshot(manager, job_id).result["archive"])

        manager.discard(job_id)

        assert not archive_path.parent.exists()
        assert manager.snapshot(job_id) is None
        assert manager._jobs == {}

    def test_discard_of_a_running_job_stops_it_first(self, make_manager, tmp_path):
        """Discarding a running job stops the worker, then deletes the workspace."""
        manager = make_manager()
        job_id = _submit(
            manager,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        workspace = Path(_read_report(tmp_path / "report.json")["cwd"])
        record = _job_record(manager, job_id)

        manager.discard(job_id)
        _wait_for_worker_end(record)

        _assert_reaped(record.process)
        assert record.state is JobState.CANCELLED
        assert not workspace.exists()
        assert manager.snapshot(job_id) is None
        assert manager._jobs == {}

    def test_failed_hosted_job_is_deleted_at_once(self, make_manager, tmp_path):
        """A failed hosted job gives no partial results: the workspace goes at the worker exit."""
        manager = make_manager()
        job_id = _submit(
            manager,
            action="error",
            report=str(tmp_path / "report.json"),
            write_file={"path": "outputs/partial.csv", "size_bytes": 5},
        )
        workspace = Path(_read_report(tmp_path / "report.json")["cwd"])
        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.FAILED
        assert snapshot.error == {
            "type": "error",
            "message": "Fake failure.",
            "warning_count": 2,
            "run_log_tail": ["WARNING fake run-log line"],
        }
        assert snapshot.archive_available is False
        assert not workspace.exists()

    def test_hosted_start_deletes_only_marked_workspaces(self, make_manager, workspace_parent):
        """Hosted start-up deletes leftover workspaces with the marker, and nothing else (AC7)."""
        workspace_root = workspace_parent / "envoi-webapp-jobs"
        workspace_root.mkdir(mode=0o700, parents=True)
        leftover = workspace_root / ("A" * 22)
        (leftover / "outputs").mkdir(parents=True)
        (leftover / ".envoi-webapp-job").touch()
        (leftover / "outputs" / "result.csv").write_text("x", encoding="utf-8")
        without_marker = workspace_root / ("B" * 22)
        without_marker.mkdir()
        other_name = workspace_root / "not-a-job-workspace"
        other_name.mkdir()
        (other_name / ".envoi-webapp-job").touch()
        file_with_token_name = workspace_root / ("C" * 22)
        file_with_token_name.write_text("x", encoding="utf-8")
        sibling = workspace_root.parent / "other-files"
        sibling.mkdir()

        make_manager("local")
        assert leftover.exists()

        make_manager("hosted")
        assert not leftover.exists()
        assert without_marker.is_dir()
        assert other_name.is_dir()
        assert file_with_token_name.is_file()
        assert sibling.is_dir()

    def test_second_hosted_manager_keeps_the_live_workspaces(self, make_manager, tmp_path):
        """The clean-up runs once per process and root, so a second manager deletes nothing."""
        first_manager = make_manager()
        job_id = _submit(
            first_manager,
            action="wait",
            release=str(tmp_path / "release"),
            report=str(tmp_path / "report.json"),
        )
        workspace = Path(_read_report(tmp_path / "report.json")["cwd"])

        second_manager = make_manager()

        assert second_manager._workspace_root == first_manager._workspace_root
        assert (workspace / ".envoi-webapp-job").is_file()
        (tmp_path / "release").touch()
        snapshot = _ended_snapshot(first_manager, job_id)
        assert snapshot.state is JobState.SUCCEEDED
        assert snapshot.archive_available is True


class TestWorkerFailures:
    def test_crashed_worker_gives_failed_and_logs_the_end_of_its_stderr(
        self, make_manager, tmp_path, caplog
    ):
        """A worker that exits without a final message gives FAILED with a general message."""
        caplog.set_level(logging.WARNING, logger="envoi_webapp.jobs")
        manager = make_manager()
        job_id = _submit(manager, action="crash", report=str(tmp_path / "report.json"))
        workspace = Path(_read_report(tmp_path / "report.json")["cwd"])
        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.FAILED
        assert snapshot.error == {
            "type": "error",
            "message": "The extraction process ended unexpectedly.",
            "warning_count": 0,
            "run_log_tail": [],
        }
        assert snapshot.stop_reason is None
        assert "exit code 3" in caplog.text
        assert "fake worker crashed on purpose" in caplog.text
        assert not workspace.exists()

    def test_line_that_is_not_json_is_skipped_and_not_logged(self, make_manager, caplog):
        """A line that is not a message is skipped. The log says so without the line content."""
        caplog.set_level(logging.WARNING, logger="envoi_webapp.jobs")
        manager = make_manager()
        job_id = _submit(manager, action="not_json")

        assert _ended_snapshot(manager, job_id).state is JobState.SUCCEEDED
        assert "skipped 1 line(s)" in caplog.text
        assert "this line is not JSON" not in caplog.text

    @pytest.mark.parametrize(
        ("mode", "reported_archive"),
        [("hosted", "another path"), ("hosted", None), ("local", "another path")],
    )
    def test_done_with_another_archive_gives_failed(
        self, make_manager, tmp_path, caplog, mode, reported_archive
    ):
        """A done message whose archive is not the request's archive path gives FAILED."""
        caplog.set_level(logging.WARNING, logger="envoi_webapp.jobs")
        manager = make_manager(mode)
        other_archive_path = str(tmp_path / "elsewhere.zip")
        job_id = _submit(
            manager,
            output_dir=tmp_path / "user-outputs" if mode == "local" else None,
            archive_override=other_archive_path if reported_archive else None,
            report=str(tmp_path / "report.json"),
        )
        workspace = Path(_read_report(tmp_path / "report.json")["cwd"])
        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.FAILED
        assert snapshot.result is None
        assert snapshot.error == {
            "type": "error",
            "message": "The extraction process sent a result that the web app cannot use.",
            "warning_count": 0,
            "run_log_tail": [],
        }
        assert snapshot.archive_available is False
        assert "archive path other than the one in its request" in caplog.text
        assert other_archive_path not in caplog.text
        assert not workspace.exists()

    def test_worker_that_exits_before_reading_the_request_gives_failed(self, make_manager):
        """A worker that ends before it reads the request gives FAILED, and no process remains."""
        manager = make_manager(worker_command=[sys.executable, "-c", "pass"])
        # The request is larger than a pipe buffer, so the write fails when the worker exits.
        job_id = _submit(manager, points=pd.DataFrame({"value": range(1_000_000)}))
        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.FAILED
        assert snapshot.error["message"] == "The extraction process could not start."

    def test_request_that_cannot_be_pickled_raises_and_frees_the_slot(self, make_manager):
        """A programming error in the request raises, and the session can submit again."""
        manager = make_manager()

        with pytest.raises((pickle.PicklingError, AttributeError)):
            _submit(manager, not_picklable=lambda: None)

        assert _folders(manager._workspace_root) == []
        assert _ended_snapshot(manager, _submit(manager)).state is JobState.SUCCEEDED

    def test_worker_command_that_cannot_start_gives_failed(self, make_manager, tmp_path, caplog):
        """A worker command that does not exist gives FAILED and frees the slot."""
        caplog.set_level(logging.WARNING, logger="envoi_webapp.jobs")
        manager = make_manager(worker_command=[str(tmp_path / "no-such-program")])

        job_id = _submit(manager)
        snapshot = manager.snapshot(job_id)

        assert snapshot.state is JobState.FAILED
        assert snapshot.error["message"] == "The extraction process could not start."
        assert "could not start the worker process" in caplog.text
        assert _folders(manager._workspace_root) == []
        assert manager.snapshot(_submit(manager)).state is JobState.FAILED


class TestSnapshot:
    def test_progress_keeps_the_latest_message_per_segment_in_first_seen_order(self, make_manager):
        """Thousands of progress lines leave one message per segment, in first-seen order."""
        manager = make_manager()

        def progress(dataset: str, completed: int) -> dict:
            return {
                "type": "progress",
                "batch_id": "extract_01_fake",
                "dataset": dataset,
                "window_size_m": 0,
                "mode": "tabular",
                "completed": completed,
                "total": 2_000,
                "unit": "points",
            }

        messages = [progress("first_seen", 0)]
        for completed in range(1, 2_001):
            messages.append(progress("second_seen", completed))
            messages.append(progress("first_seen", completed))
        job_id = _submit(manager, action="progress", messages=messages)

        snapshot = _ended_snapshot(manager, job_id)

        assert snapshot.state is JobState.SUCCEEDED
        assert snapshot.progress == (progress("first_seen", 2_000), progress("second_seen", 2_000))

    def test_unknown_job(self, make_manager):
        """An unknown job ID gives None, and the other calls do nothing."""
        manager = make_manager()

        assert manager.snapshot("unknown") is None
        assert manager.cancel("unknown") is False
        manager.mark_downloaded("unknown")
        manager.discard("unknown")


class _CountingLocalSettings:
    """A settings factory for local mode that counts how often it is called."""

    def __init__(self, workspace_parent: Path) -> None:
        self.workspace_parent = workspace_parent
        self.calls = 0

    def __call__(self) -> WebappSettings:
        self.calls += 1
        return WebappSettings(mode="local", workspace_parent=self.workspace_parent, limits=None)


class TestProcessJobManager:
    @pytest.fixture(autouse=True)
    def _no_process_job_manager(self):
        # Start and end without a job manager of the process, so that no other
        # test gets the manager of this test.
        jobs._reset_job_manager_for_tests()
        yield
        jobs._reset_job_manager_for_tests()

    @pytest.fixture
    def local_settings(self, workspace_parent) -> _CountingLocalSettings:
        return _CountingLocalSettings(workspace_parent)

    def test_every_call_returns_the_same_manager(self, local_settings):
        """The first call creates the manager. Later calls return it and read no settings."""
        first_manager = jobs.get_job_manager(local_settings)

        assert isinstance(first_manager, JobManager)
        assert jobs.get_job_manager(local_settings) is first_manager
        assert local_settings.calls == 1

    def test_concurrent_first_calls_create_one_manager(self, local_settings):
        """Sessions that ask for the manager at the same time all get the same manager."""

        def slow_local_settings() -> WebappSettings:
            # Long enough for every thread to call get_job_manager() meanwhile.
            time.sleep(0.2)
            return local_settings()

        managers: list[JobManager] = []
        threads = [
            threading.Thread(
                target=lambda: managers.append(jobs.get_job_manager(slow_local_settings))
            )
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(_WAIT_TIMEOUT_S)

        assert len(managers) == 4
        assert all(manager is managers[0] for manager in managers)
        assert local_settings.calls == 1

    @pytest.mark.parametrize(
        ("problem", "expected_message"),
        [
            ("invalid_mode", "ENVOI_WEBAPP_MODE must be 'local' or 'hosted'"),
            pytest.param(
                "unsafe_root",
                "gives permissions to the group or to others",
                marks=pytest.mark.skipif(
                    not _IS_POSIX, reason="The owner and mode check runs on POSIX only."
                ),
            ),
        ],
    )
    def test_failed_creation_stores_nothing(
        self, workspace_parent, local_settings, problem, expected_message
    ):
        """A creation error reaches the caller, and the next call creates a manager."""
        if problem == "invalid_mode":

            def failing_settings() -> WebappSettings:
                return load_settings({"ENVOI_WEBAPP_MODE": "public"})

        else:
            root = workspace_parent / "envoi-webapp-jobs"
            root.mkdir(mode=0o700, parents=True)
            os.chmod(root, 0o777)

            def failing_settings() -> WebappSettings:
                return WebappSettings(
                    mode="hosted", workspace_parent=workspace_parent, limits=HostedLimits()
                )

        with pytest.raises(ValueError, match=expected_message):
            jobs.get_job_manager(failing_settings)

        manager = jobs.get_job_manager(local_settings)
        assert isinstance(manager, JobManager)
        assert local_settings.calls == 1
        assert jobs.get_job_manager(failing_settings) is manager
