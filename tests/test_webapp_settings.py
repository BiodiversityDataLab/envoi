from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from envoi_webapp.helpers import RASTER_OUTPUT, TABULAR_OUTPUT, DatasetSelection
from envoi_webapp.settings import (
    HostedLimits,
    check_hosted_limits,
    check_upload,
    format_bytes,
    format_minutes,
    load_settings,
)

_MB = 1024 * 1024
_GB = 1024 * 1024 * 1024
_LIMITS = HostedLimits()
# The Streamlit settings of the hosted container image.
_HOSTED_STREAMLIT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "deploy" / "serve" / ".streamlit" / "config.toml"
)


def _points(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "occurrenceID": [f"P{index}" for index in range(row_count)],
            "decimalLatitude": [59.0] * row_count,
            "decimalLongitude": [18.0] * row_count,
        }
    )


def _tabular(*window_sizes: int, dataset: str = "dem_copernicus_glo30") -> DatasetSelection:
    return DatasetSelection(dataset, TABULAR_OUTPUT, tuple(window_sizes), ("mean",))


def _raster(*window_sizes: int, dataset: str = "dem_copernicus_glo30") -> DatasetSelection:
    return DatasetSelection(dataset, RASTER_OUTPUT, tuple(window_sizes))


def _csv_bytes(line_count: int, *, final_line_break: bool = True, line_end: str = "\n") -> bytes:
    lines = ["occurrenceID,decimalLatitude,decimalLongitude"]
    lines.extend(f"P{index},59.0,18.0" for index in range(line_count - 1))
    text = line_end.join(lines)
    return (text + line_end if final_line_break else text).encode("utf-8")


class TestLoadSettings:
    @pytest.fixture(autouse=True)
    def _platform_is_not_windows(self, monkeypatch):
        # load_settings() refuses hosted mode on Windows. The other rules are the
        # same on each platform, so these tests run as on Linux everywhere.
        monkeypatch.setattr(sys, "platform", "linux")

    def test_defaults_are_local_mode_in_the_temporary_folder(self):
        """Without variables, the app runs in local mode without limits."""
        settings = load_settings({})

        assert settings.mode == "local"
        assert settings.is_hosted is False
        assert settings.limits is None
        assert settings.workspace_parent == Path(tempfile.gettempdir())

    def test_hosted_mode_has_the_d8_limits(self):
        """Hosted mode gets the starting values of plan decision D8."""
        settings = load_settings({"ENVOI_WEBAPP_MODE": "hosted"})

        assert settings.mode == "hosted"
        assert settings.is_hosted is True
        assert settings.limits == HostedLimits(
            max_upload_bytes=50 * _MB,
            max_upload_lines=10_001,
            max_input_rows=10_000,
            max_dataset_rows=10,
            tabular_request_budget=20_000,
            raster_tile_budget=1_000,
            max_tabular_window_m=10_000,
            max_raster_window_m=2_000,
            max_concurrent_jobs=2,
            max_run_time_s=3_600,
            max_workspace_bytes=1 * _GB,
            disk_budget_bytes=4 * _GB,
            abandon_timeout_s=600,
            retention_after_download_s=600,
            max_retention_s=1_800,
        )

    @pytest.mark.parametrize(
        ("value", "expected_mode"),
        [("local", "local"), ("hosted", "hosted"), (" Hosted ", "hosted"), ("", "local")],
    )
    def test_mode_values(self, value, expected_mode):
        """The mode ignores case and spaces. An empty value means local mode."""
        assert load_settings({"ENVOI_WEBAPP_MODE": value}).mode == expected_mode

    @pytest.mark.parametrize("value", ["public", "server", "1"])
    def test_invalid_mode_raises(self, value):
        """Another mode value raises ValueError that names the variable and the valid values."""
        with pytest.raises(ValueError, match="ENVOI_WEBAPP_MODE must be 'local' or 'hosted'"):
            load_settings({"ENVOI_WEBAPP_MODE": value})

    @pytest.mark.parametrize("mode", ["local", "hosted"])
    def test_workspace_variable_sets_the_parent_folder(self, tmp_path, mode):
        """ENVOI_WEBAPP_WORKSPACE sets the parent folder of the workspace root in both modes."""
        settings = load_settings(
            {"ENVOI_WEBAPP_MODE": mode, "ENVOI_WEBAPP_WORKSPACE": str(tmp_path)}
        )

        assert settings.workspace_parent == tmp_path

    def test_workspace_parent_expands_the_home_folder(self):
        """A parent that starts with ~ is expanded to the home folder."""
        settings = load_settings({"ENVOI_WEBAPP_WORKSPACE": "~/envoi-jobs"})

        assert settings.workspace_parent == Path("~/envoi-jobs").expanduser()

    def test_relative_workspace_parent_raises(self):
        """A relative workspace parent raises ValueError."""
        with pytest.raises(ValueError, match="ENVOI_WEBAPP_WORKSPACE must be an absolute"):
            load_settings({"ENVOI_WEBAPP_WORKSPACE": "relative/folder"})

    @pytest.mark.parametrize(("value", "expected_jobs"), [("1", 1), (" 3 ", 3), ("", 2)])
    def test_max_jobs(self, value, expected_jobs):
        """ENVOI_WEBAPP_MAX_JOBS sets the concurrent-job limit of hosted mode."""
        settings = load_settings({"ENVOI_WEBAPP_MODE": "hosted", "ENVOI_WEBAPP_MAX_JOBS": value})

        assert settings.limits.max_concurrent_jobs == expected_jobs

    @pytest.mark.parametrize("value", ["0", "-1", "two", "2.5"])
    def test_invalid_max_jobs_raises(self, value):
        """A max-jobs value that is not a positive whole number raises ValueError."""
        with pytest.raises(ValueError, match="ENVOI_WEBAPP_MAX_JOBS must be a positive whole"):
            load_settings({"ENVOI_WEBAPP_MODE": "hosted", "ENVOI_WEBAPP_MAX_JOBS": value})

    def test_local_mode_ignores_max_jobs(self):
        """Local mode has no server-wide limit, so it does not read ENVOI_WEBAPP_MAX_JOBS."""
        settings = load_settings({"ENVOI_WEBAPP_MAX_JOBS": "not a number"})

        assert settings.limits is None

    def test_reads_the_process_environment_by_default(self, monkeypatch, tmp_path):
        """Without an argument, the function reads os.environ."""
        monkeypatch.setenv("ENVOI_WEBAPP_MODE", "hosted")
        monkeypatch.setenv("ENVOI_WEBAPP_WORKSPACE", str(tmp_path))
        monkeypatch.setenv("ENVOI_WEBAPP_MAX_JOBS", "4")

        settings = load_settings()

        assert settings.mode == "hosted"
        assert settings.workspace_parent == tmp_path
        assert settings.limits.max_concurrent_jobs == 4

    def test_hosted_mode_on_windows_raises(self, monkeypatch):
        """Hosted mode is refused on Windows. Local mode works there."""
        monkeypatch.setattr(sys, "platform", "win32")

        with pytest.raises(ValueError, match="runs only on Linux or macOS"):
            load_settings({"ENVOI_WEBAPP_MODE": "hosted"})
        assert load_settings({}).mode == "local"


class TestFormat:
    @pytest.mark.parametrize(
        ("size_bytes", "expected"),
        [
            (50 * _MB, "50 MB"),
            (int(51.5 * _MB), "51.5 MB"),
            (1 * _GB, "1 GB"),
            (int(1.5 * _GB), "1.5 GB"),
            (4 * _GB, "4 GB"),
        ],
    )
    def test_format_bytes(self, size_bytes, expected):
        """Sizes show in MB below 1 GB and in GB from 1 GB, with at most one decimal."""
        assert format_bytes(size_bytes) == expected

    @pytest.mark.parametrize(
        ("duration_s", "expected"), [(60, "1 minute"), (600, "10 minutes"), (3_600, "60 minutes")]
    )
    def test_format_minutes(self, duration_s, expected):
        """Durations show in whole minutes."""
        assert format_minutes(duration_s) == expected


class TestCheckUpload:
    def test_file_at_the_limits_passes(self):
        """A file of exactly the size limit and the line limit passes."""
        raw_bytes = _csv_bytes(10_001)

        assert check_upload(50 * _MB, raw_bytes, _LIMITS) == []

    def test_file_above_the_size_limit_is_rejected(self):
        """One byte above the size limit gives one message that states the limit."""
        messages = check_upload(50 * _MB + 1, _csv_bytes(3), _LIMITS)

        assert len(messages) == 1
        assert "The limit is 50 MB" in messages[0]

    def test_file_above_the_line_limit_is_rejected(self):
        """One line above the line limit gives one message that states the limit."""
        messages = check_upload(1_000, _csv_bytes(10_002), _LIMITS)

        assert len(messages) == 1
        assert "The file has 10,002 lines" in messages[0]
        assert "The limit is 10,001 lines" in messages[0]
        assert "10,000 data rows" in messages[0]

    @pytest.mark.parametrize(("line_count", "message_count"), [(10_001, 0), (10_002, 1)])
    def test_last_line_without_line_break_counts(self, line_count, message_count):
        """A last line without a line break counts as a line."""
        raw_bytes = _csv_bytes(line_count, final_line_break=False)

        assert len(check_upload(1_000, raw_bytes, _LIMITS)) == message_count

    @pytest.mark.parametrize("line_end", ["\r\n", "\r"], ids=["crlf", "cr"])
    @pytest.mark.parametrize(("line_count", "message_count"), [(10_001, 0), (10_002, 1)])
    def test_crlf_and_cr_line_ends_count_like_lf(self, line_end, line_count, message_count):
        """CRLF (Windows) and CR (old Mac) line ends count like LF, at the limit and above it."""
        messages = check_upload(1_000, _csv_bytes(line_count, line_end=line_end), _LIMITS)

        assert len(messages) == message_count
        if message_count:
            assert f"The file has {line_count:,} lines" in messages[0]

    def test_empty_file_passes(self):
        """An empty file passes here. The CSV validation rejects it later."""
        assert check_upload(0, b"", _LIMITS) == []

    def test_both_limits_give_two_messages(self):
        """Each exceeded limit gives its own message."""
        assert len(check_upload(50 * _MB + 1, _csv_bytes(10_002), _LIMITS)) == 2

    def test_image_upload_limit_matches_the_hosted_limit(self):
        """Streamlit's upload limit in the image equals HostedLimits.max_upload_bytes."""
        tomllib = pytest.importorskip("tomllib")
        with _HOSTED_STREAMLIT_CONFIG_PATH.open("rb") as config_file:
            streamlit_config = tomllib.load(config_file)

        # Streamlit counts whole MB of 1024 * 1024 bytes.
        assert _LIMITS.max_upload_bytes % _MB == 0
        assert streamlit_config["server"]["maxUploadSize"] == _LIMITS.max_upload_bytes // _MB


class TestCheckHostedLimits:
    def test_request_at_every_limit_passes(self):
        """A request at each limit passes. The two budgets do not add up."""
        small_limits = dataclasses.replace(
            _LIMITS, tabular_request_budget=20, raster_tile_budget=10
        )
        # 10 rows, 2 points x 10 tabular window sizes = 20 requests, and the largest windows.
        selections = [_tabular(0, 10_000), _raster(2_000), *[_tabular(100)] * 8]

        assert check_hosted_limits(_points(2), selections, small_limits) == []
        assert check_hosted_limits(_points(10_000), [_tabular(0, 100)], _LIMITS) == []
        assert check_hosted_limits(_points(1_000), [_raster(100)], _LIMITS) == []

    def test_input_rows(self):
        """More input rows than the limit give a message that states the limit."""
        assert check_hosted_limits(_points(10_000), [_tabular(0)], _LIMITS) == []

        messages = check_hosted_limits(_points(10_001), [_tabular(0)], _LIMITS)

        assert len(messages) == 1
        assert "The CSV has 10,001 rows. The limit is 10,000 rows." in messages[0]

    def test_data_product_rows(self):
        """More data-product rows than the limit give a message that states the limit."""
        assert check_hosted_limits(_points(1), [_tabular(100)] * 10, _LIMITS) == []

        messages = check_hosted_limits(_points(1), [_tabular(100)] * 11, _LIMITS)

        assert len(messages) == 1
        assert "11 data-product rows. The limit is 10 rows." in messages[0]

    def test_tabular_request_budget_counts_the_point_value_as_one_window_size(self):
        """Points x the tabular window sizes may not exceed the budget. Point counts as one."""
        assert check_hosted_limits(_points(10_000), [_tabular(0, 100)], _LIMITS) == []
        assert check_hosted_limits(_points(10_000), [_tabular(0), _tabular(100)], _LIMITS) == []

        messages = check_hosted_limits(_points(10_000), [_tabular(0, 100, 200)], _LIMITS)

        assert len(messages) == 1
        assert "10,000 points x 3 window sizes = 30,000 requests" in messages[0]
        assert "The limit is 20,000" in messages[0]

    def test_tabular_request_budget_one_step_above(self):
        """One request above the tabular budget gives a message."""
        small_limits = dataclasses.replace(_LIMITS, tabular_request_budget=9)

        assert check_hosted_limits(_points(3), [_tabular(0, 100, 200)], small_limits) == []
        messages = check_hosted_limits(_points(3), [_tabular(0, 100, 200, 300)], small_limits)

        assert len(messages) == 1
        assert "= 12 requests. The limit is 9." in messages[0]

    def test_raster_tile_budget(self):
        """Points x the raster window sizes may not exceed the tile budget."""
        assert check_hosted_limits(_points(500), [_raster(100), _raster(200)], _LIMITS) == []

        messages = check_hosted_limits(_points(1_001), [_raster(100)], _LIMITS)

        assert len(messages) == 1
        assert "1,001 points x 1 window size = 1,001 tiles. The limit is 1,000." in messages[0]

    @pytest.mark.parametrize(
        ("selection", "limit_text"),
        [
            (_tabular(100, 10_001, dataset="worldclim_bio"), "10,000 m for tabular output"),
            (_raster(2_001, dataset="worldclim_bio"), "2,000 m for raster output"),
            (_raster(5_000, dataset="worldclim_bio"), "2,000 m for raster output"),
        ],
    )
    def test_window_size_limits(self, selection, limit_text):
        """A window size above the limit of the row's output type gives a message for that row."""
        messages = check_hosted_limits(_points(1), [_tabular(10_000), selection], _LIMITS)

        assert len(messages) == 1
        assert messages[0].startswith("Data-product row 2 (worldclim_bio): the window size ")
        assert limit_text in messages[0]

    def test_each_exceeded_limit_gives_a_message(self):
        """Several exceeded limits give one message each."""
        selections = [_tabular(0, 20_000)] * 6 + [_raster(3_000)] * 5

        messages = check_hosted_limits(_points(10_001), selections, _LIMITS)

        # Rows, data-product rows, 11 window rows, tabular budget, raster budget.
        assert len(messages) == 1 + 1 + 11 + 1 + 1
