from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from envoi.progress import ProgressEvent
from envoi_webapp.job_protocol import (
    JobRequest,
    JobSnapshot,
    JobState,
    encode_message,
    parse_message,
)

_KEY_TEXT = json.dumps(
    {
        "type": "service_account",
        "private_key_id": "protocol-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nPROTOCOLKEYLINE\n-----END PRIVATE KEY-----\n",
        "client_email": "protocol@example.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
)


def _progress_message() -> dict:
    return {
        "type": "progress",
        "batch_id": "extract_01_dem",
        "dataset": "dem_copernicus_glo30",
        "window_size_m": 100,
        "mode": "tabular",
        "completed": 3,
        "total": 10,
        "unit": "points",
    }


def _done_message() -> dict:
    return {
        "type": "done",
        "outputs": {"extract_01_dem": "extract_01_dem.csv", "extract_02_dem": "extract_02_dem"},
        "archive": "/workspace/job/results.zip",
        "warning_count": 2,
        "run_log": "envoi-run-log-20261008T120000Z.txt",
    }


def _error_message() -> dict:
    return {
        "type": "error",
        "message": "Earth Engine refused the key.",
        "warning_count": 1,
        "run_log_tail": ["WARNING first line", "ERROR second line"],
    }


def _job_request() -> JobRequest:
    return JobRequest(
        points=pd.DataFrame(
            {
                "occurrenceID": ["a", "b"],
                "decimalLatitude": [59.1, 59.2],
                "decimalLongitude": [18.1, 18.2],
            }
        ),
        run_configs=[{"batch_id": "extract_01_dem", "datasets": ["dem"], "settings": {}}],
        input_crs="EPSG:4326",
        output_dir=Path("/workspace/job/outputs"),
        archive_path=Path("/workspace/job/results.zip"),
        run_log_name="envoi-run-log-20261008T120000Z.txt",
        credentials_json=_KEY_TEXT,
    )


class TestMessageRoundTrip:
    @pytest.mark.parametrize("build_message", [_progress_message, _done_message, _error_message])
    def test_message_survives_encode_and_parse(self, build_message):
        """Each message type comes back unchanged from encode_message and parse_message."""
        message = build_message()

        assert parse_message(encode_message(message)) == message

    def test_done_message_without_archive_survives(self):
        """A done message of local mode, with no archive, comes back unchanged."""
        message = {**_done_message(), "archive": None}

        assert parse_message(encode_message(message)) == message

    def test_encoded_message_is_one_ascii_line(self):
        """Non-ASCII text and line breaks inside strings still give one ASCII line."""
        message = {
            **_error_message(),
            "message": "Fel för punkt Å\nsecond line",
            "run_log_tail": ["Ünïcode line", "line with\r\nbreak"],
        }

        encoded = encode_message(message)

        assert encoded.isascii()
        assert encoded.endswith("\n")
        assert encoded.count("\n") == 1
        assert parse_message(encoded) == message

    def test_progress_message_fields_match_progress_event(self):
        """A progress message built from the fields of a ProgressEvent is valid."""
        event = ProgressEvent(
            batch_id="extract_01_dem",
            dataset="dem",
            window_size_m=250,
            mode="raster",
            completed=1,
            total=2,
            unit="tiles",
        )
        message = {"type": "progress", **dataclasses.asdict(event)}

        assert parse_message(encode_message(message)) == message

    def test_parse_message_accepts_bytes(self):
        """A line read from a binary channel parses like a text line."""
        encoded = encode_message(_progress_message())

        assert parse_message(encoded.encode("ascii")) == _progress_message()


class TestInvalidMessages:
    @pytest.mark.parametrize(
        "line",
        [
            "",
            "\n",
            "not json",
            "Warning: printed by a library",
            "[1, 2, 3]",
            '"progress"',
            "null",
            '{"type": "unknown"}',
            '{"type": ["progress"]}',
            '{"batch_id": "a"}',
            b"\xff\xfe\x00not utf-8",
        ],
    )
    def test_parse_message_returns_none_for_bad_line(self, line):
        """A line that is not JSON, or not a message object, gives None."""
        assert parse_message(line) is None

    @pytest.mark.parametrize(
        ("build_message", "change"),
        [
            (_progress_message, {"completed": "3"}),
            (_progress_message, {"total": True}),
            (_progress_message, {"window_size_m": 100.5}),
            (_progress_message, {"mode": "vector"}),
            (_done_message, {"outputs": {"extract_01_dem": 5}}),
            (_done_message, {"outputs": ["extract_01_dem.csv"]}),
            (_done_message, {"archive": 0}),
            (_done_message, {"warning_count": None}),
            (_error_message, {"message": None}),
            (_error_message, {"run_log_tail": "one line"}),
            (_error_message, {"run_log_tail": ["line", 2]}),
            (_error_message, {"extra_field": "value"}),
        ],
    )
    def test_parse_message_returns_none_for_wrong_field(self, build_message, change):
        """A message with a wrong field type, wrong value, or extra field gives None."""
        line = json.dumps({**build_message(), **change})

        assert parse_message(line) is None

    @pytest.mark.parametrize(
        ("build_message", "missing_field"),
        [
            (_progress_message, "unit"),
            (_done_message, "run_log"),
            (_error_message, "run_log_tail"),
        ],
    )
    def test_parse_message_returns_none_for_missing_field(self, build_message, missing_field):
        """A message without one of its fields gives None."""
        message = build_message()
        del message[missing_field]

        assert parse_message(json.dumps(message)) is None

    def test_encode_message_rejects_invalid_message(self):
        """encode_message raises instead of writing a line that parse_message skips."""
        message = _progress_message()
        del message["total"]

        with pytest.raises(ValueError, match="Not a valid job message"):
            encode_message(message)


class TestJobRequest:
    def test_repr_contains_no_key_text(self):
        """repr() of a request shows no part of the key."""
        text = repr(_job_request())

        assert "credentials_json" not in text
        for secret in (
            _KEY_TEXT,
            "PROTOCOLKEYLINE",
            "protocol-key-id",
            "protocol@example.iam.gserviceaccount.com",
        ):
            assert secret not in text
        assert "extract_01_dem" in text

    def test_request_survives_pickle(self):
        """The worker gets the same points, settings, and key from the pickled request."""
        request = _job_request()

        restored = pickle.loads(pickle.dumps(request))

        pd.testing.assert_frame_equal(restored.points, request.points)
        assert restored.run_configs == request.run_configs
        assert restored.output_dir == request.output_dir
        assert restored.archive_path == request.archive_path
        assert restored.run_log_name == request.run_log_name
        assert restored.credentials_json == _KEY_TEXT


class TestJobState:
    def test_state_values(self):
        """The job states use the names of plan decision D12."""
        assert {state.value for state in JobState} == {
            "running",
            "succeeded",
            "failed",
            "cancelled",
            "stopped",
        }

    @pytest.mark.parametrize("state", list(JobState))
    def test_only_running_is_not_final(self, state):
        """Every state except RUNNING is a final state."""
        assert state.is_final is (state is not JobState.RUNNING)


class TestJobSnapshot:
    def test_new_snapshot_has_no_result(self):
        """A snapshot of a job that just started has no progress, result, or error."""
        snapshot = JobSnapshot(job_id="job-1", state=JobState.RUNNING, started_at=10.0)

        assert snapshot.ended_at is None
        assert snapshot.progress == ()
        assert snapshot.result is None
        assert snapshot.error is None
        assert snapshot.stop_reason is None
