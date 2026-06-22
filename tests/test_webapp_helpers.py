from __future__ import annotations

import json
import stat
from pathlib import Path

import pandas as pd
import pytest

from envoi_webapp.helpers import (
    RASTER_OUTPUT,
    TABULAR_OUTPUT,
    DatasetSelection,
    build_run_config,
    normalize_crs,
    parse_window_sizes,
    redact_credential_secrets,
    run_extraction,
    temporary_service_account_file,
    validate_output_dir,
    validate_points_dataframe,
    validate_service_account_json,
)


def _points_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gbifID": ["a", "b"],
            "decimalLatitude": [59.1, 59.2],
            "decimalLongitude": [18.1, 18.2],
        }
    )


def _credential_bytes() -> bytes:
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "demo",
            "private_key_id": "private-key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\nPRIVATEKEY\n-----END PRIVATE KEY-----\n",
            "client_email": "svc@example.iam.gserviceaccount.com",
            "client_id": "123",
        }
    ).encode("utf-8")


def test_validate_points_dataframe_accepts_optional_date_absent():
    result = validate_points_dataframe(_points_df())

    assert result.row_count == 2
    assert result.has_date is False
    assert "gbifID" in result.columns


def test_validate_points_dataframe_rejects_missing_gbif_columns():
    with pytest.raises(ValueError, match="missing required column"):
        validate_points_dataframe(pd.DataFrame({"x": [1], "y": [2]}))


def test_normalize_crs_accepts_epsg_variants():
    assert normalize_crs("") == "EPSG:4326"
    assert normalize_crs("4326") == "EPSG:4326"
    assert normalize_crs("epsg:3006") == "EPSG:3006"


def test_normalize_crs_rejects_invalid_crs():
    with pytest.raises(ValueError, match="Invalid CRS"):
        normalize_crs("not-a-crs")


def test_validate_output_dir_creates_and_checks_writable_path(tmp_path):
    output_dir = validate_output_dir(tmp_path / "outputs")

    assert output_dir.exists()
    assert output_dir.is_dir()


def test_parse_window_sizes_accepts_comma_separated_positive_integers():
    assert parse_window_sizes("100, 250;500") == (100, 250, 500)


def test_build_run_config_tabular():
    rows = [DatasetSelection("dem_copernicus_glo30", (100, 250), ("mean", "std"))]

    config = build_run_config(rows, TABULAR_OUTPUT)

    assert config == [
        {
            "batch_id": "extract_01_dem_copernicus_glo30",
            "datasets": ["dem_copernicus_glo30"],
            "settings": {
                "output_type": "tabular",
                "window_size_m": [100, 250],
                "statistics": ["mean", "std"],
                "output_file_format": "csv",
            },
        }
    ]


def test_build_run_config_raster_uses_10m_resampling_and_no_statistics():
    rows = [DatasetSelection("dem_copernicus_glo30", (200,), ("mean",))]

    config = build_run_config(rows, RASTER_OUTPUT)

    assert config[0]["settings"] == {
        "output_type": "raster",
        "window_size_m": 200,
        "resample_m": 10,
    }


def test_validate_service_account_json_rejects_bad_shape():
    with pytest.raises(ValueError, match="service account"):
        validate_service_account_json(b'{"type": "authorized_user"}')


def test_temporary_service_account_file_exists_during_context_and_is_private():
    payload = _credential_bytes()

    with temporary_service_account_file(payload) as path:
        assert path.exists()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600
        assert json.loads(path.read_text())["private_key_id"] == "private-key-id"

    assert not path.exists()


def test_temporary_service_account_file_is_deleted_after_failure():
    payload = _credential_bytes()
    path_seen: Path | None = None

    with pytest.raises(RuntimeError, match="boom"):
        with temporary_service_account_file(payload) as path:
            path_seen = path
            raise RuntimeError("boom")

    assert path_seen is not None
    assert not path_seen.exists()


def test_redact_credential_secrets_removes_key_material():
    payload = _credential_bytes()
    message = (
        "failed with PRIVATEKEY and svc@example.iam.gserviceaccount.com "
        f"inside {payload.decode('utf-8')}"
    )

    safe_message = redact_credential_secrets(message, payload)

    assert "PRIVATEKEY" not in safe_message
    assert "svc@example.iam.gserviceaccount.com" not in safe_message
    assert payload.decode("utf-8") not in safe_message


def test_run_extraction_initializes_gee_with_temp_key_and_passes_expected_args(tmp_path):
    payload = _credential_bytes()
    init_paths: list[Path] = []
    captured: dict = {}

    def fake_init(credentials_path):
        path = Path(credentials_path)
        assert path.exists()
        init_paths.append(path)

    def fake_extract(df, config, **kwargs):
        assert init_paths[-1].exists()
        captured["df"] = df
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"extract_01_dem": kwargs["output_dir"] / "extract_01_dem.csv"}

    outputs = run_extraction(
        _points_df(),
        [DatasetSelection("dem", (100,), ("mean",))],
        TABULAR_OUTPUT,
        tmp_path / "outputs",
        "EPSG:4326",
        payload,
        init_gee_func=fake_init,
        extract_func=fake_extract,
    )

    assert outputs["extract_01_dem"].name == "extract_01_dem.csv"
    assert captured["config"][0]["batch_id"] == "extract_01_dem"
    assert captured["kwargs"]["input_crs"] == "EPSG:4326"
    assert captured["kwargs"]["id_column"] == "gbifID"
    assert captured["kwargs"]["quiet"] is True
    assert init_paths and not init_paths[0].exists()
