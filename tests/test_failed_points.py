"""Tests of failed points with the local adapter, through ``extract()``.

A failed point is a point whose read raised an exception that the adapter
caught. Here a stub makes one read raise ``RasterioIOError``. A point outside
the raster's extent is not a failed point. The Earth Engine paths are tested
offline in ``tests/test_gee_adapter_offline.py``.
"""

from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest
import rasterio.io
import rasterio.warp
from rasterio.errors import RasterioIOError

from envoi import update_catalog
from envoi.adapters import local_adapter
from envoi.extract import extract

# The error text of a failed second read (see _fail_reads). The stub message
# has a line break, which the error text collapses to one space.
_SECOND_READ_ERROR = "RasterioIOError: stub: damaged block in read 2"


@pytest.fixture(autouse=True)
def register_local_dem(dem_tif):
    """Register the single-band local DEM as ``dem_local`` (conftest resets the catalog)."""
    update_catalog(
        {"datasets": {"dem_local": {"data_source": "local", "path": str(dem_tif), "bands": 1}}}
    )


@pytest.fixture
def three_points(sample_df) -> pd.DataFrame:
    """The first three points of ``sample_df``, all inside the DEM."""
    return sample_df.head(3)


def _with_outside_point(points: pd.DataFrame) -> pd.DataFrame:
    """Return ``points`` plus a fourth point, ``OUTSIDE``, far outside the DEM."""
    outside_point = pd.DataFrame(
        [
            {
                "occurrenceID": "OUTSIDE",
                "n_otu": 0,
                "decimalLatitude": 0.0,
                "decimalLongitude": 0.0,
                "eventDate": "2025-01-01",
            }
        ]
    )
    return pd.concat([points, outside_point], ignore_index=True)


def _fail_reads(monkeypatch, target, attribute_name: str, failing_reads: set[int] | None) -> list:
    """Make ``target.<attribute_name>`` raise ``RasterioIOError`` on some reads.

    Reads are counted from 1, and ``failing_reads=None`` makes every read fail.
    The local adapter reads its points in input order, so read N is the read of
    the N-th point that reaches ``target``. Other reads call the real function.

    Returns the list of the raised errors, in read order.
    """
    original_read = getattr(target, attribute_name)
    raised_errors: list[RasterioIOError] = []
    read_count = 0

    def read_or_fail(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        if failing_reads is None or read_count in failing_reads:
            error = RasterioIOError(f"stub: damaged block\n in read {read_count}")
            raised_errors.append(error)
            raise error
        return original_read(*args, **kwargs)

    monkeypatch.setattr(target, attribute_name, read_or_fail)
    return raised_errors


def _tabular_config(batch_id: str, statistics: list[str]) -> dict:
    """Return a tabular run config for ``dem_local`` with a 100 m window."""
    return {
        "batch_id": batch_id,
        "datasets": ["dem_local"],
        "settings": {
            "output_type": "tabular",
            "statistics": statistics,
            "window_size_m": 100,
        },
    }


def _raster_config(batch_id: str, **extra_settings) -> dict:
    """Return a raster run config for ``dem_local`` with a 200 m window."""
    return {
        "batch_id": batch_id,
        "datasets": ["dem_local"],
        "settings": {"output_type": "raster", "window_size_m": 200, **extra_settings},
    }


def _run_and_record_warnings(*args, **kwargs):
    """Call ``extract()`` and return ``(outputs, failed_point_warnings)``.

    ``failed_point_warnings`` holds the text of each failed-point warning that
    the call emitted. Other warnings are ignored.
    """
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        outputs = extract(*args, **kwargs)
    failed_point_records = [
        recorded for recorded in recorded_warnings if "points failed" in str(recorded.message)
    ]
    # The warning points at the code that called extract(), not at envoi.
    assert all(recorded.filename == __file__ for recorded in failed_point_records)
    return outputs, [str(recorded.message) for recorded in failed_point_records]


def _read_sidecar(path) -> dict:
    """Read a metadata sidecar with ``json``. This fails if the file is not valid JSON."""
    return json.loads(path.read_text())


class TestFailedPointsTabular:
    """A failed read gives a failed point; the run continues and reports it."""

    def test_failed_window_read_is_reported(self, monkeypatch, three_points, tmp_path):
        """One failed read of three: error column, null statistics, sidecar fields, one warning."""
        _fail_reads(monkeypatch, local_adapter, "rio_mask", {2})

        outputs, failed_point_warnings = _run_and_record_warnings(
            three_points, _tabular_config("failed", ["mean"]), output_dir=tmp_path
        )

        # The statistics of the failed point are null; the other points have values.
        stats_table = pd.read_csv(outputs["failed"])
        assert stats_table["dem_local_mean_100m"].isna().tolist() == [False, True, False]

        # The QC error column has the error text of the failed point and null elsewhere.
        qc_table = pd.read_csv(tmp_path / "failed_qc.csv")
        assert qc_table.loc[1, "dem_local_error_100m"] == _SECOND_READ_ERROR
        assert qc_table["dem_local_error_100m"].isna().tolist() == [True, False, True]
        # The QC flags of the failed point are those of a point outside the extent.
        assert qc_table["dem_local_in_extent_100m"].tolist() == [True, False, True]
        assert qc_table.loc[1, "dem_local_n_pixels_100m"] == 0
        assert qc_table.loc[1, "dem_local_coverage_pct_100m"] == 0.0
        assert not qc_table.loc[1, "dem_local_had_nodata_100m"]

        # The sidecar counts the failed point and lists its ID, also in ids_no_data.
        sidecar = _read_sidecar(tmp_path / "failed_metadata.json")
        quality = sidecar["datasets"]["dem_local"]["quality"]["100m"]
        assert quality["n_errors"] == 1
        assert quality["ids_errors"] == ["T6WPKD"]
        assert quality["ids_no_data"] == ["T6WPKD"]

        # Exactly one failed-point warning, which the sidecar also records.
        assert len(failed_point_warnings) == 1
        warning_text = failed_point_warnings[0]
        assert "Dataset 'dem_local', window 100 m: 1 of 3 points failed" in warning_text
        assert "QC column 'dem_local_error_100m'" in warning_text
        assert f"First error: {_SECOND_READ_ERROR}" in warning_text
        assert sidecar["warnings"]["failed_dem_local_100m"] == warning_text

    def test_batch_without_failures_has_no_error_column(self, three_points, tmp_path):
        """Without a failed point: no error column, n_errors 0, ids_errors [], no warning."""
        outputs, failed_point_warnings = _run_and_record_warnings(
            three_points, _tabular_config("no_failure", ["mean"]), output_dir=tmp_path
        )

        qc_table = pd.read_csv(tmp_path / "no_failure_qc.csv")
        assert "dem_local_error_100m" not in qc_table.columns
        assert not any("error" in column for column in pd.read_csv(outputs["no_failure"]).columns)

        sidecar = _read_sidecar(tmp_path / "no_failure_metadata.json")
        quality = sidecar["datasets"]["dem_local"]["quality"]["100m"]
        assert quality["n_errors"] == 0
        assert quality["ids_errors"] == []
        assert failed_point_warnings == []
        assert "warnings" not in sidecar or not any(
            key.startswith("failed_") for key in sidecar["warnings"]
        )

    def test_failed_pixel_read_is_reported(self, monkeypatch, three_points, tmp_path):
        """A failed pixel read in a point-only run gives a failed point; the run continues."""
        _fail_reads(monkeypatch, rasterio.io.DatasetReader, "sample", {2})

        outputs, failed_point_warnings = _run_and_record_warnings(
            three_points, _tabular_config("failed_pixel", ["point"]), output_dir=tmp_path
        )

        stats_table = pd.read_csv(outputs["failed_pixel"])
        assert stats_table["dem_local_point"].isna().tolist() == [False, True, False]
        qc_table = pd.read_csv(tmp_path / "failed_pixel_qc.csv")
        assert qc_table["dem_local_error_point"].isna().tolist() == [True, False, True]
        assert qc_table.loc[1, "dem_local_error_point"] == _SECOND_READ_ERROR
        quality = _read_sidecar(tmp_path / "failed_pixel_metadata.json")["datasets"]["dem_local"][
            "quality"
        ]["point"]
        assert (quality["n_errors"], quality["ids_errors"]) == (1, ["T6WPKD"])
        assert len(failed_point_warnings) == 1
        assert "QC column 'dem_local_error_point'" in failed_point_warnings[0]
        # A point-only run has no window, so the warning names point values.
        assert "Dataset 'dem_local', point values: 1 of 3" in failed_point_warnings[0]

    def test_failed_pixel_read_with_window_statistics(self, monkeypatch, three_points, tmp_path):
        """When the window read works and the pixel read fails, every statistic of the point is null."""
        _fail_reads(monkeypatch, rasterio.io.DatasetReader, "sample", {2})

        outputs, _ = _run_and_record_warnings(
            three_points, _tabular_config("failed_mixed", ["mean", "point"]), output_dir=tmp_path
        )

        stats_table = pd.read_csv(outputs["failed_mixed"])
        assert stats_table["dem_local_mean_100m"].isna().tolist() == [False, True, False]
        assert stats_table["dem_local_point"].isna().tolist() == [False, True, False]
        qc_table = pd.read_csv(tmp_path / "failed_mixed_qc.csv")
        assert qc_table.loc[1, "dem_local_error_100m"] == _SECOND_READ_ERROR
        assert qc_table["dem_local_in_extent_100m"].tolist() == [True, False, True]
        assert qc_table.loc[1, "dem_local_coverage_pct_100m"] == 0.0

    @pytest.mark.parametrize(
        ("statistic", "target", "attribute_name", "quality_key"),
        [
            ("mean", local_adapter, "rio_mask", "100m"),
            ("point", rasterio.io.DatasetReader, "sample", "point"),
        ],
    )
    def test_point_outside_extent_is_not_failed(
        self, monkeypatch, three_points, tmp_path, statistic, target, attribute_name, quality_key
    ):
        """A point outside the extent has no error, next to a point whose read failed."""
        _fail_reads(monkeypatch, target, attribute_name, {2})

        _run_and_record_warnings(
            _with_outside_point(three_points),
            _tabular_config("outside", [statistic]),
            output_dir=tmp_path,
        )

        qc_table = pd.read_csv(tmp_path / "outside_qc.csv")
        assert qc_table[f"dem_local_error_{quality_key}"].isna().tolist() == [
            True,
            False,
            True,
            True,
        ]
        assert qc_table[f"dem_local_in_extent_{quality_key}"].tolist() == [True, False, True, False]
        quality = _read_sidecar(tmp_path / "outside_metadata.json")["datasets"]["dem_local"][
            "quality"
        ][quality_key]
        assert quality["ids_errors"] == ["T6WPKD"]
        assert quality["ids_no_data"] == ["T6WPKD", "OUTSIDE"]

    def test_point_outside_extent_alone_gives_no_error_column(self, three_points, tmp_path):
        """A batch whose only problem is a point outside the extent has no error column."""
        _run_and_record_warnings(
            _with_outside_point(three_points),
            _tabular_config("outside_only", ["mean"]),
            output_dir=tmp_path,
        )

        qc_table = pd.read_csv(tmp_path / "outside_only_qc.csv")
        assert "dem_local_error_100m" not in qc_table.columns
        quality = _read_sidecar(tmp_path / "outside_only_metadata.json")["datasets"]["dem_local"][
            "quality"
        ]["100m"]
        assert (quality["n_errors"], quality["ids_errors"]) == (0, [])
        assert quality["ids_no_data"] == ["OUTSIDE"]

    def test_every_point_failing_raises_with_the_first_error(
        self, monkeypatch, three_points, tmp_path
    ):
        """When every read fails, extract() raises RuntimeError chained to the first error."""
        raised_errors = _fail_reads(monkeypatch, local_adapter, "rio_mask", None)

        with pytest.raises(RuntimeError, match="every point failed") as error_info:
            extract(three_points, _tabular_config("all_failed", ["mean"]), output_dir=tmp_path)

        message = str(error_info.value)
        assert "Dataset 'dem_local', window 100 m" in message
        assert "First error: RasterioIOError: stub: damaged block in read 1" in message
        assert error_info.value.__cause__ is raised_errors[0]
        assert not (tmp_path / "all_failed.csv").exists()


class TestFailedTiles:
    """A failed read gives a failed tile; the manifest and the sidecar report it."""

    def test_failed_tile_is_in_the_manifest_and_the_sidecar(
        self, monkeypatch, three_points, tmp_path
    ):
        """One failed window read of three: manifest error column, tile summary, one warning."""
        _fail_reads(monkeypatch, local_adapter, "rio_mask", {2})

        _, failed_point_warnings = _run_and_record_warnings(
            three_points, _raster_config("tiles"), output_dir=tmp_path
        )

        tile_dir = tmp_path / "tiles" / "dem_local"
        manifest = pd.read_csv(tile_dir / "tiles_manifest.csv")
        assert list(manifest.columns) == ["occurrenceID", "tile_filename", "exported", "error"]
        assert manifest["exported"].tolist() == [True, False, True]
        assert manifest.loc[1, "error"] == _SECOND_READ_ERROR
        assert manifest["error"].isna().tolist() == [True, False, True]
        assert not (tile_dir / "T6WPKD-dem_local.tif").exists()

        sidecar = _read_sidecar(tmp_path / "tiles" / "tiles_metadata.json")
        assert sidecar["datasets"]["dem_local"]["quality"]["tiles"] == {
            "n_exported": 2,
            "n_failed": 1,
            "total": 3,
            "n_errors": 1,
            "ids_errors": ["T6WPKD"],
        }
        assert len(failed_point_warnings) == 1
        assert "tiles_manifest.csv" in failed_point_warnings[0]
        assert sidecar["warnings"]["failed_dem_local_200m"] == failed_point_warnings[0]

    def test_tile_outside_extent_is_not_a_failed_tile(self, three_points, tmp_path):
        """A tile not exported because its point is outside the extent has no error."""
        _, failed_point_warnings = _run_and_record_warnings(
            _with_outside_point(three_points), _raster_config("outside_tiles"), output_dir=tmp_path
        )

        manifest = pd.read_csv(tmp_path / "outside_tiles" / "dem_local" / "tiles_manifest.csv")
        assert list(manifest.columns) == ["occurrenceID", "tile_filename", "exported"]
        assert manifest["exported"].tolist() == [True, True, True, False]
        sidecar = _read_sidecar(tmp_path / "outside_tiles" / "outside_tiles_metadata.json")
        assert sidecar["datasets"]["dem_local"]["quality"]["tiles"] == {
            "n_exported": 3,
            "n_failed": 1,
            "total": 4,
            "n_errors": 0,
            "ids_errors": [],
        }
        assert failed_point_warnings == []

    def test_failed_reprojection_is_a_failed_tile(self, monkeypatch, three_points, tmp_path):
        """A read error while a tile is reprojected (resample_m) gives a failed tile."""
        _fail_reads(monkeypatch, rasterio.warp, "reproject", {2})

        _run_and_record_warnings(
            three_points, _raster_config("resampled", resample_m=25), output_dir=tmp_path
        )

        tile_dir = tmp_path / "resampled" / "dem_local"
        manifest = pd.read_csv(tile_dir / "tiles_manifest.csv")
        assert manifest["exported"].tolist() == [True, False, True]
        assert manifest.loc[1, "error"] == _SECOND_READ_ERROR
        assert sorted(path.name for path in tile_dir.glob("*.tif")) == [
            "T2T1DJ-dem_local.tif",
            "TCSZX1-dem_local.tif",
        ]

    def test_every_tile_failing_raises_with_the_first_error(
        self, monkeypatch, three_points, tmp_path
    ):
        """When every tile fails, extract() raises RuntimeError chained to the first error."""
        raised_errors = _fail_reads(monkeypatch, local_adapter, "rio_mask", None)

        with pytest.raises(RuntimeError, match="every point failed") as error_info:
            extract(three_points, _raster_config("all_tiles_failed"), output_dir=tmp_path)

        assert "First error: RasterioIOError: stub: damaged block in read 1" in str(
            error_info.value
        )
        assert error_info.value.__cause__ is raised_errors[0]


class TestSidecarJson:
    """The exception objects of failed points never reach the metadata sidecar."""

    @pytest.mark.parametrize("output_type", ["tabular", "raster"])
    def test_sidecar_with_a_failed_point_is_valid_json(
        self, monkeypatch, three_points, tmp_path, output_type
    ):
        """The sidecar of a run with a failed point is written and loads with json."""
        _fail_reads(monkeypatch, local_adapter, "rio_mask", {2})
        if output_type == "tabular":
            config = _tabular_config("json_check", ["mean"])
            sidecar_path = tmp_path / "json_check_metadata.json"
        else:
            config = _raster_config("json_check")
            sidecar_path = tmp_path / "json_check" / "json_check_metadata.json"

        _run_and_record_warnings(three_points, config, output_dir=tmp_path)

        sidecar = _read_sidecar(sidecar_path)
        quality = sidecar["datasets"]["dem_local"]["quality"]
        quality_summary = quality["100m"] if output_type == "tabular" else quality["tiles"]
        assert quality_summary["ids_errors"] == ["T6WPKD"]
