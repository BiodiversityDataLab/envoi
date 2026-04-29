"""Tests for the extract() pipeline using local raster data."""

from pathlib import Path
import json

import pandas as pd
import pytest
import rasterio

from biodata.extract import extract
from biodata import update_catalog, reset_catalog

DATA_DIR = Path("data/for_testing")
SAMPLE_CSV = DATA_DIR / "adrian_example.csv"
DEM_TIF = DATA_DIR / "dem/TG4NHB-dem.tif"

CATALOG = {
    "datasets": {
        "dem_local": {
            "data_source": "local",
            "path": str(DEM_TIF),
            "bands": 1,
        },
        "slope_local": {
            "data_source": "local",
            "path": str(DEM_TIF),
            "bands": 2,
        },
    }
}


@pytest.fixture(autouse=True)
def register_test_catalog():
    """Register the local test datasets before each test and clean up after."""
    update_catalog(CATALOG)
    yield
    reset_catalog()


@pytest.fixture
def sample_df():
    return pd.read_csv(SAMPLE_CSV)


# ------------------------------------------------------------------
# Tabular output
# ------------------------------------------------------------------


class TestTabular:
    def test_basic_stats(self, sample_df, tmp_path):
        """Stats csv has reducer columns, QC csv has QA columns."""
        outputs = extract(
            sample_df,
            {
                "batch_id": "dem_100m",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean", "std"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["dem_100m"])
        qc_df = pd.read_csv(outputs["dem_100m_qc"])

        assert len(stats_df) == len(sample_df)
        assert len(qc_df) == len(sample_df)

        # Reducer columns in stats only
        assert "dem_local_mean_100m" in stats_df.columns
        assert "dem_local_std_100m" in stats_df.columns
        assert "dem_local_mean_100m" not in qc_df.columns

        # QA columns in QC only
        assert "dem_local_coverage_pct_100m" in qc_df.columns
        assert "dem_local_coverage_pct_100m" not in stats_df.columns

    def test_row_order_preserved(self, sample_df, tmp_path):
        """IDs in output match input order."""
        outputs = extract(
            sample_df,
            {
                "batch_id": "test",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        result = pd.read_csv(outputs["test"])
        assert list(result["id"]) == list(sample_df["id"])

    def test_csv_format(self, sample_df, tmp_path):
        """Default format is csv; parquet can be requested explicitly."""
        outputs = extract(
            sample_df,
            {
                "batch_id": "csv_test",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        assert outputs["csv_test"].suffix == ".csv"
        assert outputs["csv_test_qc"].suffix == ".csv"
        result = pd.read_csv(outputs["csv_test"])
        assert len(result) == len(sample_df)

    def test_multiple_reducers(self, sample_df, tmp_path):
        """All requested reducers produce columns."""
        reducers = ["mean", "median", "min", "max", "std"]
        outputs = extract(
            sample_df,
            {
                "batch_id": "multi",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": reducers,
                    "window_size_m": 200,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["multi"])
        for r in reducers:
            assert f"dem_local_{r}_200m" in stats_df.columns

    def test_multiple_datasets(self, sample_df, tmp_path):
        """Multiple datasets produce separate columns."""
        outputs = extract(
            sample_df,
            {
                "batch_id": "multi_pred",
                "datasets": ["dem_local", "slope_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["multi_pred"])
        assert "dem_local_mean_100m" in stats_df.columns
        assert "slope_local_mean_100m" in stats_df.columns

    def test_point_reducer(self, sample_df, tmp_path):
        """reducers: [point] samples exact pixel values."""
        outputs = extract(
            sample_df,
            {
                "batch_id": "point_test",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["point"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["point_test"])
        assert "dem_local_point" in stats_df.columns
        # Point values should be finite numbers for in-extent points
        assert stats_df["dem_local_point"].notna().any()


# ------------------------------------------------------------------
# Raster output
# ------------------------------------------------------------------


class TestRaster:
    def test_export_tiles(self, sample_df, tmp_path):
        """kind: raster produces one GeoTIFF per point."""
        extract(
            sample_df,
            {
                "batch_id": "tiles",
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        tile_dir = tmp_path / "tiles" / "dem_local"
        tifs = list(tile_dir.glob("*.tif"))
        assert len(tifs) == len(sample_df)

    def test_tile_dimensions_consistent(self, sample_df, tmp_path):
        """All exported tiles have identical dimensions."""
        extract(
            sample_df,
            {
                "batch_id": "dim_test",
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        tile_dir = tmp_path / "dim_test" / "dem_local"
        sizes = set()
        for tif in tile_dir.glob("*.tif"):
            with rasterio.open(tif) as src:
                sizes.add((src.width, src.height))
        assert len(sizes) == 1, f"Inconsistent tile sizes: {sizes}"

    def test_resample_m(self, sample_df, tmp_path):
        """resample_m produces tiles with expected pixel count."""
        extract(
            sample_df,
            {
                "batch_id": "resamp",
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": 200, "resample_m": 25},
            },
            output_dir=tmp_path,
        )

        tile_dir = tmp_path / "resamp" / "dem_local"
        expected_pixels = round(200 / 25)  # 8
        for tif in tile_dir.glob("*.tif"):
            with rasterio.open(tif) as src:
                assert src.width == expected_pixels
                assert src.height == expected_pixels

    def test_raster_metadata_json(self, sample_df, tmp_path):
        """Raster output writes a sidecar metadata JSON."""
        extract(
            sample_df,
            {
                "batch_id": "meta_test",
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        meta_path = tmp_path / "meta_test" / "meta_test_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["config"]["window_size_m"] == 200
        assert "dem_local" in meta["datasets"]


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------


class TestMetadata:
    def test_metadata_structure(self, sample_df, tmp_path):
        """Metadata JSON has run, config, datasets, quality sections."""
        outputs = extract(
            sample_df,
            {
                "batch_id": "meta",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        meta_path = outputs["meta"].parent / "meta_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert "run" in meta
        assert "timestamp" in meta["run"]
        assert "package_version" in meta["run"]
        assert meta["run"]["n_points"] == len(sample_df)

        assert meta["config"]["batch_id"] == "meta"
        assert meta["config"]["statistics"] == ["mean"]

        ds_meta = meta["datasets"]["dem_local"]
        assert ds_meta["data_source"] == "local"
        assert "native_crs" in ds_meta
        assert "native_spatial_resolution_m" in ds_meta

        assert "quality" in meta["datasets"]["dem_local"]


# ------------------------------------------------------------------
# Multiple outputs (list config)
# ------------------------------------------------------------------


class TestMultipleOutputs:
    def test_list_config(self, sample_df, tmp_path):
        """Passing a list of dicts processes all outputs."""
        outputs = extract(
            sample_df,
            [
                {
                    "batch_id": "stats_out",
                    "datasets": ["dem_local"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean"],
                        "window_size_m": 100,
                    },
                },
                {
                    "batch_id": "tiles_out",
                    "datasets": ["dem_local"],
                    "settings": {"output_type": "raster", "window_size_m": 200},
                },
            ],
            output_dir=tmp_path,
        )

        assert "stats_out" in outputs
        assert outputs["stats_out"].exists()
        tile_dir = tmp_path / "tiles_out" / "dem_local"
        assert any(tile_dir.glob("*.tif"))


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestErrors:
    def test_missing_columns(self, tmp_path):
        """Raises ValueError if required columns are missing."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="missing required column"):
            extract(
                df,
                {
                    "batch_id": "fail",
                    "datasets": ["dem_local"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean"],
                        "window_size_m": 100,
                    },
                },
                output_dir=tmp_path,
            )

    def test_unknown_dataset(self, sample_df, tmp_path):
        """Raises KeyError for a dataset not in catalog."""
        with pytest.raises(KeyError, match="nonexistent"):
            extract(
                sample_df,
                {
                    "batch_id": "fail",
                    "datasets": ["nonexistent"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean"],
                        "window_size_m": 100,
                    },
                },
                output_dir=tmp_path,
            )

    def test_invalid_kind(self, sample_df, tmp_path):
        """Raises ValueError for an unknown output kind."""
        with pytest.raises(ValueError, match="Unknown or missing output_type"):
            extract(
                sample_df,
                {
                    "batch_id": "fail",
                    "datasets": ["dem_local"],
                    "settings": {"output_type": "banana", "window_size_m": 100},
                },
                output_dir=tmp_path,
            )

    def test_tabular_requires_statistics(self, sample_df, tmp_path):
        """Raises ValueError when tabular output omits the statistics list."""
        with pytest.raises(ValueError, match="statistics"):
            extract(
                sample_df,
                {
                    "batch_id": "fail",
                    "datasets": ["dem_local"],
                    "settings": {"output_type": "tabular", "window_size_m": 100},
                },
                output_dir=tmp_path,
            )
