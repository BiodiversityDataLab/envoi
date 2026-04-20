"""Tests for the enrich() pipeline using local raster data."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import rasterio

from biodata.enrich import enrich

DATA_DIR = Path("data/for_testing")
SAMPLE_CSV = DATA_DIR / "adrian_example.csv"
DEM_TIF = DATA_DIR / "dem/TG4NHB-dem.tif"

CATALOG = {
    "datasets": {
        "dem_local": {
            "source": "local",
            "path": str(DEM_TIF),
        },
        "slope_local": {
            "source": "local",
            "path": str(DEM_TIF),
            "bands": 2,
        },
    }
}


@pytest.fixture
def sample_df():
    return pd.read_csv(SAMPLE_CSV)


# ------------------------------------------------------------------
# Tabular output
# ------------------------------------------------------------------

class TestTabular:
    def test_basic_stats(self, sample_df, tmp_path):
        """Stats csv has reducer columns, QC csv has QA columns."""
        outputs = enrich(sample_df, {
            "name": "dem_100m",
            "predictors": ["dem_local"],
            "output": {"kind": "tabular", "reducers": ["mean", "std"], "window_m": 100},
        }, catalog=CATALOG, out_dir=tmp_path)

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
        outputs = enrich(sample_df, {
            "name": "test",
            "predictors": ["dem_local"],
            "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 100},
        }, catalog=CATALOG, out_dir=tmp_path)

        result = pd.read_csv(outputs["test"])
        assert list(result["id"]) == list(sample_df["id"])

    def test_csv_format(self, sample_df, tmp_path):
        """Default format is csv; parquet can be requested explicitly."""
        outputs = enrich(sample_df, {
            "name": "csv_test",
            "predictors": ["dem_local"],
            "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 100},
        }, catalog=CATALOG, out_dir=tmp_path)

        assert outputs["csv_test"].suffix == ".csv"
        assert outputs["csv_test_qc"].suffix == ".csv"
        result = pd.read_csv(outputs["csv_test"])
        assert len(result) == len(sample_df)

    def test_multiple_reducers(self, sample_df, tmp_path):
        """All requested reducers produce columns."""
        reducers = ["mean", "median", "min", "max", "std"]
        outputs = enrich(sample_df, {
            "name": "multi",
            "predictors": ["dem_local"],
            "output": {"kind": "tabular", "reducers": reducers, "window_m": 200},
        }, catalog=CATALOG, out_dir=tmp_path)

        stats_df = pd.read_csv(outputs["multi"])
        for r in reducers:
            assert f"dem_local_{r}_200m" in stats_df.columns

    def test_multiple_predictors(self, sample_df, tmp_path):
        """Multiple predictors produce separate columns."""
        outputs = enrich(sample_df, {
            "name": "multi_pred",
            "predictors": ["dem_local", "slope_local"],
            "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 100},
        }, catalog=CATALOG, out_dir=tmp_path)

        stats_df = pd.read_csv(outputs["multi_pred"])
        assert "dem_local_mean_100m" in stats_df.columns
        assert "slope_local_mean_100m" in stats_df.columns

    def test_point_reducer(self, sample_df, tmp_path):
        """reducers: [point] samples exact pixel values."""
        outputs = enrich(sample_df, {
            "name": "point_test",
            "predictors": ["dem_local"],
            "output": {"kind": "tabular", "reducers": ["point"], "window_m": 100},
        }, catalog=CATALOG, out_dir=tmp_path)

        stats_df = pd.read_csv(outputs["point_test"])
        assert "dem_local_point" in stats_df.columns
        # Point values should be finite numbers for in-extent points
        assert stats_df["dem_local_point"].notna().any()

    def test_point_mixed_with_window_reducers(self, sample_df, tmp_path):
        """reducers: [mean, std, point] returns all three column families."""
        outputs = enrich(sample_df, {
            "name": "mixed",
            "predictors": ["dem_local"],
            "output": {
                "kind": "tabular",
                "reducers": ["mean", "std", "point"],
                "window_m": 100,
            },
        }, catalog=CATALOG, out_dir=tmp_path)

        stats_df = pd.read_csv(outputs["mixed"])
        for col in ("dem_local_mean_100m", "dem_local_std_100m", "dem_local_point"):
            assert col in stats_df.columns, f"missing {col}"
            assert stats_df[col].notna().any(), f"all-null {col}"


# ------------------------------------------------------------------
# Raster output
# ------------------------------------------------------------------

class TestRaster:
    def test_export_tiles(self, sample_df, tmp_path):
        """kind: raster produces one GeoTIFF per point."""
        outputs = enrich(sample_df, {
            "name": "tiles",
            "predictors": ["dem_local"],
            "output": {"kind": "raster", "window_m": 200},
        }, catalog=CATALOG, out_dir=tmp_path)

        tile_dir = tmp_path / "tiles" / "dem_local"
        tifs = list(tile_dir.glob("*.tif"))
        assert len(tifs) == len(sample_df)

    def test_tile_dimensions_consistent(self, sample_df, tmp_path):
        """All exported tiles have identical dimensions."""
        enrich(sample_df, {
            "name": "dim_test",
            "predictors": ["dem_local"],
            "output": {"kind": "raster", "window_m": 200},
        }, catalog=CATALOG, out_dir=tmp_path)

        tile_dir = tmp_path / "dim_test" / "dem_local"
        sizes = set()
        for tif in tile_dir.glob("*.tif"):
            with rasterio.open(tif) as src:
                sizes.add((src.width, src.height))
        assert len(sizes) == 1, f"Inconsistent tile sizes: {sizes}"

    def test_resample_m(self, sample_df, tmp_path):
        """resample_m produces tiles with expected pixel count."""
        enrich(sample_df, {
            "name": "resamp",
            "predictors": ["dem_local"],
            "output": {"kind": "raster", "window_m": 200, "resample_m": 25},
        }, catalog=CATALOG, out_dir=tmp_path)

        tile_dir = tmp_path / "resamp" / "dem_local"
        expected_pixels = round(200 / 25)  # 8
        for tif in tile_dir.glob("*.tif"):
            with rasterio.open(tif) as src:
                assert src.width == expected_pixels
                assert src.height == expected_pixels

    def test_raster_metadata_json(self, sample_df, tmp_path):
        """Raster output writes a sidecar metadata JSON."""
        enrich(sample_df, {
            "name": "meta_test",
            "predictors": ["dem_local"],
            "output": {"kind": "raster", "window_m": 200},
        }, catalog=CATALOG, out_dir=tmp_path)

        meta_path = tmp_path / "meta_test" / "meta_test_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["config"]["window_m"] == 200
        assert "dem_local" in meta["features"]


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------

class TestMetadata:
    def test_metadata_structure(self, sample_df, tmp_path):
        """Metadata JSON has run, config, features, quality sections."""
        outputs = enrich(sample_df, {
            "name": "meta",
            "predictors": ["dem_local"],
            "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 100},
        }, catalog=CATALOG, out_dir=tmp_path)

        meta_path = outputs["meta"].parent / "meta_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert "run" in meta
        assert "timestamp" in meta["run"]
        assert "package_version" in meta["run"]
        assert meta["run"]["n_points"] == len(sample_df)

        assert meta["config"]["name"] == "meta"
        assert meta["config"]["reducers"] == ["mean"]

        feat = meta["features"]["dem_local"]
        assert feat["source"] == "local"
        assert "native_crs" in feat
        assert "native_spatial_resolution_m" in feat

        assert "quality" in meta
        assert "dem_local" in meta["quality"]


# ------------------------------------------------------------------
# Multiple outputs (list cfg)
# ------------------------------------------------------------------

class TestMultipleOutputs:
    def test_list_cfg(self, sample_df, tmp_path):
        """Passing a list of dicts processes all outputs."""
        outputs = enrich(sample_df, [
            {
                "name": "stats_out",
                "predictors": ["dem_local"],
                "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 100},
            },
            {
                "name": "tiles_out",
                "predictors": ["dem_local"],
                "output": {"kind": "raster", "window_m": 200},
            },
        ], catalog=CATALOG, out_dir=tmp_path)

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
            enrich(df, {
                "name": "fail",
                "predictors": ["dem_local"],
                "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 100},
            }, catalog=CATALOG, out_dir=tmp_path)

    def test_unknown_feature(self, sample_df, tmp_path):
        """Raises KeyError for a feature not in catalog."""
        with pytest.raises(KeyError, match="nonexistent"):
            enrich(sample_df, {
                "name": "fail",
                "predictors": ["nonexistent"],
                "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 100},
            }, catalog=CATALOG, out_dir=tmp_path)

    def test_invalid_kind(self, sample_df, tmp_path):
        """Raises ValueError for an unknown output kind."""
        with pytest.raises(ValueError, match="Unknown output kind"):
            enrich(sample_df, {
                "name": "fail",
                "predictors": ["dem_local"],
                "output": {"kind": "banana", "window_m": 100},
            }, catalog=CATALOG, out_dir=tmp_path)


# ------------------------------------------------------------------
# Multi-band regression
# ------------------------------------------------------------------

def test_multiband_local_with_per_band_nodata(tmp_path):
    """fetch_values must tolerate bands with different nodata patterns."""
    from rasterio.transform import from_origin

    arr = np.arange(30 * 30, dtype="float32").reshape(1, 30, 30)
    arr3 = np.concatenate([arr, arr * 2, arr * 3], axis=0)
    arr3[0, 5, 5] = -9999    # nodata in band 1 only
    arr3[1, 10, 10] = -9999  # nodata in band 2 only
    path = tmp_path / "mb.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=30, width=30, count=3,
        dtype="float32", crs="EPSG:32634", nodata=-9999,
        transform=from_origin(349170, 6986638, 10, 10),
    ) as dst:
        dst.write(arr3)

    catalog = {"datasets": {"mb": {
        "source": "local", "path": str(path), "data_type": "continuous",
        "bands": [1, 2, 3],
    }}}
    df = pd.DataFrame({"id": ["a"], "lat": [62.9768], "lon": [18.0268]})
    outputs = enrich(df, {
        "name": "mb_test", "predictors": ["mb"],
        "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 200, "format": "csv"},
    }, catalog=catalog, out_dir=tmp_path)

    stats_df = pd.read_csv(outputs["mb_test"])
    for col in ("mb_b1_mean_200m", "mb_b2_mean_200m", "mb_b3_mean_200m"):
        assert col in stats_df.columns
        assert stats_df[col].notna().all()
