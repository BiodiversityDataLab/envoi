"""Tests for GEE adapter — skipped when GEE authentication is unavailable."""

import json

import pandas as pd
import pytest
from envoi.extract import extract
from envoi import update_catalog, reset_catalog

try:
    from envoi.auth import init_gee

    init_gee()
    GEE_AVAILABLE = True
except Exception:
    GEE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GEE_AVAILABLE, reason="GEE authentication unavailable")

CATALOG = {
    "datasets": {
        "dem_aster": {
            "data_source": "earth_engine",
            "path": "projects/sat-io/open-datasets/ASTER/GDEM",
            "data_type": "continuous",
        },
    }
}

# A couple of known points in Sweden (same as local test data)
SAMPLE_DF = pd.DataFrame(
    {
        "id": ["A", "B"],
        "lat": [62.9768783, 62.9812956],
        "lon": [18.026823, 18.0309905],
        "date": ["2020-06-01", "2020-06-01"],
    }
)


@pytest.fixture(autouse=True)
def register_test_catalog():
    """Register the GEE test datasets before each test and clean up after."""
    update_catalog(CATALOG)
    yield
    reset_catalog()


class TestGeeTabular:
    def test_stats(self, tmp_path):
        """GEE adapter returns non-null stats for known locations."""
        outputs = extract(
            SAMPLE_DF,
            {
                "batch_id": "gee_stats",
                "datasets": ["dem_aster"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 200,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["gee_stats"])
        assert len(stats_df) == 2
        assert "dem_aster_mean_200m" in stats_df.columns
        assert stats_df["dem_aster_mean_200m"].notna().all()

    def test_point_reducer(self, tmp_path):
        """GEE point sampling returns a value per point."""
        outputs = extract(
            SAMPLE_DF,
            {
                "batch_id": "gee_point",
                "datasets": ["dem_aster"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["point"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["gee_point"])
        assert "dem_aster_point" in stats_df.columns
        assert stats_df["dem_aster_point"].notna().any()


class TestGeeRaster:
    def test_export_tiles(self, tmp_path):
        """GEE raster export produces GeoTIFF files."""
        extract(
            SAMPLE_DF,
            {
                "batch_id": "gee_tiles",
                "datasets": ["dem_aster"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        tile_dir = tmp_path / "gee_tiles" / "dem_aster"
        tifs = list(tile_dir.glob("*.tif"))
        assert len(tifs) == 2

    def test_resample_m(self, tmp_path):
        """GEE export with resample_m produces correctly sized tiles."""
        import rasterio

        extract(
            SAMPLE_DF,
            {
                "batch_id": "gee_resamp",
                "datasets": ["dem_aster"],
                "settings": {"output_type": "raster", "window_size_m": 200, "resample_m": 50},
            },
            output_dir=tmp_path,
        )

        tile_dir = tmp_path / "gee_resamp" / "dem_aster"
        expected = round(200 / 50)  # 4x4
        for tif in tile_dir.glob("*.tif"):
            with rasterio.open(tif) as src:
                assert src.width == expected
                assert src.height == expected

    def test_metadata_json(self, tmp_path):
        """GEE raster output includes metadata with native CRS/scale."""
        extract(
            SAMPLE_DF,
            {
                "batch_id": "gee_meta",
                "datasets": ["dem_aster"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        meta_path = tmp_path / "gee_meta" / "gee_meta_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "dem_aster" in meta["datasets"]
        assert meta["datasets"]["dem_aster"]["data_source"] == "earth_engine"
