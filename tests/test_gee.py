"""Tests for GEE adapter — skipped when GEE authentication is unavailable."""

import json
import pandas as pd
import pytest
from biodata.enrich import enrich

try:
    from biodata.auth import init_gee

    init_gee()
    GEE_AVAILABLE = True
except Exception:
    GEE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GEE_AVAILABLE, reason="GEE authentication unavailable")


CATALOG = {
    "datasets": {
        "dem_aster": {
            "source": "earth_engine",
            "path": "projects/sat-io/open-datasets/ASTER/GDEM",
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


class TestGeeTabular:
    def test_stats(self, tmp_path):
        """GEE adapter returns non-null stats for known locations."""
        outputs = enrich(
            SAMPLE_DF,
            {
                "name": "gee_stats",
                "predictors": ["dem_aster"],
                "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 200},
            },
            catalog=CATALOG,
            out_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["gee_stats"])
        assert len(stats_df) == 2
        assert "dem_aster_mean_200m" in stats_df.columns
        assert stats_df["dem_aster_mean_200m"].notna().all()

    def test_point_reducer(self, tmp_path):
        """GEE point sampling returns a value per point."""
        outputs = enrich(
            SAMPLE_DF,
            {
                "name": "gee_point",
                "predictors": ["dem_aster"],
                "output": {"kind": "tabular", "reducers": ["point"], "window_m": 100},
            },
            catalog=CATALOG,
            out_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["gee_point"])
        assert "dem_aster_point" in stats_df.columns
        assert stats_df["dem_aster_point"].notna().any()


class TestGeeRaster:
    def test_export_tiles(self, tmp_path):
        """GEE raster export produces GeoTIFF files."""
        enrich(
            SAMPLE_DF,
            {
                "name": "gee_tiles",
                "predictors": ["dem_aster"],
                "output": {"kind": "raster", "window_m": 200},
            },
            catalog=CATALOG,
            out_dir=tmp_path,
        )

        tile_dir = tmp_path / "gee_tiles" / "dem_aster"
        tifs = list(tile_dir.glob("*.tif"))
        assert len(tifs) == 2

    def test_resample_m(self, tmp_path):
        """GEE export with resample_m produces correctly sized tiles."""
        import rasterio

        enrich(
            SAMPLE_DF,
            {
                "name": "gee_resamp",
                "predictors": ["dem_aster"],
                "output": {"kind": "raster", "window_m": 200, "resample_m": 50},
            },
            catalog=CATALOG,
            out_dir=tmp_path,
        )

        tile_dir = tmp_path / "gee_resamp" / "dem_aster"
        expected = round(200 / 50)  # 4x4
        for tif in tile_dir.glob("*.tif"):
            with rasterio.open(tif) as src:
                assert src.width == expected
                assert src.height == expected

    def test_metadata_json(self, tmp_path):
        """GEE raster output includes metadata with native CRS/scale."""
        enrich(
            SAMPLE_DF,
            {
                "name": "gee_meta",
                "predictors": ["dem_aster"],
                "output": {"kind": "raster", "window_m": 200},
            },
            catalog=CATALOG,
            out_dir=tmp_path,
        )

        meta_path = tmp_path / "gee_meta" / "gee_meta_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "dem_aster" in meta["features"]
        assert meta["features"]["dem_aster"]["source"] == "earth_engine"
