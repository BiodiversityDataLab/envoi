"""Tests for new Earth Engine catalog features.

Skipped when GEE authentication is unavailable.
Add a new test class or method here whenever a new GEE dataset is added to the catalog.
"""

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

# Known points in Sweden — within extent of all global datasets
SAMPLE_DF = pd.DataFrame(
    {
        "id": ["A", "B"],
        "lat": [62.9768783, 62.9812956],
        "lon": [18.026823, 18.0309905],
        "date": ["2020-06-01", "2020-06-01"],
    }
)


def _make_catalog(*datasets, data_type="continuous"):
    """Helper to build a catalog dict from (name, path) tuples.

    `data_type` is required for earth_engine entries by the catalog validator,
    so it gets stamped onto every entry here. Defaults to "continuous" since
    that fits most ecological rasters; pass data_type="categorical" for land
    cover etc.
    """
    return {
        "datasets": {
            name: {"data_source": "earth_engine", "path": path, "data_type": data_type}
            for name, path in datasets
        }
    }


def _run_stats(df, dataset_name, catalog, tmp_path, reducers=None):
    """Run tabular stats and return the stats DataFrame."""
    reducers = reducers or ["mean"]
    update_catalog(catalog)
    try:
        outputs = extract(
            df,
            {
                "batch_id": "test",
                "datasets": [dataset_name],
                "settings": {
                    "output_type": "tabular",
                    "statistics": reducers,
                    "window_size_m": 200,
                },
            },
            output_dir=tmp_path,
        )
        return pd.read_csv(outputs["test"])
    finally:
        reset_catalog()


# ------------------------------------------------------------------
# Static datasets (IMAGE type — no date filtering)
# ------------------------------------------------------------------


class TestStaticDatasets:
    """Datasets that are single images, not time series."""

    def test_dem_aster(self, tmp_path):
        cat = _make_catalog(("dem_aster", "projects/sat-io/open-datasets/ASTER/GDEM"))
        df = _run_stats(SAMPLE_DF, "dem_aster", cat, tmp_path)
        assert df["dem_aster_mean_200m"].notna().all()
        # Elevation in Sweden should be roughly 0-1000m
        assert df["dem_aster_mean_200m"].between(0, 1000).all()

    def test_dem_glo30(self, tmp_path):
        # GLO30 has 5 bands (DEM, EDM, FLM, HEM, WBM) — check the DEM band
        cat = _make_catalog(("dem_glo30", "COPERNICUS/DEM/GLO30"))
        df = _run_stats(SAMPLE_DF, "dem_glo30", cat, tmp_path)
        assert df["dem_glo30_DEM_mean_200m"].notna().all()

    def test_bioclim(self, tmp_path):
        # Bioclim has 19 bands — check first and last
        cat = _make_catalog(("climate_bioclim", "WORLDCLIM/V1/BIO"))
        df = _run_stats(SAMPLE_DF, "climate_bioclim", cat, tmp_path)
        assert df["climate_bioclim_bio01_mean_200m"].notna().all()
        assert df["climate_bioclim_bio19_mean_200m"].notna().all()
        bioclim_cols = [c for c in df.columns if c.startswith("climate_bioclim_bio")]
        assert len(bioclim_cols) == 19

    def test_human_impact_index(self, tmp_path):
        cat = _make_catalog(("hii", "projects/HII/v1/hii"))
        df = _run_stats(SAMPLE_DF, "hii", cat, tmp_path)
        assert df["hii_mean_200m"].notna().all()

    def test_era5_monthly(self, tmp_path):
        cat = {
            "datasets": {
                "era5": {
                    "data_source": "earth_engine",
                    "path": "ECMWF/ERA5/MONTHLY",
                    "data_type": "continuous",
                }
            }
        }
        df = _run_stats(SAMPLE_DF, "era5", cat, tmp_path)
        # ERA5 has 9 bands — check first one
        era5_cols = [c for c in df.columns if c.startswith("era5_") and "_mean_" in c]
        assert len(era5_cols) > 0
        assert df[era5_cols[0]].notna().all()

    def test_satellite_embeddings(self, tmp_path):
        cat = {
            "datasets": {
                "sat_emb": {
                    "data_source": "earth_engine",
                    "path": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
                    "data_type": "continuous",
                    "dataset_spec": {
                        "use_utm_zone": True,
                        "collection_date_policy": "contains",
                    },
                }
            }
        }
        df = _run_stats(SAMPLE_DF, "sat_emb", cat, tmp_path)
        # 64-band embeddings — check at least one
        emb_cols = [c for c in df.columns if c.startswith("sat_emb_") and "_mean_" in c]
        assert len(emb_cols) == 64
        assert df[emb_cols[0]].notna().all()


# ------------------------------------------------------------------
# Land use / land cover
# ------------------------------------------------------------------


class TestLandCover:
    def test_esa_worldcover(self, tmp_path):
        cat = _make_catalog(("lulc", "ESA/WorldCover/v200"), data_type="categorical")
        df = _run_stats(SAMPLE_DF, "lulc", cat, tmp_path)
        assert df["lulc_mean_200m"].notna().all()

    def test_cgls_lc100(self, tmp_path):
        cat = _make_catalog(
            ("lc100", "COPERNICUS/Landcover/100m/Proba-V-C3/Global"), data_type="categorical"
        )
        df = _run_stats(SAMPLE_DF, "lc100", cat, tmp_path)
        lc_cols = [c for c in df.columns if c.startswith("lc100_") and "_mean_" in c]
        assert len(lc_cols) > 0


# ------------------------------------------------------------------
# Point sampling across datasets
# ------------------------------------------------------------------


class TestPointSampling:
    """Verify point reducer works for different dataset types."""

    def test_point_dem_aster(self, tmp_path):
        cat = _make_catalog(("dem_aster", "projects/sat-io/open-datasets/ASTER/GDEM"))
        update_catalog(cat)
        try:
            outputs = extract(
                SAMPLE_DF,
                {
                    "batch_id": "pt",
                    "datasets": ["dem_aster"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["point"],
                        "window_size_m": 100,
                    },
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()
        df = pd.read_csv(outputs["pt"])
        assert df["dem_aster_point"].notna().all()

    def test_point_dem_glo30_with_window_stats(self, tmp_path):
        # Regression test: dem_glo30 is an IMAGE_COLLECTION with tiled global
        # coverage and derived bands (slope, aspect). When "point" was combined
        # with window reducers like "mean"/"std", the adapter cached a
        # no-coords global static image during the band-name probe and then
        # reused it for per-point sampling, causing img.sample() to return
        # empty props and silently dropping every "_point" column from the
        # output.
        cat = {
            "datasets": {
                "dem_glo30": {
                    "data_source": "earth_engine",
                    "path": "COPERNICUS/DEM/GLO30",
                    "data_type": "continuous",
                    "bands": ["DEM"],
                    "derived_bands": ["slope", "aspect"],
                }
            }
        }
        update_catalog(cat)
        try:
            outputs = extract(
                SAMPLE_DF,
                {
                    "batch_id": "pt",
                    "datasets": ["dem_glo30"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean", "std", "point"],
                        "window_size_m": 200,
                    },
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()
        df = pd.read_csv(outputs["pt"])
        # All three bands must produce point columns alongside window stats.
        for band in ("DEM", "slope", "aspect"):
            assert f"dem_glo30_{band}_point" in df.columns
            assert df[f"dem_glo30_{band}_point"].notna().all()

    def test_point_worldcover(self, tmp_path):
        cat = _make_catalog(("lulc", "ESA/WorldCover/v200"), data_type="categorical")
        update_catalog(cat)
        try:
            outputs = extract(
                SAMPLE_DF,
                {
                    "batch_id": "pt",
                    "datasets": ["lulc"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["point"],
                        "window_size_m": 100,
                    },
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()
        df = pd.read_csv(outputs["pt"])
        assert df["lulc_point"].notna().all()


# ------------------------------------------------------------------
# Raster export across datasets
# ------------------------------------------------------------------


class TestRasterExport:
    """Verify raster export works for different dataset types."""

    def test_tiles_dem_glo30(self, tmp_path):
        cat = _make_catalog(("dem_glo30", "COPERNICUS/DEM/GLO30"))
        update_catalog(cat)
        try:
            extract(
                SAMPLE_DF,
                {
                    "batch_id": "tiles",
                    "datasets": ["dem_glo30"],
                    "settings": {"output_type": "raster", "window_size_m": 200},
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()
        tifs = list((tmp_path / "tiles" / "dem_glo30").glob("*.tif"))
        assert len(tifs) == 2

    def test_tiles_worldcover_resample(self, tmp_path):
        import rasterio

        cat = _make_catalog(("lulc", "ESA/WorldCover/v200"), data_type="categorical")
        update_catalog(cat)
        try:
            extract(
                SAMPLE_DF,
                {
                    "batch_id": "tiles",
                    "datasets": ["lulc"],
                    "settings": {"output_type": "raster", "window_size_m": 200, "resample_m": 50},
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()
        expected = round(200 / 50)  # 4x4
        for tif in (tmp_path / "tiles" / "lulc").glob("*.tif"):
            with rasterio.open(tif) as src:
                assert src.width == expected
                assert src.height == expected


# ------------------------------------------------------------------
# Automatic date selection for ImageCollections
# ------------------------------------------------------------------


class TestAutoDateSelection:
    """Verify automatic nearest-image date selection for collections."""

    def test_collection_no_date_column(self, tmp_path):
        """DataFrame without a date column should use most recent image."""
        df_no_date = pd.DataFrame(
            {
                "id": ["A", "B"],
                "lat": [62.9768783, 62.9812956],
                "lon": [18.026823, 18.0309905],
            }
        )
        cat = _make_catalog(("climate_bioclim", "WORLDCLIM/V1/BIO"))
        df = _run_stats(df_no_date, "climate_bioclim", cat, tmp_path)
        assert df["climate_bioclim_bio01_mean_200m"].notna().all()

    def test_dem_glo30_no_date_column(self, tmp_path):
        """DEM collection without date column should still return values."""
        df_no_date = pd.DataFrame(
            {
                "id": ["A", "B"],
                "lat": [62.9768783, 62.9812956],
                "lon": [18.026823, 18.0309905],
            }
        )
        cat = _make_catalog(("dem_glo30", "COPERNICUS/DEM/GLO30"))
        df = _run_stats(df_no_date, "dem_glo30", cat, tmp_path)
        assert df["dem_glo30_DEM_mean_200m"].notna().all()

    def test_date_clamping_to_range(self, tmp_path):
        """Dates outside collection range should clamp to nearest boundary."""
        df_old_date = pd.DataFrame(
            {
                "id": ["A", "B"],
                "lat": [62.9768783, 62.9812956],
                "lon": [18.026823, 18.0309905],
                "date": ["1920-01-01", "2099-01-01"],
            }
        )
        cat = _make_catalog(("era5", "ECMWF/ERA5/MONTHLY"))
        df = _run_stats(df_old_date, "era5", cat, tmp_path)
        era5_cols = [c for c in df.columns if c.startswith("era5_") and "_mean_" in c]
        assert len(era5_cols) > 0
        assert df[era5_cols[0]].notna().all()
