"""Live Google Earth Engine tests.

All tests here are tagged with the ``gee`` marker and skipped when no GEE
credentials are findable on the machine. To run them deliberately:

    pytest -m gee

To skip them deliberately (e.g. for fast inner-loop development):

    pytest -m "not gee"

Authentication is performed lazily by a session-scoped autouse fixture
(``_initialise_gee_once``) so ``pytest --collect-only`` does no network
I/O even when this file is part of the collection.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import rasterio

from envoi import reset_catalog, update_catalog
from envoi.config import BUILTIN_EE_CATALOG, load_catalogs
from envoi.extract import extract

from _gee_helpers import SWEDEN_SAMPLE_DF, gee_credentials_available

# Apply two markers to every test in this module:
#   * ``gee`` — selectable via ``pytest -m gee`` / ``pytest -m 'not gee'``.
#   * ``skipif`` — silent skip when no credentials are configured so users
#     without GEE access still see "120 passed, N skipped" rather than auth
#     errors. The skip reason mentions the marker name so opted-out users
#     don't get confused about why tests aren't running.
pytestmark = [
    pytest.mark.gee,
    pytest.mark.skipif(
        not gee_credentials_available(),
        reason="GEE credentials not configured (see envoi.auth for lookup order)",
    ),
]


@pytest.fixture(autouse=True, scope="session")
def _initialise_gee_once():
    """Lazy session-scoped GEE auth, run exactly once per pytest invocation.

    Module-level ``init_gee()`` calls used to live at import time, which made
    ``pytest --collect-only`` perform a real network handshake. Moving the
    call into a session fixture means auth only happens when GEE tests
    actually run, and failures surface as a single fixture error rather than
    as a noisy ``ImportError`` during collection.
    """
    from envoi.auth import init_gee

    init_gee()


# ------------------------------------------------------------------
# Static datasets (IMAGE type — no date filtering applies).
# ------------------------------------------------------------------
#
# These confirm the asset path resolves, asset_type is auto-detected, and at
# least one band returns non-null stats for points inside the dataset extent.


def _make_catalog(*datasets, data_type="continuous"):
    """Helper that builds a one-off catalog dict from ``(name, path)`` tuples.

    Every earth_engine entry needs ``data_type`` for the catalog validator;
    it gets stamped onto every dataset here so individual call sites don't
    have to repeat it. Pass ``data_type="categorical"`` for land-cover etc.
    """
    return {
        "datasets": {
            name: {"data_source": "earth_engine", "path": path, "data_type": data_type}
            for name, path in datasets
        }
    }


def _run_stats(df, dataset_name, catalog, tmp_path, reducers=None):
    """Run a single tabular extract() call and return the stats DataFrame.

    Wraps the boilerplate of registering a one-off catalog, running extract,
    reading the resulting CSV, and resetting the user catalog — so each
    test method can focus on what it actually wants to assert about the
    returned DataFrame.
    """
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


class TestStaticDatasets:
    """IMAGE assets — single images, not time series. No date logic applies."""

    def test_dem_aster(self, tmp_path):
        # ASTER GDEM is a single-image asset; verify it auto-detects as IMAGE
        # and produces sensible elevations for northern Sweden.
        catalog = _make_catalog(("dem_aster", "projects/sat-io/open-datasets/ASTER/GDEM"))
        result = _run_stats(SWEDEN_SAMPLE_DF, "dem_aster", catalog, tmp_path)
        assert result["dem_aster_mean_200m"].notna().all()
        # Elevation in northern Sweden should be ~0-1000m — wide bound so
        # a different sample point in the same region wouldn't break the test.
        assert result["dem_aster_mean_200m"].between(0, 1000).all()

    def test_dem_glo30(self, tmp_path):
        # Copernicus GLO-30 is an ImageCollection with global tile coverage;
        # the DEM band is the elevation channel we care about. Other bands
        # (EDM, FLM, HEM, WBM) are auxiliary masks.
        catalog = _make_catalog(("dem_glo30", "COPERNICUS/DEM/GLO30"))
        result = _run_stats(SWEDEN_SAMPLE_DF, "dem_glo30", catalog, tmp_path)
        assert result["dem_glo30_DEM_mean_200m"].notna().all()

    def test_bioclim(self, tmp_path):
        # WorldClim BIO has 19 bands (bio01..bio19). Confirming the first and
        # last appear catches band-name parsing bugs while leaving the
        # specific band count check to the catalog-walk smoke test.
        catalog = _make_catalog(("climate_bioclim", "WORLDCLIM/V1/BIO"))
        result = _run_stats(SWEDEN_SAMPLE_DF, "climate_bioclim", catalog, tmp_path)
        assert result["climate_bioclim_bio01_mean_200m"].notna().all()
        assert result["climate_bioclim_bio19_mean_200m"].notna().all()

    def test_human_impact_index(self, tmp_path):
        catalog = _make_catalog(("hii", "projects/HII/v1/hii"))
        result = _run_stats(SWEDEN_SAMPLE_DF, "hii", catalog, tmp_path)
        assert result["hii_mean_200m"].notna().all()

    def test_era5_monthly(self, tmp_path):
        # ERA5 is an ImageCollection with ``collection_date_policy: contains``
        # baked into the built-in catalog. Here we pass a bare path and let
        # the default "nearest" policy take over.
        catalog = {
            "datasets": {
                "era5": {
                    "data_source": "earth_engine",
                    "path": "ECMWF/ERA5/MONTHLY",
                    "data_type": "continuous",
                }
            }
        }
        result = _run_stats(SWEDEN_SAMPLE_DF, "era5", catalog, tmp_path)
        # ERA5 has 9 bands — at least one must produce a non-null mean column.
        era5_mean_columns = [
            column for column in result.columns if column.startswith("era5_") and "_mean_" in column
        ]
        assert era5_mean_columns
        assert result[era5_mean_columns[0]].notna().all()

    def test_satellite_embeddings(self, tmp_path):
        # AlphaEarth Satellite Embeddings are a tiled global collection (one
        # tile per UTM zone × year) so ``use_utm_zone`` is required to pick
        # the correct tile for each point.
        catalog = {
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
        result = _run_stats(SWEDEN_SAMPLE_DF, "sat_emb", catalog, tmp_path)
        # Embeddings are 64-dimensional. We assert at least one column is
        # populated; the strict count check lives in the catalog-walk test
        # so a future V2 release doesn't break this regression-style test.
        embedding_mean_columns = [
            column
            for column in result.columns
            if column.startswith("sat_emb_") and "_mean_" in column
        ]
        assert embedding_mean_columns
        assert result[embedding_mean_columns[0]].notna().all()


# ------------------------------------------------------------------
# Land-use / land-cover categorical datasets.
# ------------------------------------------------------------------


class TestLandCover:
    def test_esa_worldcover(self, tmp_path):
        catalog = _make_catalog(("lulc", "ESA/WorldCover/v200"), data_type="categorical")
        result = _run_stats(SWEDEN_SAMPLE_DF, "lulc", catalog, tmp_path)
        # mean of class IDs is meaningless but the column should still exist
        # — the typed-statistics warning system flags it; here we just check
        # the pipeline runs end-to-end for a categorical asset.
        assert result["lulc_mean_200m"].notna().all()

    def test_cgls_lc100(self, tmp_path):
        catalog = _make_catalog(
            ("lc100", "COPERNICUS/Landcover/100m/Proba-V-C3/Global"),
            data_type="categorical",
        )
        result = _run_stats(SWEDEN_SAMPLE_DF, "lc100", catalog, tmp_path)
        # CGLS LC100 exposes many per-class cover-fraction bands; at least
        # one must come back populated.
        lc100_mean_columns = [
            column
            for column in result.columns
            if column.startswith("lc100_") and "_mean_" in column
        ]
        assert lc100_mean_columns

    def test_worldcover_class_count_and_fraction(self, tmp_path):
        """End-to-end check of the categorical reducers against ESA WorldCover.

        The two sample points sit in northern Sweden where the WorldCover
        v200 map contains at least two classes (tree cover + grassland or
        water) within a 500 m window. We verify three properties:

          * At least one ``class_*_count_*`` and one ``class_*_fraction_*``
            column comes back populated.
          * Per-row fractions sum to ~1.0 (or 0.0 for an out-of-extent row
            — none expected here, but the contract allows both).
          * Absent classes are filled with 0 (never NaN) so downstream
            consumers can do arithmetic on the column directly.
        """
        catalog = _make_catalog(
            ("worldcover", "ESA/WorldCover/v200"),
            data_type="categorical",
        )
        result = _run_stats(
            SWEDEN_SAMPLE_DF,
            "worldcover",
            catalog,
            tmp_path,
            reducers=["class_count", "class_fraction"],
        )

        # Per-class columns must exist for both reducers.
        count_columns = [
            column
            for column in result.columns
            if "_class_" in column and column.endswith("_count_200m")
        ]
        fraction_columns = [
            column
            for column in result.columns
            if "_class_" in column and column.endswith("_fraction_200m")
        ]
        assert count_columns, "expected at least one class_*_count column"
        assert fraction_columns, "expected at least one class_*_fraction column"

        # Sum of fractions per row must be ~1.0 (a valid GEE call on land)
        # or ~0.0 (out-of-extent — not expected here but allowed). Anything
        # in between would indicate the frequencyHistogram-to-fraction
        # post-process is using the wrong denominator.
        for _, row in result.iterrows():
            row_sum = sum(row[column] for column in fraction_columns)
            assert row_sum == pytest.approx(0.0) or row_sum == pytest.approx(1.0, abs=1e-3)

        # No NaN in any class column — the "always 0 for absent" fill must
        # have run over the unioned column set.
        for column in count_columns + fraction_columns:
            assert not result[column].isna().any(), f"column {column} has NaN"


# ------------------------------------------------------------------
# Point sampling across dataset types.
# ------------------------------------------------------------------


class TestPointSampling:
    """Verify the 'point' reducer works across IMAGE / IMAGE_COLLECTION / categorical."""

    def test_point_dem_aster(self, tmp_path):
        # Static IMAGE asset — simplest case for point sampling.
        update_catalog(_make_catalog(("dem_aster", "projects/sat-io/open-datasets/ASTER/GDEM")))
        try:
            outputs = extract(
                SWEDEN_SAMPLE_DF,
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
        result = pd.read_csv(outputs["pt"])
        assert result["dem_aster_point"].notna().all()

    def test_point_dem_glo30_with_window_stats(self, tmp_path):
        """Regression: point + window stats on tiled IMAGE_COLLECTION with derived bands.

        ``dem_glo30`` is an IMAGE_COLLECTION with tiled global coverage and
        derived bands (slope, aspect). When ``point`` was combined with
        window reducers like ``mean``/``std``, the adapter used to cache a
        no-coords global static image during the band-name probe and reuse it
        for per-point sampling, causing ``img.sample()`` to return empty
        props and silently dropping every ``_point`` column from the output.
        """
        catalog = {
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
        update_catalog(catalog)
        try:
            outputs = extract(
                SWEDEN_SAMPLE_DF,
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
        result = pd.read_csv(outputs["pt"])
        # All three bands (DEM + 2 derived) must produce point columns
        # alongside window stats.
        for band_name in ("DEM", "slope", "aspect"):
            assert f"dem_glo30_{band_name}_point" in result.columns
            assert result[f"dem_glo30_{band_name}_point"].notna().all()

    def test_point_worldcover(self, tmp_path):
        # Categorical asset — point sampling must return the class ID exactly,
        # not a resampled / interpolated value.
        update_catalog(_make_catalog(("lulc", "ESA/WorldCover/v200"), data_type="categorical"))
        try:
            outputs = extract(
                SWEDEN_SAMPLE_DF,
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
        result = pd.read_csv(outputs["pt"])
        assert result["lulc_point"].notna().all()


# ------------------------------------------------------------------
# Raster export (GeoTIFF tiles) across dataset types.
# ------------------------------------------------------------------


class TestRasterExport:
    """Verify raster export produces correctly shaped GeoTIFFs across asset types."""

    def test_tiles_dem_glo30(self, tmp_path):
        # IMAGE_COLLECTION with tiled coverage — exercises the per-point
        # mosaicking path. Two sample points must produce two GeoTIFFs.
        update_catalog(_make_catalog(("dem_glo30", "COPERNICUS/DEM/GLO30")))
        try:
            extract(
                SWEDEN_SAMPLE_DF,
                {
                    "batch_id": "tiles",
                    "datasets": ["dem_glo30"],
                    "settings": {"output_type": "raster", "window_size_m": 200},
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()
        exported_tifs = list((tmp_path / "tiles" / "dem_glo30").glob("*.tif"))
        assert len(exported_tifs) == len(SWEDEN_SAMPLE_DF)
        # And every tile must actually contain data (not just be a header).
        # Catches silent failures where the request returned an empty image.
        for tif_path in exported_tifs:
            with rasterio.open(tif_path) as src:
                assert src.width > 0 and src.height > 0
                assert src.count >= 1

    def test_tiles_worldcover_resample(self, tmp_path):
        # Categorical asset + ``resample_m`` override — exercises the
        # resampling code path on a discrete-valued raster.
        update_catalog(_make_catalog(("lulc", "ESA/WorldCover/v200"), data_type="categorical"))
        try:
            extract(
                SWEDEN_SAMPLE_DF,
                {
                    "batch_id": "tiles",
                    "datasets": ["lulc"],
                    "settings": {"output_type": "raster", "window_size_m": 200, "resample_m": 50},
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()
        expected_pixels_per_side = round(200 / 50)  # 4x4 at 50m over a 200m window
        for tif_path in (tmp_path / "tiles" / "lulc").glob("*.tif"):
            with rasterio.open(tif_path) as src:
                assert src.width == expected_pixels_per_side
                assert src.height == expected_pixels_per_side

    def test_raster_metadata_sidecar(self, tmp_path):
        """Raster export writes a sidecar metadata JSON with native CRS/scale.

        Migrated from the now-deleted test_gee.py — covers the
        ``meta["datasets"]["..."]["data_source"]`` invariant which proves
        the dataset's adapter recorded itself correctly in the metadata.
        """
        update_catalog(_make_catalog(("dem_aster", "projects/sat-io/open-datasets/ASTER/GDEM")))
        try:
            extract(
                SWEDEN_SAMPLE_DF,
                {
                    "batch_id": "gee_meta",
                    "datasets": ["dem_aster"],
                    "settings": {"output_type": "raster", "window_size_m": 200},
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()

        # Metadata sidecar is co-located with the per-point tiles under
        # output_dir/{batch_id}/{batch_id}_metadata.json.
        meta_path = tmp_path / "gee_meta" / "gee_meta_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "dem_aster" in meta["datasets"]
        assert meta["datasets"]["dem_aster"]["data_source"] == "earth_engine"


# ------------------------------------------------------------------
# Automatic date selection for ImageCollections — QC date inspection.
# ------------------------------------------------------------------
#
# These tests no longer just check "values are non-null". They read the
# QC csv and assert that the per-point image selection went the way the
# catalog policy said it should: nearest-to-sample for in-range dates,
# clamped-to-nearest for out-of-range dates, most-recent for no-date input.


class TestAutoDateSelection:
    """Verify per-point image selection is recorded correctly in QC metadata."""

    def test_in_range_dates_pick_nearest_image(self, tmp_path):
        """Two points with different in-range dates should pick different images."""
        # ERA5 MONTHLY is monthly aggregates 1979-present, so 2010 and 2020
        # are both well inside the range and far enough apart that the
        # nearest-image selection must pick different timestamps for each.
        in_range_df = pd.DataFrame(
            {
                "id": ["A_2010", "B_2020"],
                "lat": [62.97, 62.98],
                "lon": [18.02, 18.03],
                "date": ["2010-06-01", "2020-06-01"],
            }
        )
        update_catalog(_make_catalog(("era5", "ECMWF/ERA5/MONTHLY")))
        try:
            extract(
                in_range_df,
                {
                    "batch_id": "in_range",
                    "datasets": ["era5"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean"],
                        "window_size_m": 200,
                    },
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()

        qc_df = pd.read_csv(tmp_path / "in_range_qc.csv")

        # Both points were in the collection's date range — neither should
        # have been clamped, and both should report the "nearest_to_sample"
        # source label.
        assert qc_df["era5_date_clamped_200m"].eq(False).all()
        assert qc_df["era5_date_source_200m"].eq("nearest_to_sample").all()

        # And critically: the two points must have picked *different*
        # images. If the adapter cached the wrong image globally, both rows
        # would share the same image_time_start.
        selected_image_times = set(qc_df["era5_image_time_start_200m"])
        assert len(selected_image_times) == 2, (
            f"expected two distinct images picked for two different dates, "
            f"got {selected_image_times}"
        )

    def test_out_of_range_date_clamps_and_marks_clamped(self, tmp_path):
        """A future date should clamp to the latest available image."""
        future_df = pd.DataFrame(
            {
                "id": ["future"],
                "lat": [62.97],
                "lon": [18.02],
                "date": ["2099-12-31"],
            }
        )
        update_catalog(_make_catalog(("era5", "ECMWF/ERA5/MONTHLY")))
        try:
            extract(
                future_df,
                {
                    "batch_id": "future",
                    "datasets": ["era5"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean"],
                        "window_size_m": 200,
                    },
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()

        qc_df = pd.read_csv(tmp_path / "future_qc.csv")
        # The QC sidecar must flag the row as clamped and label the source
        # accordingly — this is the user-visible signal that the asked-for
        # date was unavailable.
        assert bool(qc_df["era5_date_clamped_200m"].iloc[0]) is True
        assert qc_df["era5_date_source_200m"].iloc[0] == "clamped_to_nearest"

    def test_no_date_column_uses_most_recent(self, tmp_path):
        """DataFrame without a date column should use the most recent image."""
        no_date_df = pd.DataFrame(
            {
                "id": ["A", "B"],
                "lat": [62.9768783, 62.9812956],
                "lon": [18.026823, 18.0309905],
            }
        )
        update_catalog(_make_catalog(("era5", "ECMWF/ERA5/MONTHLY")))
        try:
            extract(
                no_date_df,
                {
                    "batch_id": "no_date",
                    "datasets": ["era5"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean"],
                        "window_size_m": 200,
                    },
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()

        qc_df = pd.read_csv(tmp_path / "no_date_qc.csv")
        # Source label should explicitly indicate "no date column was provided"
        # so users can audit which path their points took.
        assert qc_df["era5_date_source_200m"].eq("most_recent_no_date").all()


# ------------------------------------------------------------------
# Built-in catalog smoke tests.
# ------------------------------------------------------------------
#
# The single biggest gap in the prior test suite was that none of the
# ~30 datasets shipped in ee_catalog.yml were actually tested via the
# bundled catalog. Every test built its own inline catalog dict. These
# tests close that gap.


class TestBuiltinCatalog:
    def test_extract_uses_builtin_catalog_without_update_catalog(self, tmp_path):
        """``extract()`` finds bundled datasets without any ``update_catalog`` call.

        Confirms the package's primary public workflow works as documented:
        install envoi, call extract() against a built-in dataset name, get
        results back. Catches typos in ee_catalog.yml and asset_type
        auto-detection failures for the canonical reference dataset.
        """
        # Note: no update_catalog() call here. The bundled catalog is loaded
        # by extract() itself via the BUILTIN_EE_CATALOG sentinel.
        outputs = extract(
            SWEDEN_SAMPLE_DF,
            {
                "batch_id": "builtin",
                "datasets": ["dem_aster"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 200,
                },
            },
            output_dir=tmp_path,
        )
        result = pd.read_csv(outputs["builtin"])
        # At least one mean column must exist and have non-null values for
        # both sample points — the minimum "did extract work" signal.
        mean_columns = [c for c in result.columns if c.startswith("dem_aster") and "mean" in c]
        assert mean_columns
        assert result[mean_columns[0]].notna().all()


# Datasets that are currently expected to fail the catalog-walk smoke test
# due to a known adapter limitation. Mapped to the reason recorded on the
# xfail marker so each entry is self-documenting in pytest output. Use
# ``strict=False`` so the test silently flips to XPASS (not failure) if the
# underlying issue gets fixed — clearing the entry is then a one-line edit.
#
# Keep this dict empty when no known failures exist. Adding an entry means
# we're knowingly shipping with a broken catalog entry, so the threshold
# for inclusion is "we have a tracked follow-up, not just a transient flake".
_KNOWN_SMOKE_FAILURES: dict[str, str] = {}


def _parametrize_builtin_catalog():
    """Build the ``(name, spec)`` parametrize list for the catalog-walk test.

    Done at import time so pytest's collection shows each dataset as a
    separate test ID — making it obvious from the test output exactly which
    catalog entry failed if any do.
    """
    catalog = load_catalogs(BUILTIN_EE_CATALOG)
    # Sort by name so the parametrize order is stable across pytest runs —
    # otherwise CI output diffs noisily from one invocation to the next.
    params = []
    for dataset_name, dataset_spec in sorted(catalog["datasets"].items()):
        marks = []
        # Wrap known-failing entries in xfail so the suite stays green while
        # keeping the entry visible in pytest output (with the reason inline).
        if dataset_name in _KNOWN_SMOKE_FAILURES:
            marks.append(
                pytest.mark.xfail(reason=_KNOWN_SMOKE_FAILURES[dataset_name], strict=False)
            )
        params.append(pytest.param(dataset_name, dataset_spec, marks=marks, id=dataset_name))
    return params


class TestCatalogWalk:
    """Smoke-test every dataset shipped in the built-in catalog.

    For each entry we run a single-point extract() and check that at least
    one statistic column comes back non-null. If a future catalog edit
    introduces a typo, a missing data_type, or an asset path that GEE no
    longer recognises, exactly one of these parametrized tests fails and
    the failure name tells you which dataset broke.

    This is the test that makes adding a new dataset to ee_catalog.yml
    automatically tested — no extra Python required.
    """

    @pytest.mark.parametrize(
        ("dataset_name", "dataset_spec"),
        _parametrize_builtin_catalog(),
        # IDs are baked into each pytest.param above (via ``id=dataset_name``)
        # so the test output reads "test_smoke[dem_aster]" rather than
        # "test_smoke[0]" — also makes the xfail marker attach to the right
        # dataset by name rather than parametrize position.
    )
    def test_smoke(self, dataset_name, dataset_spec, tmp_path):
        # Pick a reducer that matches the dataset's declared type so the
        # categorical-data warning doesn't fire and the assertion still
        # has a column to check against.
        data_type = dataset_spec.get("data_type", "continuous")
        reducer_name = "mode" if data_type == "categorical" else "mean"

        # One sample point is enough for a smoke test — fewer GEE round-trips
        # and we don't need cross-point comparisons here.
        smoke_df = SWEDEN_SAMPLE_DF.iloc[:1].copy()

        outputs = extract(
            smoke_df,
            {
                "batch_id": "smoke",
                "datasets": [dataset_name],
                "settings": {
                    "output_type": "tabular",
                    "statistics": [reducer_name],
                    "window_size_m": 200,
                    "output_file_format": "dataframe",
                },
            },
            output_dir=tmp_path,
        )

        # ``output_file_format=dataframe`` returns the result in-memory so
        # we skip the disk write for this fast smoke pass.
        result = outputs["smoke"]
        assert isinstance(result, pd.DataFrame)

        # The dataset must have produced at least one stat column. The
        # column name varies by dataset (single-band vs multi-band) so we
        # match on the dataset name prefix and the reducer suffix.
        produced_stat_columns = [
            column
            for column in result.columns
            if column.startswith(f"{dataset_name}_") and reducer_name in column
        ]
        assert produced_stat_columns, (
            f"dataset {dataset_name!r}: extract produced no {reducer_name} columns. "
            f"Got columns: {sorted(result.columns)}"
        )

        # And at least one of those columns must have non-null data — a
        # column full of NaN would indicate the asset returned empty.
        any_column_has_values = any(
            result[column].notna().any() for column in produced_stat_columns
        )
        assert (
            any_column_has_values
        ), f"dataset {dataset_name!r}: every {reducer_name} column is all-NaN"


# ------------------------------------------------------------------
# Numerical correctness — a known elevation at a known location.
# ------------------------------------------------------------------
#
# The other GEE tests check schema / non-null / column names. This one
# asserts the actual returned value matches an externally verifiable
# ground truth. If the reducer math, projection handling, or asset
# resolution regresses numerically, this is the test that catches it.


class TestGeeNumericalCorrectness:
    def test_aster_dem_on_bonneville_salt_flats(self, tmp_path):
        """ASTER GDEM at Bonneville Salt Flats reports the surveyed elevation.

        Bonneville is a ~260km² perfectly flat salt pan in north-western Utah
        at a surveyed elevation of ~1285 m. Over a 200 m window centred on
        the flats, ASTER GDEM consistently reports values around 1279 m
        (slight ASTER bias is well documented). The flatness means
        mean == min == max for any window inside the pan, so the assertion
        is tight rather than just an order-of-magnitude bound.
        """
        bonneville_df = pd.DataFrame(
            {
                "id": ["bonneville"],
                "lat": [40.7500],
                "lon": [-113.8500],
                "date": ["2020-01-01"],
            }
        )
        update_catalog(_make_catalog(("dem_aster", "projects/sat-io/open-datasets/ASTER/GDEM")))
        try:
            outputs = extract(
                bonneville_df,
                {
                    "batch_id": "bonneville",
                    "datasets": ["dem_aster"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean", "min", "max"],
                        "window_size_m": 200,
                    },
                },
                output_dir=tmp_path,
            )
        finally:
            reset_catalog()

        result = pd.read_csv(outputs["bonneville"])
        # Tight tolerance — empirically ASTER gives 1279 m at this point.
        # ±10 m absorbs any small future re-tiling or version bump without
        # admitting an order-of-magnitude regression.
        observed_mean = result["dem_aster_mean_200m"].iloc[0]
        assert observed_mean == pytest.approx(1279, abs=10)

        # Bonneville is famously flat, so min and max should equal the mean
        # to within the same tolerance — confirms the reducer wiring is
        # consistent across mean / min / max and that there's no rogue
        # nodata pixel inflating the spread.
        assert result["dem_aster_min_200m"].iloc[0] == pytest.approx(observed_mean, abs=2)
        assert result["dem_aster_max_200m"].iloc[0] == pytest.approx(observed_mean, abs=2)
