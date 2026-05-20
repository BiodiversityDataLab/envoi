"""Tests for the extract() pipeline using local raster data.

The synthetic DEM and ``sample_df`` fixtures are provided by ``conftest.py``,
so this module no longer references any on-disk fixture paths.
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

from envoi.extract import extract
from envoi import update_catalog, reset_catalog


@pytest.fixture(autouse=True)
def register_test_catalog(dem_tif):
    """Register the local test datasets before each test and clean up after.

    The catalog is built per-test (rather than at module load time) because
    the synthetic DEM path is a session-scoped tmp file — its location isn't
    known until the ``dem_tif`` fixture runs.
    """
    dem_path = str(dem_tif)
    catalog = {
        "datasets": {
            "dem_local": {
                "data_source": "local",
                "path": dem_path,
                "bands": 1,
            },
            "slope_local": {
                "data_source": "local",
                "path": dem_path,
                "bands": 2,
            },
            # Multi-band local entry used by the per-call band override tests.
            # The synthetic DEM has 3 bands; this registration exposes all of
            # them so the tests can shrink the band list at call time.
            "multi_band_local": {
                "data_source": "local",
                "path": dem_path,
                "bands": [1, 2, 3],
            },
            # Categorical entry — same underlying file, tagged so the typed-stats
            # tests can exercise the data_type dispatch without a real categorical raster.
            "dem_local_categorical": {
                "data_source": "local",
                "path": dem_path,
                "bands": 1,
                "data_type": "categorical",
            },
        }
    }
    update_catalog(catalog)
    yield
    reset_catalog()


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
        # QC file is written alongside the stats file but not included in the return dict.
        qc_df = pd.read_csv(tmp_path / "dem_100m_qc.csv")

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
        assert (tmp_path / "csv_test_qc.csv").exists()
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

    def test_multiband_out_of_extent_keeps_per_band_schema(self, sample_df, tmp_path):
        """Out-of-extent points in a multi-band dataset must produce per-band-named
        stat columns (b1_mean, b2_mean, ...) — not flat-named columns (mean, std).

        Regression: previously fetch_values returned a 1D empty array for out-of-
        extent points regardless of band count, so the multi-band branch in
        _fetch_stats_single (which keys off ``ndim == 2``) silently fell through
        to the single-band path and wrote flat keys. With three bands, that meant
        a row could end up populating ``mean``/``std`` columns that don't exist
        for any other row, polluting the output schema.
        """
        # Append one out-of-extent point (lat=0, lon=0 is well outside the
        # northern-Sweden raster used by the sample fixtures). This guarantees
        # at least one row hits the rio_mask exception path.
        out_of_extent_row = pd.DataFrame(
            [{"id": "OOB", "n_otu": 0, "lat": 0.0, "lon": 0.0, "date": "2025-01-01"}]
        )
        df_with_oob = pd.concat([sample_df, out_of_extent_row], ignore_index=True)

        outputs = extract(
            df_with_oob,
            {
                "batch_id": "mb_oob",
                "datasets": ["multi_band_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean", "std"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["mb_oob"])

        # Per-band columns must exist for every band×reducer combination.
        for band_index in (1, 2, 3):
            for reducer_name in ("mean", "std"):
                expected_column = f"multi_band_local_b{band_index}_{reducer_name}_100m"
                assert expected_column in stats_df.columns, (
                    f"missing expected column {expected_column}; " f"got {sorted(stats_df.columns)}"
                )

        # Flat-named columns must NOT exist — they are the symptom of the bug.
        forbidden_columns = {"multi_band_local_mean_100m", "multi_band_local_std_100m"}
        leaked_columns = forbidden_columns & set(stats_df.columns)
        assert (
            not leaked_columns
        ), f"flat-named columns leaked into multi-band output: {leaked_columns}"

        # The out-of-extent row should have NaN in every per-band stat column.
        oob_row = stats_df.loc[stats_df["id"] == "OOB"].iloc[0]
        for band_index in (1, 2, 3):
            for reducer_name in ("mean", "std"):
                value = oob_row[f"multi_band_local_b{band_index}_{reducer_name}_100m"]
                assert pd.isna(
                    value
                ), f"expected NaN at b{band_index}_{reducer_name} for OOB point, got {value}"


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

    def test_multiband_tile_preserves_all_bands(self, sample_df, tmp_path):
        """Multi-band local raster is exported with every band intact."""
        # `multi_band_local` is registered with bands=[1, 2, 3], so the
        # exported GeoTIFFs must have count == 3 — previously the adapter
        # silently kept only band 1.
        extract(
            sample_df,
            {
                "batch_id": "mb_tiles",
                "datasets": ["multi_band_local"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        tile_dir = tmp_path / "mb_tiles" / "multi_band_local"
        tifs = list(tile_dir.glob("*.tif"))
        assert tifs, "expected at least one tile to be exported"
        for tif in tifs:
            with rasterio.open(tif) as src:
                assert src.count == 3, f"{tif.name} has {src.count} bands, expected 3"

    def test_uint8_no_nodata_propagates_synthetic_sentinel(self, tmp_path):
        """A no-declared-nodata uint8 raster's exported tile declares the
        synthetic 255 sentinel and writes it into polygon-exterior corners.

        Regression: previously meta["nodata"] reported None for bands without
        declared nodata while window_arr contained fabricated 0s. Downstream
        consumers reading the tile back had no signal that the corner pixels
        were synthetic, silently folding them into stats.
        """
        raster_path = _build_synthetic_raster(
            tmp_path / "uint8_nonodata.tif",
            dtype="uint8",
            fill_value=10,
            declare_nodata=False,
        )
        update_catalog(
            {"datasets": {"uint8_test": {"data_source": "local", "path": str(raster_path)}}}
        )

        # Place the point at the centre of the synthetic raster. The polygon
        # is built in UTM and reprojected to the raster's EPSG:4326 CRS, so it
        # ends up as a non-axis-aligned quadrilateral — corner pixels of the
        # cropped bounding box fall outside the polygon and get masked.
        df = pd.DataFrame({"id": ["centre"], "lat": [62.98], "lon": [18.025]})
        extract(
            df,
            {
                "batch_id": "uint8_tile",
                "datasets": ["uint8_test"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        tile_path = next((tmp_path / "uint8_tile" / "uint8_test").glob("*.tif"))
        with rasterio.open(tile_path) as exported:
            assert exported.dtypes[0] == "uint8"
            # Synthetic sentinel for uint8 is np.iinfo(uint8).max == 255.
            assert (
                exported.nodata == 255
            ), f"expected exported tile to declare nodata=255, got {exported.nodata}"
            tile_data = exported.read(1)
            # Real pixels (filled with 10) and synthetic corners (255) should both appear.
            assert (tile_data == 10).any(), "expected at least one real-data pixel"
            assert (tile_data == 255).any(), "expected at least one synthetic corner pixel"

    def test_float32_no_nodata_propagates_nan_sentinel(self, tmp_path):
        """A no-declared-nodata float32 raster's exported tile declares NaN
        as nodata and writes NaN into polygon-exterior corners.
        """
        raster_path = _build_synthetic_raster(
            tmp_path / "float32_nonodata.tif",
            dtype="float32",
            fill_value=10.0,
            declare_nodata=False,
        )
        update_catalog(
            {"datasets": {"float32_test": {"data_source": "local", "path": str(raster_path)}}}
        )

        df = pd.DataFrame({"id": ["centre"], "lat": [62.98], "lon": [18.025]})
        extract(
            df,
            {
                "batch_id": "float32_tile",
                "datasets": ["float32_test"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        tile_path = next((tmp_path / "float32_tile" / "float32_test").glob("*.tif"))
        with rasterio.open(tile_path) as exported:
            assert exported.dtypes[0] == "float32"
            # NaN equality is special — must use math.isnan (or np.isnan).
            assert exported.nodata is not None and math.isnan(
                exported.nodata
            ), f"expected NaN nodata, got {exported.nodata}"
            tile_data = exported.read(1)
            assert np.isnan(tile_data).any(), "expected at least one NaN corner pixel"
            assert (tile_data == 10.0).any(), "expected at least one real-data pixel"


def _build_synthetic_raster(
    output_path: Path,
    *,
    dtype: str,
    fill_value,
    declare_nodata: bool,
) -> Path:
    """Build a small EPSG:4326 raster covering the existing test sample area.

    Used by the no-declared-nodata tile-export tests. The raster is in WGS84
    so the UTM polygon built by the adapter must be reprojected to a non-
    axis-aligned shape — that's what triggers the masked-corner code path
    inside _fill_masked_window.
    """
    # ~2km × 2km area straddling the same ground as the existing DEM fixture.
    bounds = (18.0, 62.96, 18.05, 63.0)  # (left, bottom, right, top) in degrees
    width, height = 200, 200
    transform = from_bounds(*bounds, width=width, height=height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    # Only set nodata when the test wants it declared. Omitting the key
    # results in a GeoTIFF with no declared nodata, which is exactly the
    # configuration the synthetic-sentinel logic is designed to handle.
    if declare_nodata:
        profile["nodata"] = 0

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.full((height, width), fill_value, dtype=np.dtype(dtype)), 1)
    return output_path


# ------------------------------------------------------------------
# Adapter helpers (unit tests, no rasterio fixtures)
# ------------------------------------------------------------------


class TestResolveTileDtype:
    """Unit tests for LocalRasterAdapter._resolve_tile_dtype.

    Building a real heterogeneous-dtype multi-band GeoTIFF is awkward — most
    drivers refuse mixed dtypes. So we exercise the helper directly with the
    list/scalar/None inputs it has to handle when called from export_tiles.
    """

    def test_uniform_list_returns_first_dtype(self):
        """Multi-band with all bands sharing a dtype returns that dtype."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        result = LocalRasterAdapter._resolve_tile_dtype(
            ["uint8", "uint8", "uint8"], np.dtype("float32")
        )
        assert result == np.dtype("uint8")

    def test_mixed_list_promotes_and_warns(self):
        """Multi-band with mixed dtypes promotes via np.result_type and warns."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        with pytest.warns(UserWarning, match="heterogeneous band dtypes"):
            result = LocalRasterAdapter._resolve_tile_dtype(["uint8", "float32"], np.dtype("uint8"))
        # NumPy's standard rule: uint8 + float32 → float32 (preserves both).
        assert result == np.dtype("float32")

    def test_scalar_string_returns_dtype(self):
        """Single-band passes a str — helper resolves it via np.dtype."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        result = LocalRasterAdapter._resolve_tile_dtype("int16", np.dtype("float64"))
        assert result == np.dtype("int16")

    def test_none_falls_back(self):
        """No meta dtype recorded → fall back to the in-memory window dtype."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        result = LocalRasterAdapter._resolve_tile_dtype(None, np.dtype("uint16"))
        assert result == np.dtype("uint16")


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


# ------------------------------------------------------------------
# Per-call band overrides
# ------------------------------------------------------------------


def _run_with_datasets(df, datasets, tmp_path, *, batch_id="bands_test", window_size_m=100):
    """Helper that runs extract() in tabular mode with a single mean reducer.

    Centralises the boilerplate so every band-override test can focus on the
    specific datasets shape it cares about.
    """
    return extract(
        df,
        {
            "batch_id": batch_id,
            "datasets": datasets,
            "settings": {
                "output_type": "tabular",
                "statistics": ["mean"],
                "window_size_m": window_size_m,
            },
        },
        output_dir=tmp_path,
    )


class TestPerCallBandOverrides:
    def test_shorthand_bands_override_narrows_columns(self, sample_df, tmp_path):
        """Shorthand override drops band columns that aren't in the per-call list."""
        outputs = _run_with_datasets(sample_df, [{"multi_band_local": [1]}], tmp_path)

        stats_df = pd.read_csv(outputs["bands_test"])
        # Only band 1 should appear; bands 2 and 3 are excluded by the override.
        assert "multi_band_local_b1_mean_100m" in stats_df.columns
        assert "multi_band_local_b2_mean_100m" not in stats_df.columns
        assert "multi_band_local_b3_mean_100m" not in stats_df.columns

    def test_full_form_bands_override(self, sample_df, tmp_path):
        """Full-form dict produces the same output as the shorthand."""
        outputs = _run_with_datasets(sample_df, [{"multi_band_local": {"bands": [1]}}], tmp_path)

        stats_df = pd.read_csv(outputs["bands_test"])
        assert "multi_band_local_b1_mean_100m" in stats_df.columns
        assert "multi_band_local_b2_mean_100m" not in stats_df.columns
        assert "multi_band_local_b3_mean_100m" not in stats_df.columns

    def test_plain_string_entry_unchanged(self, sample_df, tmp_path):
        """Regression: existing string-only datasets list still works."""
        outputs = _run_with_datasets(sample_df, ["dem_local"], tmp_path)

        stats_df = pd.read_csv(outputs["bands_test"])
        assert "dem_local_mean_100m" in stats_df.columns

    def test_mixed_string_and_dict_entries(self, sample_df, tmp_path):
        """A list mixing strings and dict overrides produces both datasets' columns."""
        outputs = _run_with_datasets(
            sample_df,
            ["dem_local", {"multi_band_local": [1]}],
            tmp_path,
        )

        stats_df = pd.read_csv(outputs["bands_test"])
        assert "dem_local_mean_100m" in stats_df.columns
        assert "multi_band_local_b1_mean_100m" in stats_df.columns
        assert "multi_band_local_b2_mean_100m" not in stats_df.columns

    def test_unknown_full_form_key_raises(self, sample_df, tmp_path):
        """Full-form dict with a key outside _ALLOWED_OVERRIDE_KEYS raises ValueError."""
        with pytest.raises(ValueError, match="unknown override key"):
            _run_with_datasets(sample_df, [{"dem_local": {"foo": 1}}], tmp_path)

    def test_multi_key_dict_raises(self, sample_df, tmp_path):
        """A dict entry with more than one key is rejected — each entry must be single-key."""
        with pytest.raises(ValueError, match="exactly one key"):
            _run_with_datasets(
                sample_df,
                [{"dem_local": [1], "multi_band_local": [1]}],
                tmp_path,
            )

    def test_derived_band_on_local_raises(self, sample_df, tmp_path):
        """Derived band names ('slope') on local rasters raise a clear ValueError."""
        with pytest.raises(ValueError, match="slope"):
            _run_with_datasets(sample_df, [{"dem_local": ["slope"]}], tmp_path)

    def test_empty_bands_list_raises(self, sample_df, tmp_path):
        """An empty bands list is rejected (a band override must select at least one band)."""
        with pytest.raises(ValueError, match="at least one band"):
            _run_with_datasets(sample_df, [{"dem_local": []}], tmp_path)

    def test_unknown_dataset_name_raises(self, sample_df, tmp_path):
        """Override on an unknown dataset name raises KeyError, same as for plain strings."""
        with pytest.raises(KeyError, match="not_a_dataset"):
            _run_with_datasets(sample_df, [{"not_a_dataset": [1]}], tmp_path)

    def test_metadata_records_resolved_overrides(self, sample_df, tmp_path):
        """The metadata sidecar records the resolved per-dataset bands for each entry."""
        outputs = _run_with_datasets(
            sample_df,
            ["dem_local", {"multi_band_local": [1]}],
            tmp_path,
        )

        meta_path = outputs["bands_test"].parent / "bands_test_metadata.json"
        meta = json.loads(meta_path.read_text())
        resolved = {entry["name"]: entry for entry in meta["config"]["datasets"]}

        # Catalog default for dem_local is `bands: 1` (scalar).
        assert resolved["dem_local"]["bands"] == 1
        assert resolved["dem_local"]["derived_bands"] is None
        # Override narrows multi_band_local to [1].
        assert resolved["multi_band_local"]["bands"] == [1]
        assert resolved["multi_band_local"]["derived_bands"] is None


# ------------------------------------------------------------------
# Typed statistics dict (continuous vs categorical)
# ------------------------------------------------------------------


def _run_typed_stats(df, datasets, statistics, tmp_path, *, window_size_m=100):
    """Helper for typed-statistics tests — runs extract() and returns (outputs, meta)."""
    batch_id = "typed_stats"
    outputs = extract(
        df,
        {
            "batch_id": batch_id,
            "datasets": datasets,
            "settings": {
                "output_type": "tabular",
                "statistics": statistics,
                "window_size_m": window_size_m,
            },
        },
        output_dir=tmp_path,
    )
    meta_path = outputs[batch_id].parent / f"{batch_id}_metadata.json"
    meta = json.loads(meta_path.read_text())
    return outputs, meta


class TestTypedStatistics:
    def test_flat_list_backward_compat(self, sample_df, tmp_path):
        """Flat list still works and produces the expected columns."""
        outputs, _ = _run_typed_stats(sample_df, ["dem_local"], ["mean", "std"], tmp_path)
        stats_df = pd.read_csv(outputs["typed_stats"])
        assert "dem_local_mean_100m" in stats_df.columns
        assert "dem_local_std_100m" in stats_df.columns

    def test_typed_dict_continuous_dataset(self, sample_df, tmp_path):
        """Typed dict: continuous dataset gets the continuous reducer list."""
        statistics = {"continuous": ["mean", "std"], "categorical": ["mode"]}
        outputs, _ = _run_typed_stats(sample_df, ["dem_local"], statistics, tmp_path)
        stats_df = pd.read_csv(outputs["typed_stats"])
        assert "dem_local_mean_100m" in stats_df.columns
        assert "dem_local_std_100m" in stats_df.columns
        # 'mode' is in the categorical list only — should not appear for this dataset.
        assert "dem_local_mode_100m" not in stats_df.columns

    def test_typed_dict_categorical_dataset(self, sample_df, tmp_path):
        """Typed dict: categorical dataset gets the categorical reducer list."""
        statistics = {"continuous": ["mean", "std"], "categorical": ["mode"]}
        outputs, _ = _run_typed_stats(sample_df, ["dem_local_categorical"], statistics, tmp_path)
        stats_df = pd.read_csv(outputs["typed_stats"])
        assert "dem_local_categorical_mode_100m" in stats_df.columns
        assert "dem_local_categorical_mean_100m" not in stats_df.columns

    def test_typed_dict_mixed_run(self, sample_df, tmp_path):
        """Typed dict: continuous + categorical datasets in one run each get their own reducers."""
        statistics = {"continuous": ["mean"], "categorical": ["mode"]}
        outputs, _ = _run_typed_stats(
            sample_df, ["dem_local", "dem_local_categorical"], statistics, tmp_path
        )
        stats_df = pd.read_csv(outputs["typed_stats"])
        assert "dem_local_mean_100m" in stats_df.columns
        assert "dem_local_mode_100m" not in stats_df.columns
        assert "dem_local_categorical_mode_100m" in stats_df.columns
        assert "dem_local_categorical_mean_100m" not in stats_df.columns

    def test_mode_in_both_lists_runs_for_both_types(self, sample_df, tmp_path):
        """mode in both lists produces a mode column for both continuous and categorical datasets."""
        statistics = {"continuous": ["mean", "mode"], "categorical": ["mode"]}
        outputs, _ = _run_typed_stats(
            sample_df, ["dem_local", "dem_local_categorical"], statistics, tmp_path
        )
        stats_df = pd.read_csv(outputs["typed_stats"])
        assert "dem_local_mode_100m" in stats_df.columns
        assert "dem_local_categorical_mode_100m" in stats_df.columns

    def test_dataset_without_data_type_defaults_to_continuous(self, sample_df, tmp_path):
        """A dataset with no data_type falls back to the continuous reducer list."""
        statistics = {"continuous": ["mean"], "categorical": ["mode"]}
        # dem_local has no data_type set in the test catalog — should use continuous.
        outputs, _ = _run_typed_stats(sample_df, ["dem_local"], statistics, tmp_path)
        stats_df = pd.read_csv(outputs["typed_stats"])
        assert "dem_local_mean_100m" in stats_df.columns

    def test_missing_categorical_list_raises_for_categorical_dataset(self, sample_df, tmp_path):
        """Categorical dataset in run with no 'categorical' key in stats raises ValueError."""
        with pytest.raises(ValueError, match="categorical"):
            _run_typed_stats(
                sample_df,
                ["dem_local_categorical"],
                {"continuous": ["mean"]},
                tmp_path,
            )

    def test_empty_inner_list_raises(self, sample_df, tmp_path):
        """An empty list for one type is rejected at parse time."""
        with pytest.raises(ValueError, match="non-empty list"):
            _run_typed_stats(sample_df, ["dem_local"], {"continuous": []}, tmp_path)

    def test_unknown_top_level_key_raises(self, sample_df, tmp_path):
        """A key outside {'continuous', 'categorical'} is rejected."""
        with pytest.raises(ValueError, match="unknown 'statistics' key"):
            _run_typed_stats(
                sample_df, ["dem_local"], {"continuous": ["mean"], "fooz": ["mode"]}, tmp_path
            )

    def test_unknown_reducer_raises(self, sample_df, tmp_path):
        """An unknown reducer name inside the typed dict is rejected."""
        with pytest.raises(ValueError, match="unknown reducer"):
            _run_typed_stats(sample_df, ["dem_local"], {"continuous": ["banana"]}, tmp_path)

    def test_metadata_records_user_stats_form(self, sample_df, tmp_path):
        """Metadata sidecar preserves the user's original statistics form (flat or dict)."""
        # Flat list is stored verbatim.
        _, meta_flat = _run_typed_stats(sample_df, ["dem_local"], ["mean"], tmp_path)
        assert meta_flat["config"]["statistics"] == ["mean"]

    def test_metadata_records_typed_dict_form(self, sample_df, tmp_path):
        """Metadata sidecar preserves the typed-dict form when that was supplied."""
        statistics = {"continuous": ["mean", "std"], "categorical": ["mode"]}
        _, meta = _run_typed_stats(sample_df, ["dem_local"], statistics, tmp_path)
        assert meta["config"]["statistics"] == statistics


# ------------------------------------------------------------------
# LocalRasterAdapter unit tests
# ------------------------------------------------------------------
#
# These exercise behaviours that aren't easily reachable through the
# extract() entry point: lifecycle / resource cleanup, and edge cases in
# the UTM-zone helper. They import the adapter directly.


class TestLocalRasterAdapterLifecycle:
    def test_context_manager_closes_dataset(self, dem_tif):
        """`with` block releases the rasterio dataset when it exits."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        spec = {"data_source": "local", "path": str(dem_tif), "bands": 1}
        with LocalRasterAdapter(spec) as adapter:
            # Inside the block the underlying dataset is open and usable.
            assert adapter.src.closed is False
            handle = adapter.src
        # After the block, close() has been called and the rasterio
        # DatasetReader reports itself as closed — important to avoid
        # leaking file descriptors when many datasets are processed.
        assert handle.closed is True

    def test_close_is_idempotent(self, dem_tif):
        """Calling close() twice (e.g. via with + manual close) is a no-op."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        spec = {"data_source": "local", "path": str(dem_tif), "bands": 1}
        adapter = LocalRasterAdapter(spec)
        adapter.close()
        # Second call must not raise even though the dataset is already closed.
        adapter.close()
        assert adapter.src.closed is True

    def test_get_utm_crs_clamps_at_antimeridian(self):
        """lon == 180 must produce a valid UTM zone (1-60), not zone 61."""
        from envoi.metadata import get_utm_crs

        # Northern hemisphere: zones 32601..32660. Southern: 32701..32760.
        # The naive `(lon + 180) / 6 + 1` formula gives 61 at lon == 180,
        # which would yield EPSG:32661 — outside the UTM range.
        assert get_utm_crs(180.0, 0.0) == "EPSG:32660"
        assert get_utm_crs(180.0, -1.0) == "EPSG:32760"
        # Spot-check that an ordinary longitude still resolves correctly.
        assert get_utm_crs(0.0, 0.0) == "EPSG:32631"
