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
        # Multi-band local entry used by the per-call band override tests.
        # The DEM TIFF has 3 bands; this registration exposes all of them
        # so the tests can shrink the band list at call time.
        "multi_band_local": {
            "data_source": "local",
            "path": str(DEM_TIF),
            "bands": [1, 2, 3],
        },
        # Categorical entry — same underlying file, tagged so the typed-stats
        # tests can exercise the data_type dispatch without a real categorical raster.
        "dem_local_categorical": {
            "data_source": "local",
            "path": str(DEM_TIF),
            "bands": 1,
            "data_type": "categorical",
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
