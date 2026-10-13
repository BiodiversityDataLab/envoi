"""Tests for the extract() pipeline using local raster data.

The synthetic DEM and ``sample_df`` fixtures are provided by ``conftest.py``,
so this module no longer references any on-disk fixture paths.
"""

import importlib
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

from envoi import reset_catalog, update_catalog
from envoi.extract import extract


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

    def test_input_crs_keeps_original_and_adds_wgs84(self, sample_df, tmp_path):
        """With input_crs set, the output keeps the original coordinates and
        adds reprojected *_wgs84 columns.

        The sample points are WGS84; we project them to the DEM's UTM zone
        (EPSG:32634) and feed those projected coordinates as input. The output
        must round-trip the projected coordinates the user supplied in
        decimalLatitude/decimalLongitude, and expose the WGS84 reprojection in
        decimalLatitude_wgs84/decimalLongitude_wgs84.
        """
        from pyproj import Transformer

        # Project the WGS84 sample points into UTM 34N so we have realistic
        # input-CRS coordinates to feed back in via input_crs.
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
        easting, northing = to_utm.transform(
            sample_df["decimalLongitude"].values, sample_df["decimalLatitude"].values
        )
        projected_df = sample_df.copy()
        projected_df["decimalLongitude"] = easting
        projected_df["decimalLatitude"] = northing

        outputs = extract(
            projected_df,
            {
                "batch_id": "crs_test",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
            input_crs="EPSG:32634",
        )
        result = pd.read_csv(outputs["crs_test"])

        # The user's lat/lon columns hold the original (UTM) coordinates.
        np.testing.assert_allclose(result["decimalLatitude"], northing)
        np.testing.assert_allclose(result["decimalLongitude"], easting)

        # The reprojected WGS84 coordinates are surfaced in extra columns and
        # match the original WGS84 the points came from.
        assert "decimalLatitude_wgs84" in result.columns
        assert "decimalLongitude_wgs84" in result.columns
        np.testing.assert_allclose(
            result["decimalLatitude_wgs84"], sample_df["decimalLatitude"], atol=1e-6
        )
        np.testing.assert_allclose(
            result["decimalLongitude_wgs84"], sample_df["decimalLongitude"], atol=1e-6
        )

    def test_input_crs_wgs84_columns_follow_core_columns(self, sample_df, tmp_path):
        """With input_crs, *_wgs84 columns follow the core columns in both output tables."""
        from pyproj import Transformer

        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
        easting, northing = to_utm.transform(
            sample_df["decimalLongitude"].values, sample_df["decimalLatitude"].values
        )
        projected_df = sample_df.copy()
        projected_df["decimalLongitude"] = easting
        projected_df["decimalLatitude"] = northing

        outputs = extract(
            projected_df,
            {
                "batch_id": "crs_order",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
            input_crs="EPSG:32634",
        )
        stats_df = pd.read_csv(outputs["crs_order"])
        qc_df = pd.read_csv(tmp_path / "crs_order_qc.csv")

        core_and_wgs84_columns = [
            "occurrenceID",
            "decimalLatitude",
            "decimalLongitude",
            "eventDate",
            "decimalLatitude_wgs84",
            "decimalLongitude_wgs84",
        ]
        # The extra input column (n_otu) and the statistics follow the
        # WGS84 columns in the statistics table.
        assert list(stats_df.columns) == core_and_wgs84_columns + [
            "n_otu",
            "dem_local_mean_100m",
        ]
        assert list(qc_df.columns) == core_and_wgs84_columns + [
            "dem_local_in_extent_100m",
            "dem_local_n_pixels_100m",
            "dem_local_had_nodata_100m",
            "dem_local_coverage_pct_100m",
            "dem_local_region_crs_100m",
        ]
        # The internal copies of the input coordinates are in neither table.
        for output_df in (stats_df, qc_df):
            assert "lat_original" not in output_df.columns
            assert "lon_original" not in output_df.columns

    def test_no_input_crs_has_no_wgs84_columns(self, sample_df, tmp_path):
        """Without input_crs (coordinates already WGS84), no *_wgs84 columns
        are added — the output is unchanged from the default behaviour."""
        outputs = extract(
            sample_df,
            {
                "batch_id": "no_crs",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )
        result = pd.read_csv(outputs["no_crs"])
        assert "decimalLatitude_wgs84" not in result.columns
        assert "decimalLongitude_wgs84" not in result.columns

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
        assert list(result["occurrenceID"]) == list(sample_df["occurrenceID"])

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
                    "window_size_m": 0,
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
            [
                {
                    "occurrenceID": "OOB",
                    "n_otu": 0,
                    "decimalLatitude": 0.0,
                    "decimalLongitude": 0.0,
                    "eventDate": "2025-01-01",
                }
            ]
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
                assert (
                    expected_column in stats_df.columns
                ), f"missing expected column {expected_column}; got {sorted(stats_df.columns)}"

        # Flat-named columns must NOT exist — they are the symptom of the bug.
        forbidden_columns = {"multi_band_local_mean_100m", "multi_band_local_std_100m"}
        leaked_columns = forbidden_columns & set(stats_df.columns)
        assert (
            not leaked_columns
        ), f"flat-named columns leaked into multi-band output: {leaked_columns}"

        # The out-of-extent row should have NaN in every per-band stat column.
        oob_row = stats_df.loc[stats_df["occurrenceID"] == "OOB"].iloc[0]
        for band_index in (1, 2, 3):
            for reducer_name in ("mean", "std"):
                value = oob_row[f"multi_band_local_b{band_index}_{reducer_name}_100m"]
                assert pd.isna(
                    value
                ), f"expected NaN at b{band_index}_{reducer_name} for OOB point, got {value}"

    def test_user_columns_keep_their_decimals(self, sample_df, tmp_path):
        """Only envoi's statistic columns are rounded; an input column keeps all 8 decimals."""
        input_df = sample_df.copy()
        habitat_index = [0.12345678, 1.23456789, 2.34567891, 3.45678912, 4.56789123, 5.67891234]
        input_df["habitat_index"] = habitat_index

        stats_df = extract(
            input_df,
            {
                "batch_id": "decimals",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean", "point"],
                    "window_size_m": [100, 200],
                    "output_file_format": "dataframe",
                },
            },
            output_dir=tmp_path,
        )

        assert stats_df["habitat_index"].tolist() == habitat_index
        # The statistic columns of every window, and the point column, are
        # still rounded to 5 decimals (stats_output_decimals).
        for column in ("dem_local_mean_100m", "dem_local_mean_200m", "dem_local_point"):
            assert stats_df[column].tolist() == stats_df[column].round(5).tolist(), column
            assert stats_df[column].notna().all(), column


# ------------------------------------------------------------------
# Custom input column names
# ------------------------------------------------------------------


class TestCustomColumnNames:
    """The default input column names follow the GBIF / Darwin Core convention
    (``occurrenceID``, ``decimalLatitude``, ``decimalLongitude``, ``eventDate``),
    but callers can override every name via the ``*_column`` parameters. These
    tests pin the override path so a future change to the canonical-name
    rename layer can't silently break user-supplied names.
    """

    def test_legacy_short_column_names_via_overrides(self, sample_df, tmp_path):
        """Old-style short column names (``id``/``lat``/``lon``/``date``) still
        work when the user passes them through the ``*_column`` overrides —
        this is the documented migration path for callers that were on the
        pre-GBIF defaults.
        """
        # Rename the GBIF-default fixture columns back to the historical short
        # names so we can exercise the override path with realistic input.
        legacy_df = sample_df.rename(
            columns={
                "occurrenceID": "id",
                "decimalLatitude": "lat",
                "decimalLongitude": "lon",
                "eventDate": "date",
            }
        )

        # A single config dict + "dataframe" format returns the DataFrame directly.
        result = extract(
            legacy_df,
            {
                "batch_id": "legacy_names",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                    "output_file_format": "dataframe",
                },
            },
            output_dir=tmp_path,
            id_column="id",
            latitude_column="lat",
            longitude_column="lon",
            date_column="date",
        )

        # The output must round-trip the user's chosen names — not the new GBIF
        # defaults, and not the internal canonical names.
        assert "id" in result.columns
        assert "occurrenceID" not in result.columns
        assert list(result["id"]) == list(legacy_df["id"])
        # And the actual extraction still ran end-to-end.
        assert "dem_local_mean_100m" in result.columns


def _tabular_mean_config(batch_id, **extra_settings):
    """Return a tabular run config for the mean of ``dem_local`` in a 100 m window."""
    return {
        "batch_id": batch_id,
        "datasets": ["dem_local"],
        "settings": {
            "output_type": "tabular",
            "statistics": ["mean"],
            "window_size_m": 100,
            **extra_settings,
        },
    }


def _project_to_utm(sample_df):
    """Return ``(projected_df, easting, northing)`` with the sample points in EPSG:32634."""
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
    easting, northing = to_utm.transform(
        sample_df["decimalLongitude"].values, sample_df["decimalLatitude"].values
    )
    projected_df = sample_df.copy()
    projected_df["decimalLongitude"] = easting
    projected_df["decimalLatitude"] = northing
    return projected_df, easting, northing


# ------------------------------------------------------------------
# Coordinate validation
# ------------------------------------------------------------------


class TestCoordinateValidation:
    """Missing, non-numeric, and infinite coordinates stop the run before extraction."""

    @pytest.mark.parametrize(
        "bad_latitude",
        [np.nan, None, "not a number", np.inf, -np.inf],
        ids=["nan", "none", "text", "inf", "minus-inf"],
    )
    def test_invalid_latitude_raises(self, sample_df, bad_latitude, tmp_path):
        """A bad latitude raises ValueError that names the column and the row's ID."""
        df = sample_df.astype({"decimalLatitude": object})
        df.loc[1, "decimalLatitude"] = bad_latitude

        with pytest.raises(ValueError, match="missing, non-numeric, or infinite") as error_info:
            extract(df, _tabular_mean_config("coords"), output_dir=tmp_path)

        message = str(error_info.value)
        assert "'decimalLatitude'" in message
        assert "in 1 row(s)" in message
        assert "'T6WPKD'" in message
        # The value is not a CRS problem, so the message must not say so.
        assert "WGS84" not in message
        assert not any(tmp_path.iterdir())

    def test_invalid_longitude_with_input_crs_raises_before_reprojection(self, sample_df, tmp_path):
        """A text longitude raises the same ValueError when input_crs is set."""
        projected_df, _, _ = _project_to_utm(sample_df)
        projected_df = projected_df.astype({"decimalLongitude": object})
        projected_df.loc[0, "decimalLongitude"] = "349321,6"

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="'decimalLongitude' have a missing, non-numeric"):
                extract(
                    projected_df,
                    _tabular_mean_config("coords"),
                    output_dir=tmp_path,
                    input_crs="EPSG:32634",
                )

        # The reprojection notice comes after the check, so it was never emitted.
        assert not any("Reprojecting" in str(warning.message) for warning in caught_warnings)

    def test_error_lists_ten_ids_and_the_total_count(self, tmp_path):
        """With more than ten bad rows, the message lists ten IDs and the total count."""
        df = pd.DataFrame(
            {
                "site": [f"site{number}" for number in range(12)],
                "y": [62.98] * 12,
                "x": [np.nan] * 12,
            }
        )

        with pytest.raises(ValueError, match=r"in 12 row\(s\)") as error_info:
            extract(
                df,
                _tabular_mean_config("coords"),
                output_dir=tmp_path,
                id_column="site",
                latitude_column="y",
                longitude_column="x",
            )

        message = str(error_info.value)
        assert "Column(s) 'x' have" in message
        assert "'site' of these rows" in message
        assert "'site9'" in message
        assert "'site10'" not in message
        assert "... and 2 more" in message

    def test_numeric_text_coordinates_give_the_same_result(self, sample_df, tmp_path):
        """Coordinates stored as text that holds a number give the same statistics."""
        text_df = sample_df.astype({"decimalLatitude": str, "decimalLongitude": str})

        from_text = extract(
            text_df,
            _tabular_mean_config("text", output_file_format="dataframe"),
            output_dir=tmp_path / "text",
        )
        from_numbers = extract(
            sample_df,
            _tabular_mean_config("numbers", output_file_format="dataframe"),
            output_dir=tmp_path / "numbers",
        )

        pd.testing.assert_frame_equal(from_text, from_numbers)


# ------------------------------------------------------------------
# Input columns with the names of working columns
# ------------------------------------------------------------------


class TestReservedColumnNames:
    """Unmapped input columns named like envoi's working columns keep their names and values."""

    def test_extra_working_column_names_keep_their_names_and_values(self, sample_df, tmp_path):
        """Extra id, lat, lon, date, lat_original, and lon_original columns do not change the run."""
        number_of_points = len(sample_df)
        extra_columns = {
            "id": [f"dwc-{number}" for number in range(number_of_points)],
            "lat": [1.5 + number for number in range(number_of_points)],
            "lon": [-1.5 - number for number in range(number_of_points)],
            "date": [f"visit {number}" for number in range(number_of_points)],
            "lat_original": [0.25 * number for number in range(number_of_points)],
            "lon_original": [0.5 * number for number in range(number_of_points)],
        }

        outputs = extract(
            sample_df.assign(**extra_columns),
            _tabular_mean_config("extra"),
            output_dir=tmp_path / "extra",
        )
        baseline_outputs = extract(
            sample_df,
            _tabular_mean_config("baseline"),
            output_dir=tmp_path / "baseline",
        )
        stats_df = pd.read_csv(outputs["extra"])
        baseline_stats_df = pd.read_csv(baseline_outputs["baseline"])
        qc_df = pd.read_csv(tmp_path / "extra" / "extra_qc.csv")
        baseline_qc_df = pd.read_csv(tmp_path / "baseline" / "baseline_qc.csv")

        # The extra columns keep their names, positions, and values.
        assert list(stats_df.columns) == [
            "occurrenceID",
            "decimalLatitude",
            "decimalLongitude",
            "eventDate",
            "n_otu",
            *extra_columns,
            "dem_local_mean_100m",
        ]
        for column, values in extra_columns.items():
            assert stats_df[column].tolist() == values

        # The run used the mapped columns only: everything else is identical
        # to a run without the extra columns, and the QC table has none of them.
        pd.testing.assert_frame_equal(stats_df.drop(columns=list(extra_columns)), baseline_stats_df)
        pd.testing.assert_frame_equal(qc_df, baseline_qc_df)

    def test_extra_original_coordinate_columns_with_input_crs(self, sample_df, tmp_path):
        """With input_crs, extra lat_original and lon_original columns do not replace the input."""
        projected_df, easting, northing = _project_to_utm(sample_df)
        extra_columns = {
            "lat_original": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "lon_original": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }

        outputs = extract(
            projected_df.assign(**extra_columns),
            _tabular_mean_config("crs_extra"),
            output_dir=tmp_path,
            input_crs="EPSG:32634",
        )
        result = pd.read_csv(outputs["crs_extra"])

        np.testing.assert_allclose(result["decimalLatitude"], northing)
        np.testing.assert_allclose(result["decimalLongitude"], easting)
        np.testing.assert_allclose(
            result["decimalLatitude_wgs84"], sample_df["decimalLatitude"], atol=1e-6
        )
        np.testing.assert_allclose(
            result["decimalLongitude_wgs84"], sample_df["decimalLongitude"], atol=1e-6
        )
        for column, values in extra_columns.items():
            assert result[column].tolist() == values

    def test_unmapped_date_column_is_not_the_sample_date(self, sample_df, tmp_path):
        """Without eventDate, an extra date column is kept but not used as the sample date."""
        # The values are not dates: if envoi parsed them, the run would raise.
        visit_labels = [f"visit {number}" for number in range(len(sample_df))]
        df = sample_df.drop(columns=["eventDate"]).assign(date=visit_labels)

        result = extract(
            df,
            _tabular_mean_config("extra_date", output_file_format="dataframe"),
            output_dir=tmp_path,
        )

        assert result["date"].tolist() == visit_labels
        assert "eventDate" not in result.columns
        # The run behaves as a run without a date column, and the sidecar says so.
        meta = json.loads((tmp_path / "extra_date_metadata.json").read_text())
        assert "No 'eventDate' column found" in meta["warnings"]["date_parsing"]

    def test_date_column_override_uses_a_date_column(self, sample_df, tmp_path):
        """With date_column="date", the date column is the sample date."""
        df = sample_df.rename(columns={"eventDate": "date"})

        result = extract(
            df,
            _tabular_mean_config("mapped_date", output_file_format="dataframe"),
            output_dir=tmp_path,
            date_column="date",
        )

        assert result["date"].tolist() == sample_df["eventDate"].tolist()
        meta = json.loads((tmp_path / "mapped_date_metadata.json").read_text())
        assert "date_parsing" not in meta.get("warnings", {})

    def test_raster_mode_names_tiles_after_the_mapped_id_column(self, sample_df, tmp_path):
        """In raster mode, an extra id column does not replace the mapped ID column."""
        df = sample_df.assign(id=["same"] * len(sample_df))

        extract(
            df,
            {
                "batch_id": "extra_id_tiles",
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        tile_dir = tmp_path / "extra_id_tiles" / "dem_local"
        manifest = pd.read_csv(tile_dir / "tiles_manifest.csv")
        assert manifest["occurrenceID"].tolist() == sample_df["occurrenceID"].tolist()
        for occurrence_id in sample_df["occurrenceID"]:
            assert (tile_dir / f"{occurrence_id}-dem_local.tif").exists()


# ------------------------------------------------------------------
# Input columns with the names of output columns
# ------------------------------------------------------------------


@pytest.fixture
def no_adapter(monkeypatch):
    """Make the adapter lookup of extract() fail, so a test shows that no adapter was built."""
    extract_module = importlib.import_module("envoi.extract")

    def refuse_adapter(data_source):
        raise AssertionError("extract() built an adapter.")

    monkeypatch.setattr(extract_module, "get_adapter", refuse_adapter)


def _raster_config(batch_id):
    """Return a raster run config for ``dem_local`` with a 100 m window."""
    return {
        "batch_id": batch_id,
        "datasets": ["dem_local"],
        "settings": {"output_type": "raster", "window_size_m": 100},
    }


class TestInputColumnsWithOutputNames:
    """An input column with the name of an output column of the same call raises before any adapter."""

    @pytest.mark.parametrize(
        "column",
        [
            # Statistic and QC columns of the run's dataset and window.
            "dem_local_mean_100m",
            "dem_local_std_100m",
            "dem_local_coverage_pct_100m",
            "dem_local_b1_class_3_count_100m",
            # Point columns, which have no window.
            "dem_local_point",
            "dem_local_b1_point",
            "dem_local_error_point",
        ],
    )
    def test_output_column_name_raises(self, sample_df, tmp_path, no_adapter, column):
        """The column raises ValueError that names it and the output, and nothing is written."""
        df = sample_df.assign(**{column: 1.0})

        with pytest.raises(ValueError, match="have the name of a column") as error_info:
            extract(df, _tabular_mean_config("terrain"), output_dir=tmp_path)

        message = str(error_info.value)
        assert f"Input column(s) '{column}' have" in message
        assert "output(s) 'terrain'" in message
        assert "earlier envoi run" in message
        assert not any(tmp_path.iterdir())

    def test_columns_of_several_outputs_are_listed_once(self, sample_df, tmp_path, no_adapter):
        """Each colliding column is listed once, in input order, with the first output that adds it."""
        df = sample_df.assign(dem_local_point=1.0, dem_local_mean_100m=2.0, slope_local_max_50m=3.0)
        slope_config = {
            "batch_id": "slope",
            "datasets": ["slope_local"],
            "settings": {"output_type": "tabular", "statistics": ["max"], "window_size_m": 50},
        }

        with pytest.raises(ValueError) as error_info:
            extract(
                df,
                [_tabular_mean_config("terrain"), _tabular_mean_config("terrain2"), slope_config],
                output_dir=tmp_path,
            )

        message = str(error_info.value)
        assert (
            "Input column(s) 'dem_local_point', 'dem_local_mean_100m', 'slope_local_max_50m' "
            "have the name of a column that envoi adds to the output tables of output(s) "
            "'terrain', 'slope'"
        ) in message

    @pytest.mark.parametrize(
        "column",
        [
            # Another window of the run's dataset.
            "dem_local_mean_200m",
            "dem_local_mean_1000m",
            # Another dataset.
            "slope_local_mean_100m",
            # A longer catalog dataset whose name starts with "dem_local_".
            "dem_local_categorical_mode_100m",
            "dem_local_categorical_point",
            # Not an output column of the dataset.
            "dem_local_notes",
            "dem_local_100m",
            "dem_localpoint",
            "mean_100m",
        ],
    )
    def test_other_columns_are_kept(self, sample_df, tmp_path, column):
        """Columns of another window or dataset (also with a longer name) keep their values."""
        df = sample_df.assign(**{column: [float(number) for number in range(len(sample_df))]})

        result = extract(
            df, _tabular_mean_config("terrain", output_file_format="dataframe"), output_dir=tmp_path
        )

        assert result[column].tolist() == [float(number) for number in range(len(sample_df))]
        assert result["dem_local_mean_100m"].notna().all()

    def test_raster_run_does_not_check_columns(self, sample_df, tmp_path):
        """A raster run adds no columns, so an input column with a statistic name does not raise."""
        df = sample_df.assign(dem_local_mean_100m=1.0, dem_local_point=2.0)

        outputs = extract(df, _raster_config("tiles"), output_dir=tmp_path)

        assert outputs["tiles:dem_local"].is_dir()

    def test_wgs84_column_raises_when_coordinates_are_reprojected(
        self, sample_df, tmp_path, no_adapter
    ):
        """With input_crs, an input "<lat>_wgs84" or "<lon>_wgs84" column raises."""
        projected_df, _, _ = _project_to_utm(sample_df)
        df = projected_df.assign(decimalLatitude_wgs84=1.0, decimalLongitude_wgs84=2.0)

        with pytest.raises(ValueError) as error_info:
            extract(
                df, _tabular_mean_config("terrain"), output_dir=tmp_path, input_crs="EPSG:32634"
            )

        assert (
            "Input column(s) 'decimalLatitude_wgs84', 'decimalLongitude_wgs84' have the name"
        ) in str(error_info.value)
        assert not any(tmp_path.iterdir())

    def test_wgs84_column_of_renamed_coordinate_columns_raises(
        self, sample_df, tmp_path, no_adapter
    ):
        """The "_wgs84" names follow the user's own latitude and longitude column names."""
        projected_df, _, _ = _project_to_utm(sample_df)
        df = projected_df.rename(columns={"decimalLatitude": "y", "decimalLongitude": "x"})

        with pytest.raises(ValueError, match="Input column\\(s\\) 'x_wgs84' have the name"):
            extract(
                df.assign(x_wgs84=1.0, decimalLongitude_wgs84=2.0),
                _tabular_mean_config("terrain"),
                output_dir=tmp_path,
                input_crs="EPSG:32634",
                latitude_column="y",
                longitude_column="x",
            )

    @pytest.mark.parametrize("input_crs", [None, "EPSG:4326", "wgs84"])
    def test_wgs84_column_is_kept_without_reprojection(self, sample_df, tmp_path, input_crs):
        """Without a reprojection, an input "<lat>_wgs84" column is an ordinary column."""
        df = sample_df.assign(decimalLatitude_wgs84=[1.0] * len(sample_df))

        result = extract(
            df,
            _tabular_mean_config("terrain", output_file_format="dataframe"),
            output_dir=tmp_path,
            input_crs=input_crs,
        )

        assert result["decimalLatitude_wgs84"].tolist() == [1.0] * len(sample_df)

    def test_wgs84_column_in_a_raster_run_with_reprojection(self, sample_df, tmp_path):
        """A raster run with input_crs adds no "_wgs84" columns, so it does not raise."""
        projected_df, _, _ = _project_to_utm(sample_df)
        df = projected_df.assign(decimalLatitude_wgs84=1.0)

        outputs = extract(df, _raster_config("tiles"), output_dir=tmp_path, input_crs="EPSG:32634")

        assert outputs["tiles:dem_local"].is_dir()


# ------------------------------------------------------------------
# GBIF / Darwin Core eventDate parsing
# ------------------------------------------------------------------


class TestGbifEventDateParsing:
    """``eventDate`` from a GBIF download follows ISO 8601 and is allowed to
    be an interval (``start/end``) or a datetime with a time component. The
    pipeline only uses day precision downstream, so the parser must collapse
    these to a YYYY-MM-DD start without erroring.
    """

    def test_iso_interval_truncates_to_start(self):
        # Verbatim form seen in real GBIF downloads — both halves of the
        # interval include a time component. The downstream date list must
        # contain the start day, with no time.
        from envoi._input_validation import _parse_and_validate_dates

        df = pd.DataFrame(
            {
                "id": ["a"],
                "lat": [62.97],
                "lon": [18.02],
                "date": ["2026-05-12T13:00/2026-05-12T15:45"],
            }
        )
        # Two aggregated warnings: one for the interval truncation, one for
        # the time-of-day truncation. Both reference a count of 1.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, dates, _ = _parse_and_validate_dates(df)
        messages = [str(w.message) for w in caught]
        assert any("Truncated 1 ISO 8601 date interval" in m for m in messages)
        assert any("Dropped time-of-day from 1 date value" in m for m in messages)
        assert dates == ["2026-05-12"]

    def test_iso_datetime_with_time_parses_to_day(self):
        # ``2026-05-12T13:00:00Z`` and similar — the strip-on-"T" path turns
        # these into pure date strings before pandas sees them. They were
        # rejected by the original strict YYYY-MM-DD parser.
        from envoi._input_validation import _parse_and_validate_dates

        df = pd.DataFrame(
            {
                "id": ["a"],
                "lat": [62.97],
                "lon": [18.02],
                "date": ["2026-05-12T13:00:00Z"],
            }
        )
        _, dates, _ = _parse_and_validate_dates(df)
        assert dates == ["2026-05-12"]

    def test_mixed_timezones_do_not_break_parsing(self):
        # Real GBIF rows can mix tz-aware (``Z``-suffixed) and tz-naive
        # datetimes in the same column. Before stripping the time component,
        # pandas would return an object-dtype Index and the downstream
        # ``.strftime`` call blew up with AttributeError. Pin this so that
        # regression can't sneak back in.
        from envoi._input_validation import _parse_and_validate_dates

        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "lat": [62.97, 62.98],
                "lon": [18.02, 18.03],
                "date": [
                    "2026-02-11T15:01Z/2026-02-11T16:03Z",  # tz-aware interval
                    "2026-05-12T13:00/2026-05-12T15:45",  # tz-naive interval
                ],
            }
        )
        _, dates, _ = _parse_and_validate_dates(df)
        assert dates == ["2026-02-11", "2026-05-12"]

    def test_per_row_warning_is_not_emitted_for_many_intervals(self):
        # A 60-row GBIF download mustn't produce 60 per-row warnings — the
        # truncation warnings are aggregated. Two aggregated warnings (one
        # for intervals, one for time-of-day) are acceptable; per-row spam
        # is what we're guarding against.
        from envoi._input_validation import _parse_and_validate_dates

        n_rows = 60
        df = pd.DataFrame(
            {
                "id": [str(i) for i in range(n_rows)],
                "lat": [62.97] * n_rows,
                "lon": [18.02] * n_rows,
                "date": ["2026-05-12T13:00/2026-05-12T15:45"] * n_rows,
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _parse_and_validate_dates(df)
        assert len(caught) == 2, (
            f"expected 2 aggregated warnings for {n_rows} interval rows, "
            f"got {len(caught)} — per-row spam regressed"
        )

    def test_missing_date_column_does_not_raise_warning(self):
        # A missing date column is a normal case and must NOT raise a
        # UserWarning (too aggressive). It is still recorded in date_warnings
        # for the metadata sidecar, and dates comes back as None.
        from envoi._input_validation import _parse_and_validate_dates

        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "lat": [62.97, 62.98],
                "lon": [18.02, 18.03],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            _, dates, date_warnings = _parse_and_validate_dates(df, date_column_name="eventDate")
        assert dates is None
        assert len(date_warnings) == 1
        assert "No 'eventDate' column" in date_warnings[0]

    def test_plain_yyyy_mm_dd_unchanged(self):
        # Regression guard: the pre-existing happy path must still work
        # exactly the same and emit no warnings.
        from envoi._input_validation import _parse_and_validate_dates

        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "lat": [62.97, 62.98],
                "lon": [18.02, 18.03],
                "date": ["2020-06-01", "2021-07-15"],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            _, dates, date_warnings = _parse_and_validate_dates(df)
        assert dates == ["2020-06-01", "2021-07-15"]
        assert date_warnings == []

    def test_year_only_still_warns_on_preprocessed_string(self):
        # Year-only dates should still warn about incomplete precision, and
        # the warning's quoted date must be the preprocessed string (not the
        # raw value with any interval suffix).
        from envoi._input_validation import _parse_and_validate_dates

        df = pd.DataFrame(
            {
                "id": ["a"],
                "lat": [62.97],
                "lon": [18.02],
                "date": ["2026"],
            }
        )
        with pytest.warns(UserWarning, match="interpreted as 2026-01-01"):
            _, dates, _ = _parse_and_validate_dates(df)
        assert dates == ["2026-01-01"]


def _date_frame(date_values):
    """Return a working dataframe (canonical column names) with the given date values."""
    n_rows = len(date_values)
    return pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(n_rows)],
            "lat": [62.97] * n_rows,
            "lon": [18.02] * n_rows,
            "date": date_values,
        }
    )


class TestSlashDateParsing:
    """Date values with "/": an interval keeps its start, YYYY/MM/DD is a date,
    and any other form stops the run, because the order of day and month is unknown.
    """

    def test_yyyy_mm_dd_with_slashes_is_a_date(self):
        """``2021/06/15`` parses to 2021-06-15 with no warning."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            _, dates, date_warnings = _parse_and_validate_dates(_date_frame(["2021/06/15"]))
        assert dates == ["2021-06-15"]
        assert date_warnings == []

    def test_interval_keeps_its_start_with_the_interval_warning(self):
        """``2021-01-01/2021-12-31`` parses to its start and gives only the interval warning."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, dates, date_warnings = _parse_and_validate_dates(
                _date_frame(["2021-01-01/2021-12-31"])
            )
        assert dates == ["2021-01-01"]
        assert [str(w.message) for w in caught] == date_warnings
        assert len(date_warnings) == 1
        assert "Truncated 1 ISO 8601 date interval" in date_warnings[0]

    def test_interval_warning_counts_only_intervals(self):
        """YYYY/MM/DD dates are not counted as intervals."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, dates, date_warnings = _parse_and_validate_dates(
                _date_frame(["2021/06/15", "2021/07/01", "2021-01-01/2021-12-31", "2021-08-01"])
            )
        assert dates == ["2021-06-15", "2021-07-01", "2021-01-01", "2021-08-01"]
        assert date_warnings == [
            (
                "Truncated 1 ISO 8601 date interval(s) ('start/end') to their start day; "
                "nearest-image lookup only uses day precision."
            )
        ]

    @pytest.mark.parametrize(
        "bad_value",
        # An interval start needs a two-digit month and day (2021-6-15/... is not one).
        ["15/06/2021", "06/15/2021", "2021/6/5", "06/2021", "2021-6-15/2021-6-20"],
    )
    def test_other_slash_forms_raise_with_the_date_column_name(self, bad_value):
        """A value with "/" in another form raises ValueError that names the user's date column."""
        from envoi._input_validation import _parse_and_validate_dates

        with pytest.raises(ValueError, match="'collectionDate' has 1 date value") as error_info:
            _parse_and_validate_dates(_date_frame([bad_value]), date_column_name="collectionDate")
        assert repr(bad_value) in str(error_info.value)

    @pytest.mark.parametrize(
        "interval",
        [
            "2021-06-15T08:00/2021-06-15T10:00",
            "2021-06-15T08:00:00Z/2021-06-16",
            "2021-06-15 08:00/2021-06-15 10:00",
        ],
    )
    def test_interval_with_time_of_day_start_is_an_interval(self, interval):
        """An interval whose start has a time of day keeps its start day."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, dates, date_warnings = _parse_and_validate_dates(_date_frame([interval]))
        assert dates == ["2021-06-15"]
        assert "Truncated 1 ISO 8601 date interval" in date_warnings[0]

    def test_year_month_interval_start_is_an_interval(self):
        """``2021/06`` is an interval with the year-only start 2021, read as 2021-01-01."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, dates, date_warnings = _parse_and_validate_dates(_date_frame(["2021/06"]))
        assert dates == ["2021-01-01"]
        assert "Truncated 1 ISO 8601 date interval" in date_warnings[0]
        assert "Date '2021' interpreted as 2021-01-01" in date_warnings[1]

    def test_text_that_is_not_a_date_raises(self):
        """Text that is not in an accepted date form raises the error that lists the value."""
        from envoi._input_validation import _parse_and_validate_dates

        with pytest.raises(ValueError, match="'eventDate' has 1 date value") as error_info:
            _parse_and_validate_dates(_date_frame(["spring 2021"]))
        assert "cannot read: 'spring 2021'." in str(error_info.value)

    def test_one_bad_value_in_a_mixed_column_raises(self):
        """One unsupported value stops the run, and the message lists only that value."""
        from envoi._input_validation import _parse_and_validate_dates

        date_values = ["2021-06-15", "2021/06/15", "2021-01-01/2021-12-31", "15/06/2021"]
        with warnings.catch_warnings():
            # The error must come before any interval warning.
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="cannot read") as error_info:
                _parse_and_validate_dates(_date_frame(date_values))
        message = str(error_info.value)
        assert "'eventDate' has 1 date value(s)" in message
        assert "'15/06/2021'" in message
        assert "'2021/06/15'" not in message
        assert "'2021-01-01/2021-12-31'" not in message

    def test_message_lists_at_most_ten_values_and_the_total(self):
        """With many unsupported values, the message lists ten different values and the count."""
        from envoi._input_validation import _parse_and_validate_dates

        # Twelve different values, and one of them twice: 13 rows.
        date_values = [f"{day:02d}/06/2021" for day in range(1, 13)] + ["01/06/2021"]
        with pytest.raises(ValueError, match="has 13 date value") as error_info:
            _parse_and_validate_dates(_date_frame(date_values))
        message = str(error_info.value)
        assert "'10/06/2021'" in message
        assert "'11/06/2021'" not in message
        assert "... and 2 more" in message

    def test_missing_date_is_not_a_slash_value(self):
        """A missing date is skipped with its warning and does not count as a value with "/"."""
        from envoi._input_validation import _parse_and_validate_dates

        with pytest.warns(UserWarning, match="Skipping 1 row"):
            df, dates, _ = _parse_and_validate_dates(_date_frame([None, "2021/06/15"]))
        assert dates == ["2021-06-15"]
        assert df["id"].tolist() == ["p1"]

    def test_extract_stops_before_any_output(self, sample_df, tmp_path):
        """extract() raises for an unsupported value in a renamed date column and writes nothing."""
        df = sample_df.rename(columns={"eventDate": "sampleDate"})
        df.loc[2, "sampleDate"] = "02/04/1960"

        with pytest.raises(ValueError, match="'sampleDate' has 1 date value"):
            extract(
                df,
                _tabular_mean_config("slash_dates"),
                date_column="sampleDate",
                output_dir=tmp_path,
            )
        assert not any(tmp_path.iterdir())


class TestAcceptedDateForms:
    """Only the listed date forms are read; every other value raises, so envoi never guesses."""

    @pytest.mark.parametrize(
        ("date_value", "expected_day"),
        [
            ("2021-6-5", "2021-06-05"),
            ("2021-06-5", "2021-06-05"),
            ("2021-6-15", "2021-06-15"),
            ("2021-06-15", "2021-06-15"),
            ("20210615", "2021-06-15"),
            ("2021/06/15", "2021-06-15"),
            (" 2021-06-15 ", "2021-06-15"),
            ("2021-06-15/2021-06-20", "2021-06-15"),
            ("2021-06-15T08:00Z/2021-06-15T10:00Z", "2021-06-15"),
        ],
    )
    def test_date_text_parses_to_its_day(self, date_value, expected_day):
        """Each accepted date form without a time of day gives its day."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, dates, _ = _parse_and_validate_dates(_date_frame([date_value]))
        assert dates == [expected_day]

    @pytest.mark.parametrize(
        "date_value",
        [
            "2021-06-15T08",
            "2021-06-15T08:00",
            "2021-06-15T08:00Z",
            "2021-06-15T08:00:00.123+02:00",
            "2021-06-15T23:30-0500",
            "2021-06-15T23:30-05",
            "20210615T080000Z",
            "2021-06-15 08:00",
            "2021-06-15 08:00:00+02:00",
        ],
    )
    def test_time_of_day_and_time_zone_are_dropped(self, date_value):
        """A time of day after "T" or a space, with or without a time zone, gives the day as written."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, dates, date_warnings = _parse_and_validate_dates(_date_frame([date_value]))
        assert dates == ["2021-06-15"]
        assert date_warnings == [
            "Dropped time-of-day from 1 date value(s); nearest-image lookup only uses day precision."
        ]

    @pytest.mark.parametrize(
        ("date_value", "expected_day", "quoted_date"),
        [
            ("2021", "2021-01-01", "2021"),
            ("2021-6", "2021-06-01", "2021-6"),
            ("2021-06", "2021-06-01", "2021-06"),
            ("2021-06T08:00", "2021-06-01", "2021-06"),
        ],
    )
    def test_year_and_year_month_give_the_first_day_with_a_warning(
        self, date_value, expected_day, quoted_date
    ):
        """A year or a year-month is read as its first day, and the warning quotes the date."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, dates, date_warnings = _parse_and_validate_dates(_date_frame([date_value]))
        assert dates == [expected_day]
        assert (
            f"Date '{quoted_date}' interpreted as {expected_day}. Provide a full YYYY-MM-DD "
            f"date if you want a specific day."
        ) in date_warnings

    def test_full_dates_in_new_forms_give_no_warning(self):
        """YYYYMMDD and one-digit months and days are full dates: no warning."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            _, dates, date_warnings = _parse_and_validate_dates(
                _date_frame(["20210615", "2021-6-5"])
            )
        assert dates == ["2021-06-15", "2021-06-05"]
        assert date_warnings == []

    def test_datetime_column(self):
        """A datetime64 column gives the day of each value, without a warning."""
        from envoi._input_validation import _parse_and_validate_dates

        date_values = pd.to_datetime(["2021-06-15 08:00", "2022-01-02 23:59"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, dates, date_warnings = _parse_and_validate_dates(_date_frame(date_values))
        assert dates == ["2021-06-15", "2022-01-02"]
        assert date_warnings == []

    def test_time_zone_aware_datetime_column_keeps_its_own_day(self):
        """A datetime column with a time zone gives the day in that time zone, not in UTC."""
        from envoi._input_validation import _parse_and_validate_dates

        # 23:30 in New York on June 15 is 03:30 UTC on June 16.
        date_values = pd.to_datetime(["2021-06-15 23:30"]).tz_localize("America/New_York")
        _, dates, _ = _parse_and_validate_dates(_date_frame(date_values))
        assert dates == ["2021-06-15"]

    def test_object_column_of_date_and_datetime_values(self):
        """An object column of date, datetime, NumPy datetime64, and text values gives each day."""
        import datetime

        from envoi._input_validation import _parse_and_validate_dates

        date_values = pd.Series(
            [
                datetime.date(2021, 6, 15),
                datetime.datetime(2021, 6, 16, 8, 30, tzinfo=datetime.timezone.utc),
                np.datetime64("2021-06-17T12:00"),
                "2021-06-18",
            ],
            dtype=object,
        )
        _, dates, _ = _parse_and_validate_dates(_date_frame(date_values))
        assert dates == ["2021-06-15", "2021-06-16", "2021-06-17", "2021-06-18"]

    @pytest.mark.parametrize(
        "year_values",
        [
            [2021, 2019],
            [2021.0, 2019.0],
            pd.array([2021, 2019], dtype="Int64"),
            [np.int64(2021), np.int64(2019)],
        ],
    )
    def test_whole_number_years(self, year_values):
        """Integer and float year columns give January 1 of each year, with a warning."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, dates, date_warnings = _parse_and_validate_dates(_date_frame(year_values))
        assert dates == ["2021-01-01", "2019-01-01"]
        assert "Date '2021' interpreted as 2021-01-01" in date_warnings[0]

    @pytest.mark.parametrize(
        "date_values",
        [
            [20210615, 20200101],
            [np.int64(20210615), np.int64(20200101)],
            [20210615.0, 20200101.0],
        ],
    )
    def test_eight_digit_numbers_are_yyyymmdd(self, date_values):
        """Whole numbers with eight digits, as a CSV reader stores YYYYMMDD dates, are dates."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # full dates: no warning
            _, dates, date_warnings = _parse_and_validate_dates(_date_frame(date_values))
        assert dates == ["2021-06-15", "2020-01-01"]
        assert date_warnings == []

    def test_float_yyyymmdd_column_with_a_missing_date(self):
        """A float YYYYMMDD column with a missing value skips that row and reads the other dates."""
        from envoi._input_validation import _parse_and_validate_dates

        with pytest.warns(UserWarning, match="Skipping 1 row"):
            df, dates, _ = _parse_and_validate_dates(_date_frame([20210615.0, np.nan]))
        assert dates == ["2021-06-15"]
        assert df["id"].tolist() == ["p0"]

    @pytest.mark.parametrize("bad_value", [15062021, 15062021.0, 20211301, 20210230])
    def test_eight_digit_number_that_is_not_a_date_raises(self, bad_value):
        """An eight-digit number with an impossible month or day (15062021: month 20) raises."""
        from envoi._input_validation import _parse_and_validate_dates

        with pytest.raises(ValueError, match="'eventDate' has 1 date value") as error_info:
            _parse_and_validate_dates(_date_frame([bad_value]))
        assert f"are not valid dates: '{bad_value}'." in str(error_info.value)

    def test_float_year_column_with_a_missing_year(self):
        """A float year column with a missing value skips that row and reads the other years."""
        from envoi._input_validation import _parse_and_validate_dates

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df, dates, _ = _parse_and_validate_dates(_date_frame([2021.0, np.nan, 2019.0]))
        assert dates == ["2021-01-01", "2019-01-01"]
        assert df["id"].tolist() == ["p0", "p2"]

    @pytest.mark.parametrize(
        ("bad_value", "listed_value"),
        [
            ("05-06-2021", "'05-06-2021'"),
            ("05.06.2021", "'05.06.2021'"),
            ("June 5, 2021", "'June 5, 2021'"),
            ("2021.06.05", "'2021.06.05'"),
            ("202106", "'202106'"),
            ("21-06-15", "'21-06-15'"),
            ("2021-06 15", "'2021-06 15'"),
            ("2021-06-15T", "'2021-06-15T'"),
            ("2021-06-15T8:00", "'2021-06-15T8:00'"),
            ("2021/06/15 08:00", "'2021/06/15 08:00'"),
            (2021.5, "'2021.5'"),
            (202106, "'202106'"),
            (20210615.5, "'20210615.5'"),
            (21, "'21'"),
            (True, "'True'"),
        ],
    )
    def test_other_values_raise_with_the_date_column_name(self, bad_value, listed_value):
        """Any other text or number raises ValueError that names the date column and the value."""
        from envoi._input_validation import _parse_and_validate_dates

        with pytest.raises(ValueError, match="'collectionDate' has 1 date value") as error_info:
            _parse_and_validate_dates(_date_frame([bad_value]), date_column_name="collectionDate")
        message = str(error_info.value)
        assert f"cannot read: {listed_value}." in message
        assert "Write these dates as YYYY-MM-DD." in message
        # Numbers get a hint, because a file reader can store dates as numbers.
        has_number_hint = "A number is read only when it is a whole number" in message
        is_number = isinstance(bad_value, (int, float)) and not isinstance(bad_value, bool)
        assert has_number_hint == is_number

    @pytest.mark.parametrize("bad_value", ["2021-02-30", "2021-13-01", "20210230", "2021-0-10"])
    def test_dates_that_do_not_exist_raise(self, bad_value):
        """A value in an accepted form that is not a valid date raises and lists the value."""
        from envoi._input_validation import _parse_and_validate_dates

        with pytest.raises(ValueError, match="'collectionDate' has 1 date value") as error_info:
            _parse_and_validate_dates(
                _date_frame(["2021-06-15", bad_value]), date_column_name="collectionDate"
            )
        assert f"are not valid dates: '{bad_value}'." in str(error_info.value)

    @pytest.mark.parametrize(
        "bad_values",
        [
            # Twelve different values that are not in an accepted form, and one twice.
            [f"{day:02d}.06.2021" for day in range(1, 13)] + ["01.06.2021"],
            # Twelve different dates that do not exist, and one twice.
            [f"2021-{month:02d}-32" for month in range(1, 13)] + ["2021-01-32"],
        ],
    )
    def test_message_lists_at_most_ten_values_and_the_total(self, bad_values):
        """Both errors list ten different values, the number of the others, and the total."""
        from envoi._input_validation import _parse_and_validate_dates

        with pytest.raises(ValueError, match="'eventDate' has 13 date value") as error_info:
            _parse_and_validate_dates(_date_frame(bad_values))
        message = str(error_info.value)
        assert repr(bad_values[9]) in message
        assert repr(bad_values[10]) not in message
        assert "... and 2 more" in message

    def test_extract_stops_before_any_output(self, sample_df, tmp_path):
        """extract() raises for a date with an unknown order of day and month, and writes nothing."""
        df = sample_df.copy()
        df.loc[2, "eventDate"] = "02.04.1960"

        with pytest.raises(ValueError, match="'eventDate' has 1 date value"):
            extract(df, _tabular_mean_config("dot_dates"), output_dir=tmp_path)
        assert not any(tmp_path.iterdir())


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

    def test_odd_resampled_tile_has_the_point_in_its_centre_pixel(self, sample_df, tmp_path):
        """A 200 m tile at resample_m=30 is 7 × 7, with the point in its centre pixel (3, 3)."""
        from pyproj import Transformer

        extract(
            sample_df,
            {
                "batch_id": "odd_tiles",
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": 200, "resample_m": 30},
            },
            output_dir=tmp_path,
        )

        # The points are in UTM zone 34N, the CRS of every tile.
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
        tile_dir = tmp_path / "odd_tiles" / "dem_local"
        for sample_id, lat, lon in zip(
            sample_df["occurrenceID"], sample_df["decimalLatitude"], sample_df["decimalLongitude"]
        ):
            point_x, point_y = to_utm.transform(lon, lat)
            with rasterio.open(tile_dir / f"{sample_id}-dem_local.tif") as src:
                assert (src.width, src.height) == (7, 7)
                assert src.crs.to_epsg() == 32634
                tile_transform = src.transform

            # 30 m pixels, corners on multiples of 30 m.
            assert (tile_transform.a, tile_transform.e) == (30.0, -30.0)
            assert tile_transform.c / 30 == pytest.approx(round(tile_transform.c / 30))
            assert tile_transform.f / 30 == pytest.approx(round(tile_transform.f / 30))
            # Column and row of the point, counted from the tile's top-left corner.
            column = math.floor((point_x - tile_transform.c) / 30.0)
            row = math.floor((tile_transform.f - point_y) / 30.0)
            assert (column, row) == (3, 3)

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
        df = pd.DataFrame(
            {"occurrenceID": ["centre"], "decimalLatitude": [62.98], "decimalLongitude": [18.025]}
        )
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

        df = pd.DataFrame(
            {"occurrenceID": ["centre"], "decimalLatitude": [62.98], "decimalLongitude": [18.025]}
        )
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
        """Raises ValueError for a dataset not in catalog."""
        with pytest.raises(ValueError, match="nonexistent"):
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

    def test_zero_window_is_rejected_for_window_statistics(self, sample_df, tmp_path):
        with pytest.raises(ValueError, match="zero is allowed only for point-only"):
            extract(
                sample_df,
                {
                    "batch_id": "fail",
                    "datasets": ["dem_local"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean"],
                        "window_size_m": 0,
                    },
                },
                output_dir=tmp_path,
            )


class TestRunConfigValidation:
    """Every run config is checked before the first adapter is built."""

    def test_duplicate_batch_id_raises_before_any_output(self, sample_df, tmp_path):
        """Two run configs with the same batch_id raise, and nothing is written."""
        with pytest.raises(ValueError, match=r"Duplicate batch_id value\(s\) \['terrain'\]"):
            extract(
                sample_df,
                [_tabular_mean_config("terrain"), _tabular_mean_config("terrain")],
                output_dir=tmp_path,
            )
        assert not any(tmp_path.iterdir())

    def test_default_batch_id_that_matches_an_explicit_one_raises(self, sample_df, tmp_path):
        """The default name of the second run config, 'output2', collides with an explicit one."""
        config_without_batch_id = _tabular_mean_config("unused")
        del config_without_batch_id["batch_id"]

        with pytest.raises(ValueError, match=r"Duplicate batch_id value\(s\) \['output2'\]"):
            extract(
                sample_df,
                [_tabular_mean_config("output2"), config_without_batch_id],
                output_dir=tmp_path,
            )

    def test_batch_ids_that_differ_only_in_case_raise(self, sample_df, tmp_path):
        """'Terrain' and 'terrain' are one name on Windows and macOS, so they raise."""
        with pytest.raises(
            ValueError, match=r"Duplicate batch_id value\(s\) \['Terrain', 'terrain'\]"
        ):
            extract(
                sample_df,
                [_tabular_mean_config("Terrain"), _tabular_mean_config("terrain")],
                output_dir=tmp_path,
            )
        assert not any(tmp_path.iterdir())

    def test_batch_ids_that_differ_only_in_case_raise_across_modes(self, sample_df, tmp_path):
        """A raster batch_id and a tabular batch_id that differ only in case also raise."""
        raster_config = {
            "batch_id": "TILES",
            "datasets": ["dem_local"],
            "settings": {"output_type": "raster", "window_size_m": 200},
        }

        with pytest.raises(ValueError, match=r"\['TILES', 'tiles'\]"):
            extract(
                sample_df,
                [_tabular_mean_config("tiles"), raster_config],
                output_dir=tmp_path,
            )
        assert not any(tmp_path.iterdir())

    @pytest.mark.parametrize(
        ("first_batch_id", "second_batch_id"),
        [("terrain_qc", "terrain"), ("terrain", "terrain_qc"), ("Terrain", "terrain_QC")],
        ids=["qc-first", "qc-second", "letter-case"],
    )
    def test_batch_id_with_the_name_of_a_qc_table_raises(
        self, sample_df, tmp_path, first_batch_id, second_batch_id
    ):
        """Tabular batch_id 'X_qc' would write the QC table file of tabular batch_id 'X'."""
        with pytest.raises(ValueError, match="would write the same file") as error_info:
            extract(
                sample_df,
                [_tabular_mean_config(first_batch_id), _tabular_mean_config(second_batch_id)],
                output_dir=tmp_path,
            )
        assert f"'{first_batch_id}'" in str(error_info.value)
        assert f"'{second_batch_id}'" in str(error_info.value)
        assert not any(tmp_path.iterdir())

    def test_raster_batch_id_with_the_name_of_a_qc_table_is_allowed(self, sample_df, tmp_path):
        """A raster output writes into its own folder, so 'X_qc' does not collide with 'X'."""
        raster_config = {
            "batch_id": "terrain_qc",
            "datasets": ["dem_local"],
            "settings": {"output_type": "raster", "window_size_m": 200},
        }

        extract(sample_df, [_tabular_mean_config("terrain"), raster_config], output_dir=tmp_path)

        assert (tmp_path / "terrain_qc.csv").exists()
        assert (tmp_path / "terrain_qc" / "dem_local" / "T2T1DJ-dem_local.tif").exists()

    @pytest.mark.parametrize("output_type", ["tabular", "raster"])
    @pytest.mark.parametrize(
        "datasets",
        [["dem_local", "dem_local"], ["dem_local", {"dem_local": [1]}]],
        ids=["plain", "with-band-list"],
    )
    def test_dataset_listed_twice_raises(self, sample_df, tmp_path, output_type, datasets):
        """One run config that lists a dataset twice raises in both modes."""
        settings = {"output_type": output_type, "window_size_m": 200}
        if output_type == "tabular":
            settings["statistics"] = ["mean"]

        with pytest.raises(
            ValueError,
            match=r"Output 'twice': dataset\(s\) \['dem_local'\] are listed more than once",
        ):
            extract(
                sample_df,
                {"batch_id": "twice", "datasets": datasets, "settings": settings},
                output_dir=tmp_path,
            )
        assert not any(tmp_path.iterdir())

    def test_raster_datasets_that_differ_only_in_case_raise(self, sample_df, dem_tif, tmp_path):
        """Two raster tile folders whose names differ only in case would be one folder."""
        update_catalog(
            {"datasets": {"DEM_local": {"data_source": "local", "path": str(dem_tif), "bands": 1}}}
        )

        with pytest.raises(
            ValueError,
            match=r"Output 'case': datasets \['dem_local', 'DEM_local'\] differ only in letter",
        ):
            extract(
                sample_df,
                {
                    "batch_id": "case",
                    "datasets": ["dem_local", "DEM_local"],
                    "settings": {"output_type": "raster", "window_size_m": 200},
                },
                output_dir=tmp_path,
            )
        assert not any(tmp_path.iterdir())

    def test_tabular_datasets_that_differ_only_in_case_are_allowed(
        self, sample_df, dem_tif, tmp_path
    ):
        """Column names keep their letter case, so tabular mode allows such dataset names."""
        update_catalog(
            {"datasets": {"DEM_local": {"data_source": "local", "path": str(dem_tif), "bands": 1}}}
        )

        outputs = extract(
            sample_df,
            {
                "batch_id": "case",
                "datasets": ["dem_local", "DEM_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        result = pd.read_csv(outputs["case"])
        assert list(result["dem_local_mean_100m"]) == list(result["DEM_local_mean_100m"])

    def test_invalid_second_run_config_raises_before_any_output(self, sample_df, tmp_path):
        """An invalid setting in the second run config stops the call before the first runs."""
        with pytest.raises(ValueError, match="Output 'second': invalid min_coverage_pct"):
            extract(
                sample_df,
                [
                    _tabular_mean_config("first"),
                    _tabular_mean_config("second", min_coverage_pct="50"),
                ],
                output_dir=tmp_path,
            )
        assert not any(tmp_path.iterdir())

    @pytest.mark.parametrize(
        "resample_m",
        ["30", True, float("nan"), float("inf"), 0, -5],
        ids=["text", "bool", "nan", "inf", "zero", "negative"],
    )
    def test_invalid_resample_m_raises(self, sample_df, resample_m, tmp_path):
        """resample_m must be a positive, finite number that is not a bool."""
        with pytest.raises(ValueError, match="Output 'tiles': invalid resample_m"):
            extract(
                sample_df,
                {
                    "batch_id": "tiles",
                    "datasets": ["dem_local"],
                    "settings": {
                        "output_type": "raster",
                        "window_size_m": 200,
                        "resample_m": resample_m,
                    },
                },
                output_dir=tmp_path,
            )

    def test_numpy_resample_m_is_accepted(self, sample_df, tmp_path):
        """A NumPy float resample_m runs and the sidecar records it as a plain number."""
        extract(
            sample_df,
            {
                "batch_id": "tiles",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "raster",
                    "window_size_m": 200,
                    "resample_m": np.float32(30),
                },
            },
            output_dir=tmp_path,
        )

        meta = json.loads((tmp_path / "tiles" / "tiles_metadata.json").read_text())
        assert meta["config"]["resample_m"] == 30.0
        with rasterio.open(tmp_path / "tiles" / "dem_local" / "T2T1DJ-dem_local.tif") as src:
            assert src.res == (30.0, 30.0)

    @pytest.mark.parametrize(
        "min_coverage_pct",
        ["50", None, True, float("nan"), -1, 101],
        ids=["text", "none", "bool", "nan", "below-range", "above-range"],
    )
    def test_invalid_min_coverage_pct_raises(self, sample_df, min_coverage_pct, tmp_path):
        """min_coverage_pct must be a finite number from 0 to 100 that is not a bool."""
        with pytest.raises(ValueError, match="Output 'coverage': invalid min_coverage_pct"):
            extract(
                sample_df,
                _tabular_mean_config("coverage", min_coverage_pct=min_coverage_pct),
                output_dir=tmp_path,
            )

    def test_numpy_min_coverage_pct_is_accepted(self, sample_df, tmp_path):
        """A NumPy integer min_coverage_pct runs and the sidecar records it as a plain number."""
        extract(
            sample_df,
            _tabular_mean_config("coverage", min_coverage_pct=np.int64(50)),
            output_dir=tmp_path,
        )

        meta = json.loads((tmp_path / "coverage_metadata.json").read_text())
        assert meta["config"]["min_coverage_pct"] == 50


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
        """Override on an unknown dataset name raises ValueError, same as for plain strings."""
        with pytest.raises(ValueError, match="not_a_dataset"):
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
# Output file formats — parquet and in-memory DataFrame
# ------------------------------------------------------------------
#
# The "csv" format is exercised by every other test in this file. These
# tests cover the other two values of `output_file_format`:
#   * "parquet" — same on-disk shape as csv, just a different writer.
#   * "dataframe" — stats are returned in memory instead of written to disk.


class TestOutputFormats:
    def test_parquet_output_writes_parquet_file(self, sample_df, tmp_path):
        """output_file_format='parquet' writes a .parquet file readable by pandas."""
        outputs = extract(
            sample_df,
            {
                "batch_id": "parquet_test",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                    "output_file_format": "parquet",
                },
            },
            output_dir=tmp_path,
        )

        # Returned path must be a .parquet file (not .csv).
        stats_path = outputs["parquet_test"]
        assert stats_path.suffix == ".parquet", f"expected .parquet, got {stats_path.suffix}"
        assert stats_path.exists()

        # Read the file back so we know the writer produced a valid parquet
        # file — not just a renamed CSV.
        stats_df = pd.read_parquet(stats_path)
        assert len(stats_df) == len(sample_df)
        assert "dem_local_mean_100m" in stats_df.columns

        # The QC sidecar is written in the same format alongside the stats.
        qc_path = tmp_path / "parquet_test_qc.parquet"
        assert qc_path.exists()
        qc_df = pd.read_parquet(qc_path)
        assert len(qc_df) == len(sample_df)

    def test_dataframe_format_returns_in_memory_dataframe(self, sample_df, tmp_path):
        """output_file_format='dataframe' with a single config returns the stats
        DataFrame directly (not a {batch_id: df} dict)."""
        result = extract(
            sample_df,
            {
                "batch_id": "df_test",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 100,
                    "output_file_format": "dataframe",
                },
            },
            output_dir=tmp_path,
        )

        # The return value is the DataFrame itself, not a Path and not a dict.
        # This is the whole point of the "dataframe" mode — skip disk I/O for
        # the stats output when the caller wants the result in-memory, and hand
        # back a dataframe when a single config was requested.
        assert isinstance(
            result, pd.DataFrame
        ), f"expected DataFrame return, got {type(result).__name__}"
        assert len(result) == len(sample_df)
        assert "dem_local_mean_100m" in result.columns
        # Core columns (occurrenceID/decimalLatitude/decimalLongitude) are
        # preserved on the stats output so the returned DataFrame is
        # independently useful.
        assert "occurrenceID" in result.columns
        assert list(result["occurrenceID"]) == list(sample_df["occurrenceID"])

    def test_dataframe_format_list_config_keeps_dict(self, sample_df, tmp_path):
        """A list config still returns a {batch_id: df} dict even in dataframe
        mode, so multi-output runs stay addressable by key."""
        outputs = extract(
            sample_df,
            [
                {
                    "batch_id": "df_list",
                    "datasets": ["dem_local"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean"],
                        "window_size_m": 100,
                        "output_file_format": "dataframe",
                    },
                }
            ],
            output_dir=tmp_path,
        )
        assert isinstance(outputs, dict)
        assert isinstance(outputs["df_list"], pd.DataFrame)

    def test_unknown_output_format_raises(self, sample_df, tmp_path):
        """An unrecognised output_file_format value raises ValueError."""
        # Guards against typos like "parquette" / "df" silently being treated
        # as csv (the historical fallthrough behaviour).
        with pytest.raises(ValueError, match="output_file_format"):
            extract(
                sample_df,
                {
                    "batch_id": "bad_fmt",
                    "datasets": ["dem_local"],
                    "settings": {
                        "output_type": "tabular",
                        "statistics": ["mean"],
                        "window_size_m": 100,
                        "output_file_format": "parquette",
                    },
                },
                output_dir=tmp_path,
            )


# ------------------------------------------------------------------
# Numerical correctness — does the pipeline actually compute the right number?
# ------------------------------------------------------------------
#
# Every other test in this file checks schema, column names, or NaN-ness.
# These tests use a constant-valued raster (every pixel = constant_dem_value)
# so the expected mean/std/min/max are known analytically. If the reducer
# wiring breaks numerically — e.g. nodata sentinels leak into the math, or
# windows pick up neighbouring tiles — these are the assertions that catch it.


class TestNumericalCorrectness:
    def test_mean_over_constant_raster_equals_constant(
        self, sample_df, constant_dem_tif, constant_dem_value, tmp_path
    ):
        """A constant raster's window mean must equal the constant value."""
        # Register the constant raster as a one-off dataset. The autouse
        # register_test_catalog fixture's reset_catalog() teardown cleans it
        # up at the end of the test, so we don't pollute later tests.
        update_catalog(
            {
                "datasets": {
                    "dem_constant": {
                        "data_source": "local",
                        "path": str(constant_dem_tif),
                        "bands": 1,
                    }
                }
            }
        )

        outputs = extract(
            sample_df,
            {
                "batch_id": "numerical_test",
                "datasets": ["dem_constant"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean", "min", "max", "std"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["numerical_test"])

        # Every pixel is `constant_dem_value`, so the window aggregate must be
        # exactly that constant for every reducer-that-sees-data. Using
        # pytest.approx with a small absolute tolerance to absorb the
        # stats_output_decimals rounding the pipeline applies on write.
        assert stats_df["dem_constant_mean_100m"].iloc[0] == pytest.approx(constant_dem_value)
        assert stats_df["dem_constant_min_100m"].iloc[0] == pytest.approx(constant_dem_value)
        assert stats_df["dem_constant_max_100m"].iloc[0] == pytest.approx(constant_dem_value)
        # Standard deviation of a single repeated value is exactly 0 (every
        # deviation from the mean is zero).
        assert stats_df["dem_constant_std_100m"].iloc[0] == pytest.approx(0.0)

    def test_mean_constant_across_all_points(
        self, sample_df, constant_dem_tif, constant_dem_value, tmp_path
    ):
        """Every sample point on a constant raster gets the same mean — no per-point drift."""
        # Catches a class of bugs where one point's window accidentally picks
        # up neighbour data, returns NaN, or falls through to a different
        # code path than the others.
        update_catalog(
            {
                "datasets": {
                    "dem_constant": {
                        "data_source": "local",
                        "path": str(constant_dem_tif),
                        "bands": 1,
                    }
                }
            }
        )

        outputs = extract(
            sample_df,
            {
                "batch_id": "uniform_test",
                "datasets": ["dem_constant"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean"],
                    "window_size_m": 200,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["uniform_test"])
        # All N points must report exactly the constant value — no NaN, no drift.
        np.testing.assert_allclose(
            stats_df["dem_constant_mean_200m"].to_numpy(),
            np.full(len(sample_df), constant_dem_value),
        )

    def test_point_sampling_returns_exact_pixel_value(
        self, sample_df, constant_dem_tif, constant_dem_value, tmp_path
    ):
        """The 'point' reducer samples the underlying pixel value exactly."""
        # On a constant raster every pixel has the same value, so wherever the
        # point lands the sampled value must equal that constant. If the
        # adapter is doing any unexpected resampling / interpolation, this is
        # where it would show up.
        update_catalog(
            {
                "datasets": {
                    "dem_constant": {
                        "data_source": "local",
                        "path": str(constant_dem_tif),
                        "bands": 1,
                    }
                }
            }
        )

        outputs = extract(
            sample_df,
            {
                "batch_id": "point_numeric",
                "datasets": ["dem_constant"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["point"],
                    "window_size_m": 100,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["point_numeric"])
        # Every sampled point must read exactly the constant value.
        np.testing.assert_allclose(
            stats_df["dem_constant_point"].to_numpy(),
            np.full(len(sample_df), constant_dem_value),
        )


# ------------------------------------------------------------------
# Point sample of NaN and infinite pixels
# ------------------------------------------------------------------

# Two points about 100 m apart in UTM zone 34N. The raster below puts the
# first point in pixel (row 10, col 10) and the second in pixel (row 10, col 20).
_NON_FINITE_POINTS = pd.DataFrame(
    {
        "occurrenceID": ["non_finite_pixel", "valid_pixel"],
        "decimalLatitude": [62.976878, 62.976878],
        "decimalLongitude": [18.026823, 18.028790],
    }
)


def _write_raster_without_nodata(path: Path, first_point_pixel_values: list[float]) -> np.ndarray:
    """Write a float32 UTM 34N raster (10 m pixels) that declares no nodata value.

    The pixel of the first point of ``_NON_FINITE_POINTS`` gets
    ``first_point_pixel_values`` (one value per band). The other pixels are
    random values from a seeded generator. Returns the written array
    (bands, rows, columns).
    """
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
    first_x, first_y = to_utm.transform(
        _NON_FINITE_POINTS["decimalLongitude"][0], _NON_FINITE_POINTS["decimalLatitude"][0]
    )
    # 30 x 30 pixels of 10 m; the first point is in the middle of pixel (10, 10).
    west, north = first_x - 105.0, first_y + 105.0
    transform = from_bounds(west, north - 300.0, west + 300.0, north, 30, 30)

    band_count = len(first_point_pixel_values)
    rng = np.random.default_rng(seed=7)
    pixel_values = rng.uniform(100.0, 200.0, (band_count, 30, 30)).astype(np.float32)
    for band_index, pixel_value in enumerate(first_point_pixel_values):
        pixel_values[band_index, 10, 10] = pixel_value

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=30,
        width=30,
        count=band_count,
        dtype="float32",
        crs="EPSG:32634",
        transform=transform,
    ) as dst:
        dst.write(pixel_values)
    return pixel_values


class TestPointSampleNonFinitePixels:
    """A NaN or infinite pixel is missing in the point sample, also without a declared nodata."""

    def test_nan_pixel_without_nodata_is_missing(self, tmp_path):
        """A NaN pixel gives a null point value and the QC values of a nodata pixel."""
        raster_path = tmp_path / "nan_pixel.tif"
        pixel_values = _write_raster_without_nodata(raster_path, [float("nan")])
        update_catalog(
            {"datasets": {"nan_raster": {"data_source": "local", "path": str(raster_path)}}}
        )
        with rasterio.open(raster_path) as src:
            assert src.nodata is None

        outputs = extract(
            _NON_FINITE_POINTS,
            {
                "batch_id": "nan_point",
                "datasets": ["nan_raster"],
                "settings": {"output_type": "tabular", "statistics": ["point"]},
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["nan_point"])
        qc_df = pd.read_csv(tmp_path / "nan_point_qc.csv")
        assert pd.isna(stats_df.loc[0, "nan_raster_point"])
        assert stats_df.loc[1, "nan_raster_point"] == pytest.approx(float(pixel_values[0, 10, 20]))
        assert qc_df["nan_raster_in_extent_point"].tolist() == [False, True]
        assert qc_df["nan_raster_n_pixels_point"].tolist() == [0, 1]
        assert qc_df["nan_raster_had_nodata_point"].tolist() == [True, False]
        assert qc_df["nan_raster_coverage_pct_point"].tolist() == [0.0, 100.0]

    @pytest.mark.parametrize("pixel_value", [float("nan"), float("inf"), float("-inf")])
    def test_single_band_value_is_none(self, tmp_path, pixel_value):
        """The adapter returns None (not NaN or infinity) for a non-finite pixel."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        raster_path = tmp_path / "non_finite_pixel.tif"
        _write_raster_without_nodata(raster_path, [pixel_value])

        with LocalRasterAdapter({"path": str(raster_path), "bands": 1}) as adapter:
            values, meta = adapter._sample_point_single(
                _NON_FINITE_POINTS["decimalLatitude"][0], _NON_FINITE_POINTS["decimalLongitude"][0]
            )

        assert values == {"point": None}
        assert meta["in_extent"] is False
        assert meta["n_pixels"] == 0
        assert "error" not in meta

    def test_multi_band_non_finite_values_are_none(self, tmp_path):
        """In a multi-band raster, NaN and infinite bands are None, and a finite band keeps its value."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        raster_path = tmp_path / "multi_band.tif"
        _write_raster_without_nodata(raster_path, [float("nan"), float("inf"), 42.5])
        lat = _NON_FINITE_POINTS["decimalLatitude"][0]
        lon = _NON_FINITE_POINTS["decimalLongitude"][0]

        with LocalRasterAdapter({"path": str(raster_path), "bands": [1, 2, 3]}) as adapter:
            values_with_finite_band, meta_with_finite_band = adapter._sample_point_single(lat, lon)
        with LocalRasterAdapter({"path": str(raster_path), "bands": [1, 2]}) as adapter:
            values_without_finite_band, meta_without_finite_band = adapter._sample_point_single(
                lat, lon
            )

        assert values_with_finite_band == {"b1": None, "b2": None, "b3": 42.5}
        assert meta_with_finite_band["in_extent"] is True
        assert values_without_finite_band == {"b1": None, "b2": None}
        assert meta_without_finite_band["in_extent"] is False
        assert meta_without_finite_band["n_pixels"] == 0


# ------------------------------------------------------------------
# Windows smaller than one pixel, and pixel sizes in metres
# ------------------------------------------------------------------

# One point in UTM zone 34N (the first point of sample_df).
_SUB_PIXEL_POINT = pd.DataFrame(
    {
        "occurrenceID": ["corner_point"],
        "decimalLatitude": [62.976878],
        "decimalLongitude": [18.026823],
    }
)


def _write_coarse_raster_near_a_pixel_corner(path: Path) -> np.ndarray:
    """Write a 10 × 10 float32 UTM 34N raster with 100 m pixels around ``_SUB_PIXEL_POINT``.

    The point lies 10 m east of a pixel column edge and 10 m south of a pixel
    row edge, inside pixel (row 5, column 5). The nearest pixel centres are
    40 m and 60 m away in each direction, so a 50 m window contains no pixel
    centre. A window of two pixels (200 m) contains the centres of rows 4-5
    and columns 4-5. Returns the written array (rows, columns).
    """
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
    point_x, point_y = to_utm.transform(
        _SUB_PIXEL_POINT["decimalLongitude"][0], _SUB_PIXEL_POINT["decimalLatitude"][0]
    )
    west, north = point_x - 10.0 - 500.0, point_y + 10.0 + 500.0
    transform = from_bounds(west, north - 1000.0, west + 1000.0, north, 10, 10)

    random_generator = np.random.default_rng(seed=20)
    pixel_values = random_generator.uniform(100.0, 200.0, (10, 10)).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:32634",
        transform=transform,
    ) as dst:
        dst.write(pixel_values, 1)
    return pixel_values


class TestWindowSmallerThanOnePixel:
    """A local window smaller than one native pixel is widened to two pixels, as on GEE."""

    def test_window_gets_the_pixels_of_a_two_pixel_window(self, tmp_path):
        """A 50 m window at 100 m pixels gives the four pixels around the point, named 50m."""
        raster_path = tmp_path / "coarse.tif"
        pixel_values = _write_coarse_raster_near_a_pixel_corner(raster_path)
        update_catalog({"datasets": {"coarse": {"data_source": "local", "path": str(raster_path)}}})

        outputs = extract(
            _SUB_PIXEL_POINT,
            {
                "batch_id": "sub_pixel",
                "datasets": ["coarse"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean", "count"],
                    "window_size_m": 50,
                },
            },
            output_dir=tmp_path,
        )

        stats_df = pd.read_csv(outputs["sub_pixel"])
        qc_df = pd.read_csv(tmp_path / "sub_pixel_qc.csv")
        sidecar = json.loads((tmp_path / "sub_pixel_metadata.json").read_text())
        expected_mean = float(pixel_values[4:6, 4:6].astype(float).mean())
        assert stats_df.loc[0, "coarse_mean_50m"] == pytest.approx(expected_mean)
        assert stats_df.loc[0, "coarse_count_50m"] == 4
        assert qc_df.loc[0, "coarse_in_extent_50m"]
        assert qc_df.loc[0, "coarse_n_pixels_50m"] == 4
        # The widened window has the 4 pixel centres, all with data.
        assert qc_df.loc[0, "coarse_coverage_pct_50m"] == 100.0
        assert not qc_df.loc[0, "coarse_had_nodata_50m"]
        # The outputs keep the requested window, not the widened one.
        assert sidecar["config"]["window_size_m"] == 50
        assert set(sidecar["datasets"]["coarse"]["quality"]) == {"50m"}
        assert sidecar["datasets"]["coarse"]["native_spatial_resolution_m"] == 100.0

    def test_meta_keeps_the_requested_window(self, tmp_path):
        """The adapter reads a 200 m window but reports ``window_m`` 50."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        raster_path = tmp_path / "coarse.tif"
        _write_coarse_raster_near_a_pixel_corner(raster_path)

        with LocalRasterAdapter({"path": str(raster_path), "bands": 1}) as adapter:
            [(stats, meta)] = adapter.fetch_stats_batch(
                _SUB_PIXEL_POINT["decimalLatitude"],
                _SUB_PIXEL_POINT["decimalLongitude"],
                50,
                ["count"],
                disable_progress=True,
            )

        assert stats["count"] == 4
        assert meta["n_pixels"] == 4
        assert meta["window_m"] == 50

    def test_window_of_one_pixel_or_more_is_not_widened(self, tmp_path):
        """A window of one pixel (100 m at 100 m pixels) is read as requested, not widened."""
        from envoi.adapters.local_adapter import LocalRasterAdapter

        raster_path = tmp_path / "coarse.tif"
        _write_coarse_raster_near_a_pixel_corner(raster_path)

        with LocalRasterAdapter({"path": str(raster_path), "bands": 1}) as adapter:
            [(stats, meta)] = adapter.fetch_stats_batch(
                _SUB_PIXEL_POINT["decimalLatitude"],
                _SUB_PIXEL_POINT["decimalLongitude"],
                100,
                ["count"],
                disable_progress=True,
            )

        # The 100 m window reaches 50 m from the point; the nearest centres are 40 m
        # away in one direction and 60 m in the other, so it holds one pixel centre.
        # A widened (200 m) window would hold four.
        assert stats["count"] == 1
        assert meta["n_pixels"] == 1
        assert meta["window_m"] == 100


# ------------------------------------------------------------------
# Coverage of local windows: the pixels whose centre is inside the window
# ------------------------------------------------------------------

_COVERAGE_PIXEL_SIZE_M = 10.0
_COVERAGE_RASTER_SIZE = 40  # pixels per side
_COVERAGE_WINDOW_M = 100
_COVERAGE_NODATA = -9999.0


def _coverage_grid() -> tuple[float, float, float, float]:
    """Return ``(west, north, point_x, point_y)`` of the coverage raster in EPSG:32634.

    The point of ``_SUB_PIXEL_POINT`` is 2.5 m east of the west edge and 2.5 m
    south of the north edge of pixel (row 20, column 20). No pixel centre is
    then on the edge of a 100 m window, and the bounding box of the window
    has pixels whose centre is outside the window.
    """
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
    point_x, point_y = to_utm.transform(
        _SUB_PIXEL_POINT["decimalLongitude"][0], _SUB_PIXEL_POINT["decimalLatitude"][0]
    )
    return point_x - 202.5, point_y + 202.5, point_x, point_y


def _pixels_inside_coverage_window() -> np.ndarray:
    """Return a (rows, columns) mask of the pixels whose centre is inside the 100 m window."""
    west, north, point_x, point_y = _coverage_grid()
    pixel_numbers = np.arange(_COVERAGE_RASTER_SIZE)
    centre_x = west + (pixel_numbers + 0.5) * _COVERAGE_PIXEL_SIZE_M
    centre_y = north - (pixel_numbers + 0.5) * _COVERAGE_PIXEL_SIZE_M
    half_window_m = _COVERAGE_WINDOW_M / 2
    inside_columns = np.abs(centre_x - point_x) < half_window_m
    inside_rows = np.abs(centre_y - point_y) < half_window_m
    return inside_rows[:, np.newaxis] & inside_columns[np.newaxis, :]


def _write_coverage_raster(
    path: Path, pixel_values: np.ndarray, nodata=None, first_column: int = 0
) -> None:
    """Write ``pixel_values`` (bands, rows, columns) as a float32 GeoTIFF on the coverage grid.

    The file has the columns from ``first_column`` on, so with a
    ``first_column`` above 0 its west edge is east of the grid's west edge.
    """
    west, north, _, _ = _coverage_grid()
    file_west = west + first_column * _COVERAGE_PIXEL_SIZE_M
    file_width = _COVERAGE_RASTER_SIZE - first_column
    profile = {
        "driver": "GTiff",
        "height": _COVERAGE_RASTER_SIZE,
        "width": file_width,
        "count": pixel_values.shape[0],
        "dtype": "float32",
        "crs": "EPSG:32634",
        "transform": from_bounds(
            file_west,
            north - _COVERAGE_RASTER_SIZE * _COVERAGE_PIXEL_SIZE_M,
            file_west + file_width * _COVERAGE_PIXEL_SIZE_M,
            north,
            file_width,
            _COVERAGE_RASTER_SIZE,
        ),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(pixel_values[:, :, first_column:].astype(np.float32))


def _run_coverage_extraction(raster_path: Path, bands, output_dir: Path) -> tuple:
    """Run a tabular extraction of the coverage raster with ``min_coverage_pct`` 95.

    Returns ``(stats_df, qc_df, low_coverage_warnings)``.
    """
    update_catalog(
        {"datasets": {"cover": {"data_source": "local", "path": str(raster_path), "bands": bands}}}
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        outputs = extract(
            _SUB_PIXEL_POINT,
            {
                "batch_id": "coverage",
                "datasets": ["cover"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["mean", "count"],
                    "window_size_m": _COVERAGE_WINDOW_M,
                    "min_coverage_pct": 95,
                },
            },
            output_dir=output_dir,
        )
    low_coverage_warnings = [
        str(warning.message)
        for warning in caught_warnings
        if str(warning.message).startswith("Low coverage")
    ]
    stats_df = pd.read_csv(outputs["coverage"])
    qc_df = pd.read_csv(output_dir / "coverage_qc.csv")
    return stats_df, qc_df, low_coverage_warnings


class TestLocalWindowCoverage:
    """``coverage_pct`` and ``had_nodata`` count only the pixels whose centre is inside the window."""

    def test_complete_window_has_full_coverage(self, tmp_path):
        """A window without missing data gives 100 % coverage, no nodata, and no low-coverage warning."""
        raster_path = tmp_path / "constant.tif"
        _write_coverage_raster(
            raster_path, np.full((1, _COVERAGE_RASTER_SIZE, _COVERAGE_RASTER_SIZE), 5.0)
        )

        stats_df, qc_df, low_coverage_warnings = _run_coverage_extraction(
            raster_path, 1, tmp_path / "out"
        )

        # 10 × 10 pixel centres are inside the window. Its bounding box has
        # 11 × 11 pixels, which gave 82.6 % before.
        assert int(_pixels_inside_coverage_window().sum()) == 100
        assert stats_df.loc[0, "cover_count_100m"] == 100
        assert qc_df.loc[0, "cover_n_pixels_100m"] == 100
        assert qc_df.loc[0, "cover_coverage_pct_100m"] == 100.0
        assert not qc_df.loc[0, "cover_had_nodata_100m"]
        assert low_coverage_warnings == []

    @pytest.mark.parametrize("bands", [1, [1, 2]])
    def test_nodata_pixels_inside_the_window_lower_the_coverage(self, tmp_path, bands):
        """Nodata pixels inside the window lower the coverage; those outside it do not."""
        inside_window = _pixels_inside_coverage_window()
        random_generator = np.random.default_rng(seed=26)
        pixel_values = random_generator.uniform(
            100.0, 200.0, (2, _COVERAGE_RASTER_SIZE, _COVERAGE_RASTER_SIZE)
        )
        # Band 1: 7 nodata pixels inside the window, and 5 in the bounding box
        # of the window whose centre is outside it (row 25, column 25).
        for row, column in [(15, 15), (16, 22), (18, 18), (20, 20), (22, 16), (23, 24), (24, 24)]:
            pixel_values[0, row, column] = _COVERAGE_NODATA
        for row, column in [(25, 15), (25, 20), (25, 25), (15, 25), (20, 25)]:
            pixel_values[0, row, column] = _COVERAGE_NODATA
        # Band 2: 3 more nodata pixels inside the window.
        for row, column in [(17, 19), (19, 21), (21, 23)]:
            pixel_values[1, row, column] = _COVERAGE_NODATA
        raster_path = tmp_path / "with_nodata.tif"
        _write_coverage_raster(raster_path, pixel_values, nodata=_COVERAGE_NODATA)

        stats_df, qc_df, low_coverage_warnings = _run_coverage_extraction(
            raster_path, bands, tmp_path / "out"
        )

        # A pixel is valid when it has data in every selected band.
        selected_values = pixel_values[:1] if bands == 1 else pixel_values
        valid_pixels = inside_window & (selected_values != _COVERAGE_NODATA).all(axis=0)
        expected_valid = int(valid_pixels.sum())
        assert expected_valid == (93 if bands == 1 else 90)
        expected_coverage = 100.0 * expected_valid / int(inside_window.sum())
        count_column = "cover_count_100m" if bands == 1 else "cover_b1_count_100m"
        assert stats_df.loc[0, count_column] == expected_valid
        assert qc_df.loc[0, "cover_n_pixels_100m"] == expected_valid
        assert qc_df.loc[0, "cover_coverage_pct_100m"] == pytest.approx(expected_coverage)
        assert qc_df.loc[0, "cover_had_nodata_100m"]
        assert len(low_coverage_warnings) == 1

    def test_window_half_beyond_the_raster_edge_has_half_coverage(self, tmp_path):
        """The part of the window beyond the raster edge counts as missing data, as on GEE."""
        # The file starts at column 20 of the grid, the column of the point,
        # so its west edge is 2.5 m west of the point.
        raster_path = tmp_path / "edge.tif"
        _write_coverage_raster(
            raster_path,
            np.full((1, _COVERAGE_RASTER_SIZE, _COVERAGE_RASTER_SIZE), 5.0),
            first_column=20,
        )

        stats_df, qc_df, low_coverage_warnings = _run_coverage_extraction(
            raster_path, 1, tmp_path / "out"
        )

        # The window has the centres of columns 15-24 of the grid; the file
        # has columns 20-24 of them.
        inside_window = _pixels_inside_coverage_window()
        expected_valid = int(inside_window[:, 20:].sum())
        expected_coverage = 100.0 * expected_valid / int(inside_window.sum())
        assert (expected_valid, expected_coverage) == (50, 50.0)
        assert stats_df.loc[0, "cover_count_100m"] == expected_valid
        assert qc_df.loc[0, "cover_in_extent_100m"]
        assert qc_df.loc[0, "cover_n_pixels_100m"] == expected_valid
        assert qc_df.loc[0, "cover_coverage_pct_100m"] == pytest.approx(expected_coverage)
        assert qc_df.loc[0, "cover_had_nodata_100m"]
        assert len(low_coverage_warnings) == 1

    def test_nan_pixels_of_a_file_without_nodata_are_missing(self, tmp_path):
        """A single band without nodata: NaN pixels are not counted in n_pixels and lower the coverage."""
        pixel_values = np.full((1, _COVERAGE_RASTER_SIZE, _COVERAGE_RASTER_SIZE), 5.0)
        pixel_values[0, 18:20, 18:20] = np.nan  # 4 pixels inside the window
        raster_path = tmp_path / "nan_pixels.tif"
        _write_coverage_raster(raster_path, pixel_values)

        stats_df, qc_df, _ = _run_coverage_extraction(raster_path, 1, tmp_path / "out")

        assert stats_df.loc[0, "cover_count_100m"] == 96
        assert qc_df.loc[0, "cover_n_pixels_100m"] == 96
        assert qc_df.loc[0, "cover_coverage_pct_100m"] == 96.0
        assert qc_df.loc[0, "cover_had_nodata_100m"]

    @pytest.mark.parametrize("gdal_cuts_at_180", [True, False])
    def test_window_across_the_antimeridian(self, tmp_path, monkeypatch, gdal_cuts_at_180):
        """A window across 180° of a raster in degrees counts its part beyond 180° as missing."""
        from pyproj import Transformer
        from shapely.geometry import Point, Polygon, mapping, shape

        from envoi.adapters import local_adapter

        if not gdal_cuts_at_180:
            # Some GDAL builds (the macOS wheels of rasterio in CI) do not cut a
            # transformed geometry at 180°. Simulate them: project each corner
            # on its own, so the window keeps corners on both sides of 180°.
            def uncut_transform_geom(src_crs, dst_crs, geometry, precision=-1):
                transformer = Transformer.from_crs(str(src_crs), str(dst_crs), always_xy=True)
                ring = shape(geometry).exterior.coords
                return mapping(Polygon([transformer.transform(x, y) for x, y in ring]))

            monkeypatch.setattr(local_adapter, "transform_geom", uncut_transform_geom)

        # 100 x 100 pixels of 0.0005° that end at 180°. The 1000 m window of
        # the point reaches about 0.0025° beyond 180°.
        pixel_size_deg = 0.0005
        raster_path = tmp_path / "antimeridian.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_bounds(179.95, 0.0, 180.0, 0.05, 100, 100),
        ) as dst:
            dst.write(np.full((1, 100, 100), 5.0, dtype=np.float32))
        update_catalog(
            {"datasets": {"east_edge": {"data_source": "local", "path": str(raster_path)}}}
        )
        point_lat, point_lon = 0.025, 179.998

        # Record the size of every mask, to check that no mask spans 360°.
        mask_shapes = []
        original_geometry_mask = local_adapter.geometry_mask

        def recording_geometry_mask(*args, **kwargs):
            mask_shapes.append(kwargs["out_shape"])
            return original_geometry_mask(*args, **kwargs)

        monkeypatch.setattr(local_adapter, "geometry_mask", recording_geometry_mask)

        outputs = extract(
            pd.DataFrame(
                {
                    "occurrenceID": ["edge"],
                    "decimalLatitude": [point_lat],
                    "decimalLongitude": [point_lon],
                }
            ),
            {
                "batch_id": "antimeridian",
                "datasets": ["east_edge"],
                "settings": {
                    "output_type": "tabular",
                    "statistics": ["count"],
                    "window_size_m": 1000,
                },
            },
            output_dir=tmp_path / "out",
        )

        # Independent count: the window corners in degrees (UTM zone 60N),
        # with the longitudes continued past 180°, and the pixel centres of
        # the raster grid continued 20 columns past 180°.
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32660", always_xy=True)
        to_degrees = Transformer.from_crs("EPSG:32660", "EPSG:4326", always_xy=True)
        point_x, point_y = to_utm.transform(point_lon, point_lat)
        corners = [
            to_degrees.transform(point_x + offset_x, point_y + offset_y)
            for offset_x, offset_y in ((-500, -500), (500, -500), (500, 500), (-500, 500))
        ]
        window = Polygon([(corner_lon % 360.0, corner_lat) for corner_lon, corner_lat in corners])
        inside_pixels = [
            (row, column)
            for row in range(100)
            for column in range(120)
            if window.contains(
                Point(
                    179.95 + (column + 0.5) * pixel_size_deg,
                    0.05 - (row + 0.5) * pixel_size_deg,
                )
            )
        ]
        expected_total = len(inside_pixels)
        expected_valid = sum(1 for _, column in inside_pixels if column < 100)
        assert expected_valid < expected_total

        stats_df = pd.read_csv(outputs["antimeridian"])
        qc_df = pd.read_csv(tmp_path / "out" / "antimeridian_qc.csv")
        assert stats_df.loc[0, "east_edge_count_1000m"] == expected_valid
        assert qc_df.loc[0, "east_edge_n_pixels_1000m"] == expected_valid
        assert qc_df.loc[0, "east_edge_coverage_pct_1000m"] == pytest.approx(
            100.0 * expected_valid / expected_total
        )
        assert qc_df.loc[0, "east_edge_had_nodata_1000m"]
        assert max(width for _, width in mask_shapes) < 1000


class TestNativeResolutionInMetres:
    """The sidecar field ``native_spatial_resolution_m`` is in metres for every CRS."""

    def test_geographic_raster(self, tmp_path):
        """A raster with 0.001° pixels reports 111.32 m (111 319.49 m per degree)."""
        raster_path = tmp_path / "geographic.tif"
        random_generator = np.random.default_rng(seed=22)
        pixel_values = random_generator.uniform(0.0, 100.0, (50, 50)).astype(np.float32)
        # 0.05° × 0.05° around the point (lat 62.98, lon 18.03).
        transform = from_bounds(18.0, 62.95, 18.05, 63.0, 50, 50)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=50,
            width=50,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(pixel_values, 1)
        update_catalog(
            {"datasets": {"geographic": {"data_source": "local", "path": str(raster_path)}}}
        )

        extract(
            _SUB_PIXEL_POINT,
            {
                "batch_id": "geographic_run",
                "datasets": ["geographic"],
                "settings": {"output_type": "tabular", "statistics": ["mean"]},
            },
            output_dir=tmp_path,
        )

        sidecar = json.loads((tmp_path / "geographic_run_metadata.json").read_text())
        assert sidecar["datasets"]["geographic"]["native_spatial_resolution_m"] == 111.32


# ------------------------------------------------------------------
# Tile CRS in the metadata sidecar of local raster runs
# ------------------------------------------------------------------

# Two points on both sides of 18° E, the border between UTM zones 33N and 34N.
_TWO_ZONE_POINTS = pd.DataFrame(
    {
        "occurrenceID": ["zone33", "zone34"],
        "decimalLatitude": [62.95, 62.95],
        "decimalLongitude": [17.95, 18.05],
    }
)


class TestLocalTileCrsMetadata:
    """The sidecar ``tile_crs`` of a local dataset names the CRS of its tiles."""

    @pytest.fixture
    def two_zone_raster(self, tmp_path) -> str:
        """Register a geographic raster that covers both points of ``_TWO_ZONE_POINTS``."""
        raster_path = tmp_path / "two_zones.tif"
        random_generator = np.random.default_rng(seed=31)
        pixel_values = random_generator.uniform(0.0, 100.0, (100, 200)).astype(np.float32)
        # 0.2° × 0.1° with 0.001° pixels, from 17.9° E to 18.1° E.
        transform = from_bounds(17.9, 62.9, 18.1, 63.0, 200, 100)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=100,
            width=200,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(pixel_values, 1)
        update_catalog(
            {"datasets": {"two_zones": {"data_source": "local", "path": str(raster_path)}}}
        )
        return "two_zones"

    def _export_tiles(self, tmp_path, **settings) -> dict:
        """Export the tiles of both points and return the sidecar entry of the dataset."""
        extract(
            _TWO_ZONE_POINTS,
            {
                "batch_id": "zones",
                "datasets": ["two_zones"],
                "settings": {"output_type": "raster", "window_size_m": 200, **settings},
            },
            output_dir=tmp_path,
        )
        sidecar = json.loads((tmp_path / "zones" / "zones_metadata.json").read_text())
        return sidecar["datasets"]["two_zones"]

    def test_resampled_tiles_list_the_utm_zone_of_each_point(self, two_zone_raster, tmp_path):
        """With resample_m, tile_crs lists the UTM zones of the points, which the tiles use."""
        dataset_meta = self._export_tiles(tmp_path, resample_m=30)

        assert dataset_meta["tile_crs"] == ["EPSG:32633", "EPSG:32634"]
        assert dataset_meta["native_crs"] == "EPSG:4326"
        tile_dir = tmp_path / "zones" / "two_zones"
        tile_crs_codes = []
        for tile_name in ("zone33-two_zones.tif", "zone34-two_zones.tif"):
            with rasterio.open(tile_dir / tile_name) as src:
                tile_crs_codes.append(f"EPSG:{src.crs.to_epsg()}")
        assert tile_crs_codes == ["EPSG:32633", "EPSG:32634"]

    def test_native_tiles_keep_the_raster_crs(self, two_zone_raster, tmp_path):
        """Without resample_m, tiles are in the raster CRS, and tile_crs names it."""
        dataset_meta = self._export_tiles(tmp_path)

        assert dataset_meta["tile_crs"] == "EPSG:4326"
        with rasterio.open(tmp_path / "zones" / "two_zones" / "zone33-two_zones.tif") as src:
            assert src.crs.to_epsg() == 4326


# ------------------------------------------------------------------
# Point sample without the deprecated affine "*" operator
# ------------------------------------------------------------------


class TestPointSampleWithoutAffineOperator:
    """The point sample computes pixel coordinates from the inverse transform's coefficients."""

    def test_rotated_raster_gives_the_pixel_of_the_point_without_warnings(self, tmp_path):
        """On a rotated raster, the point sample reads the right pixel without affine warnings."""
        from pyproj import Transformer
        from rasterio.transform import Affine

        from envoi.adapters.local_adapter import LocalRasterAdapter

        # A raster rotated by 30°, with 10 m pixels: x = c + a·col + b·row,
        # y = f + d·col + e·row. The first point is inside pixel (row 15,
        # column 12). The second point is inside the raster's axis-aligned
        # bounding box, but outside its rotated footprint (column -2).
        pixel_m, angle = 10.0, math.radians(30)
        a, b = pixel_m * math.cos(angle), pixel_m * math.sin(angle)
        d, e = pixel_m * math.sin(angle), -pixel_m * math.cos(angle)
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
        point_x, point_y = to_utm.transform(18.026823, 62.976878)
        c = point_x - (a * 12.3 + b * 15.6)
        f = point_y - (d * 12.3 + e * 15.6)
        outside_x, outside_y = c + a * -2.0 + b * 20.0, f + d * -2.0 + e * 20.0

        random_generator = np.random.default_rng(seed=23)
        pixel_values = random_generator.uniform(100.0, 200.0, (30, 30)).astype(np.float32)
        raster_path = tmp_path / "rotated.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=30,
            width=30,
            count=1,
            dtype="float32",
            crs="EPSG:32634",
            transform=Affine(a, b, c, d, e, f),
        ) as dst:
            dst.write(pixel_values, 1)

        # Independent check of the test geometry: the inverse of the 3 × 3
        # affine matrix maps each point to its (column, row).
        affine_matrix = np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]])
        inside_column, inside_row, _ = np.linalg.solve(affine_matrix, [point_x, point_y, 1.0])
        outside_column, outside_row, _ = np.linalg.solve(affine_matrix, [outside_x, outside_y, 1.0])
        assert (math.floor(inside_column), math.floor(inside_row)) == (12, 15)
        assert outside_column < 0 <= outside_row < 30
        to_wgs84 = Transformer.from_crs("EPSG:32634", "EPSG:4326", always_xy=True)
        outside_lon, outside_lat = to_wgs84.transform(outside_x, outside_y)
        corner_xs = [c + a * column + b * row for column in (0, 30) for row in (0, 30)]
        corner_ys = [f + d * column + e * row for column in (0, 30) for row in (0, 30)]
        assert min(corner_xs) < outside_x < max(corner_xs)
        assert min(corner_ys) < outside_y < max(corner_ys)

        # affine 3 warns about "*" with PendingDeprecationWarning (and may move
        # to DeprecationWarning). The filters apply to warnings that envoi's own
        # code triggers, not to warnings inside other libraries.
        with (
            LocalRasterAdapter({"path": str(raster_path), "bands": 1}) as adapter,
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings("error", category=PendingDeprecationWarning, module=r"envoi\.")
            warnings.filterwarnings("error", category=DeprecationWarning, module=r"envoi\.")
            inside_values, inside_meta = adapter._sample_point_single(62.976878, 18.026823)
            outside_values, outside_meta = adapter._sample_point_single(outside_lat, outside_lon)

        assert inside_values == {"point": pytest.approx(float(pixel_values[15, 12]))}
        assert inside_meta["in_extent"] is True
        assert outside_values == {"point": None}
        assert outside_meta["in_extent"] is False


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
        from envoi.geo import get_utm_crs

        # Northern hemisphere: zones 32601..32660. Southern: 32701..32760.
        # The naive `(lon + 180) / 6 + 1` formula gives 61 at lon == 180,
        # which would yield EPSG:32661 — outside the UTM range.
        assert get_utm_crs(180.0, 0.0) == "EPSG:32660"
        assert get_utm_crs(180.0, -1.0) == "EPSG:32760"
        # Spot-check that an ordinary longitude still resolves correctly.
        assert get_utm_crs(0.0, 0.0) == "EPSG:32631"


# ------------------------------------------------------------------
# Categorical reducers — class_count and class_fraction expansion.
# ------------------------------------------------------------------
#
# These tests use a small uint8 raster with hand-picked class values so the
# expected per-class counts can be asserted exactly. The raster is split
# into four equal quadrants, each filled with a different class id, so any
# point near the centre sees a known mix of classes in its window.


# Pixel size and dimensions match the DEM fixture in conftest.py so the
# same sample points (around UTM 34N x≈349500, y≈6988500) fall inside this
# raster too. The four-class quadrant layout still gives every centred
# point at least two classes in any reasonable window.
_LULC_RES_M = 10.0
_LULC_HEIGHT = 500
_LULC_WIDTH = 500
_LULC_ORIGIN_X = 347020.0
_LULC_ORIGIN_Y = 6988980.0
_LULC_CRS = "EPSG:32634"

# Class IDs the synthetic raster carries — four quadrants, four classes.
# Picked to be visually distinct (not consecutive) so accidental off-by-one
# class-id bugs show up clearly in output columns.
_LULC_CLASSES = [10, 20, 30, 40]


@pytest.fixture(scope="session")
def categorical_tif(tmp_path_factory) -> Path:
    """Synthetic single-band uint8 categorical raster split into 4 quadrants.

    Each quadrant carries one of ``_LULC_CLASSES`` so any centred sample
    point with a window large enough to span ≥ 2 quadrants will exercise
    the multi-class output path. Same CRS / origin as ``dem_tif`` so the
    existing ``sample_df`` points fall inside this raster too.
    """

    arr = np.empty((_LULC_HEIGHT, _LULC_WIDTH), dtype=np.uint8)
    half_h, half_w = _LULC_HEIGHT // 2, _LULC_WIDTH // 2
    # Fill each quadrant with a distinct class id. Top-left → 10, top-right
    # → 20, bottom-left → 30, bottom-right → 40. The exact layout doesn't
    # matter beyond "every class is present in the raster" — the tests
    # assert structural properties (columns exist, fractions sum to ~1.0)
    # rather than exact per-class pixel counts.
    arr[:half_h, :half_w] = _LULC_CLASSES[0]
    arr[:half_h, half_w:] = _LULC_CLASSES[1]
    arr[half_h:, :half_w] = _LULC_CLASSES[2]
    arr[half_h:, half_w:] = _LULC_CLASSES[3]

    fixtures_dir = tmp_path_factory.mktemp("envoi_lulc_fixtures")
    lulc_path = fixtures_dir / "synthetic_lulc.tif"
    transform = from_bounds(
        _LULC_ORIGIN_X,
        _LULC_ORIGIN_Y - _LULC_HEIGHT * _LULC_RES_M,
        _LULC_ORIGIN_X + _LULC_WIDTH * _LULC_RES_M,
        _LULC_ORIGIN_Y,
        _LULC_WIDTH,
        _LULC_HEIGHT,
    )
    with rasterio.open(
        lulc_path,
        "w",
        driver="GTiff",
        height=_LULC_HEIGHT,
        width=_LULC_WIDTH,
        count=1,
        dtype="uint8",
        crs=_LULC_CRS,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)
    return lulc_path


def _run_class_extract(df, tmp_path, statistics, *, window_size_m=200):
    """Run extract() with the synthetic categorical raster and return the stats DataFrame."""
    outputs = extract(
        df,
        {
            "batch_id": "lulc_test",
            "datasets": ["lulc_local"],
            "settings": {
                "output_type": "tabular",
                "statistics": statistics,
                "window_size_m": window_size_m,
            },
        },
        output_dir=tmp_path,
    )
    return pd.read_csv(outputs["lulc_test"])


@pytest.fixture
def lulc_catalog(categorical_tif):
    """Register a categorical local raster pointing at the synthetic LULC fixture."""
    catalog = {
        "datasets": {
            "lulc_local": {
                "data_source": "local",
                "path": str(categorical_tif),
                "bands": 1,
                "data_type": "categorical",
            }
        }
    }
    update_catalog(catalog)
    yield
    reset_catalog()


class TestClassReducerColumns:
    """End-to-end checks of class_count / class_fraction column expansion."""

    def test_class_count_produces_one_column_per_class(self, sample_df, tmp_path, lulc_catalog):
        # The synthetic raster contains every class in _LULC_CLASSES. A 200 m
        # window centred near the middle of the raster spans multiple
        # quadrants, so we expect at least two classes per row — and the
        # batch-level union should include every class observed by at least
        # one row.
        stats_df = _run_class_extract(sample_df, tmp_path, {"categorical": ["class_count"]})
        observed_classes = {
            int(col.split("_class_")[1].split("_count")[0])
            for col in stats_df.columns
            if "_class_" in col and col.endswith("_count_200m")
        }
        # At least two classes appeared in some point's window.
        assert len(observed_classes) >= 2
        assert observed_classes.issubset(set(_LULC_CLASSES))

    def test_class_fraction_per_row_sums_to_one(self, sample_df, tmp_path, lulc_catalog):
        # For every row whose window saw any class at all, the per-class
        # fractions must add up to ~1.0 — that's the definitional contract
        # of class_fraction.
        stats_df = _run_class_extract(sample_df, tmp_path, {"categorical": ["class_fraction"]})
        fraction_columns = [col for col in stats_df.columns if col.endswith("_fraction_200m")]
        assert fraction_columns, "expected at least one class_fraction column"
        for _, row in stats_df.iterrows():
            row_sum = sum(row[col] for col in fraction_columns)
            # A row whose window was entirely out-of-extent will have all
            # zero fractions (the zero-fill policy), so we allow either 0
            # or ~1.0 here. The fraction-sum-to-1 contract is the load-
            # bearing assertion for any row that saw data.
            assert row_sum == pytest.approx(0.0) or row_sum == pytest.approx(1.0, abs=1e-6)

    def test_class_count_and_class_fraction_together(self, sample_df, tmp_path, lulc_catalog):
        # Asking for both reducers in one call should produce both column
        # families. Verifies the dedupe path in the GEE adapter and the
        # ordinary "both reducers run" path in the local adapter.
        stats_df = _run_class_extract(
            sample_df,
            tmp_path,
            {"categorical": ["class_count", "class_fraction"]},
        )
        has_count = any(col.endswith("_count_200m") for col in stats_df.columns)
        has_fraction = any(col.endswith("_fraction_200m") for col in stats_df.columns)
        assert has_count and has_fraction

    def test_absent_class_filled_with_zero(self, sample_df, tmp_path, lulc_catalog):
        # The zero-fill contract: a row whose window didn't contain class X
        # gets 0 (not NaN) for class_X_count and 0.0 for class_X_fraction.
        # We don't know which row contains which class without inspecting
        # the data, but we *do* know that no class column should ever be
        # NaN under the "always 0 for absent" policy.
        stats_df = _run_class_extract(
            sample_df, tmp_path, {"categorical": ["class_count", "class_fraction"]}
        )
        class_columns = [col for col in stats_df.columns if "_class_" in col]
        assert class_columns, "expected at least one class column"
        for col in class_columns:
            assert (
                not stats_df[col].isna().any()
            ), f"column {col} has NaN; expected zero-fill for absent classes"


# ------------------------------------------------------------------
# _append_stat_columns — class-key zero-fill and union-of-classes.
# ------------------------------------------------------------------
#
# These tests poke the helper directly so the zero-fill / union behaviour
# is pinned without going through the extract() pipeline (faster and
# easier to read than a synthetic-raster setup for these edge cases).


class TestAppendStatColumnsClassFill:
    def test_class_union_across_points(self):
        # Two rows: row 0 saw classes 10/20, row 1 saw classes 20/30. The
        # output should include columns for *every* class that any row saw
        # (union), with absent classes filled to 0.
        from envoi._output_assembly import _append_stat_columns

        df = pd.DataFrame({"id": ["a", "b"]})
        stats_results = [
            ({"class_10_count": 5, "class_20_count": 3}, {}),
            ({"class_20_count": 4, "class_30_count": 2}, {}),
        ]
        result, _ = _append_stat_columns(df, "lulc", window_size_m=200, stats_results=stats_results)
        # Union of classes across both rows.
        assert result["lulc_class_10_count_200m"].tolist() == [5, 0]
        assert result["lulc_class_20_count_200m"].tolist() == [3, 4]
        assert result["lulc_class_30_count_200m"].tolist() == [0, 2]

    def test_class_fraction_absent_fills_with_float_zero(self):
        # The zero-fill helper must distinguish count (int 0) from fraction
        # (float 0.0). A row that didn't see a class gets 0.0 in its
        # fraction column, preserving the float dtype.
        from envoi._output_assembly import _append_stat_columns

        df = pd.DataFrame({"id": ["a", "b"]})
        stats_results = [
            ({"class_10_fraction": 0.5, "class_20_fraction": 0.5}, {}),
            ({"class_10_fraction": 1.0}, {}),
        ]
        result, _ = _append_stat_columns(df, "lulc", window_size_m=200, stats_results=stats_results)
        # Row 1's class_20 is absent — must be filled with the float 0.0
        # (not None / NaN / int 0).
        col = "lulc_class_20_fraction_200m"
        assert result[col].tolist() == [0.5, 0.0]
        # Float dtype preserved (pandas would upcast to object on None fill).
        assert result[col].dtype.kind == "f"

    def test_non_class_columns_keep_none_for_missing(self):
        # Regression: only class_* columns get zero-fill. A plain reducer
        # (e.g. mean) that's missing for some row still produces None /
        # NaN as before, so the missing-data signal isn't lost.
        from envoi._output_assembly import _append_stat_columns

        df = pd.DataFrame({"id": ["a", "b"]})
        stats_results = [
            ({"mean": 1.5}, {}),
            ({}, {}),  # row 1 has no stats at all
        ]
        result, _ = _append_stat_columns(df, "dem", window_size_m=100, stats_results=stats_results)
        # Row 1's mean is missing — must be NaN, not 0.
        assert pd.isna(result["dem_mean_100m"].iloc[1])

    def test_multiband_class_columns_filled_per_band(self):
        # Multi-band class keys carry a "b{n}_class_..." prefix. The fill
        # helper must match those too via the optional band prefix in the
        # regex. Each band's class columns are zero-filled independently.
        from envoi._output_assembly import _append_stat_columns

        df = pd.DataFrame({"id": ["a", "b"]})
        stats_results = [
            ({"b1_class_10_count": 5, "b2_class_10_count": 4}, {}),
            ({"b1_class_20_count": 3}, {}),
        ]
        result, _ = _append_stat_columns(df, "lulc", window_size_m=200, stats_results=stats_results)
        assert result["lulc_b1_class_10_count_200m"].tolist() == [5, 0]
        assert result["lulc_b1_class_20_count_200m"].tolist() == [0, 3]
        # b2_class_10 only appears in row 0; row 1 is zero-filled.
        assert result["lulc_b2_class_10_count_200m"].tolist() == [4, 0]

    def test_named_band_class_columns_filled(self):
        """Class keys with a named band prefix are zero-filled for rows without that class."""
        from envoi._output_assembly import _append_stat_columns

        # Earth Engine prefixes class keys with the real band name
        # ("classification"), not with a generic "b{n}" index.
        df = pd.DataFrame({"id": ["a", "b"]})
        stats_results = [
            ({"classification_class_10_count": 5, "classification_class_10_fraction": 0.25}, {}),
            ({"classification_class_12_count": 20, "classification_class_12_fraction": 1.0}, {}),
        ]
        result, _ = _append_stat_columns(
            df, "lulc_naturallands_2020", window_size_m=200, stats_results=stats_results
        )
        prefix = "lulc_naturallands_2020_classification"
        assert result[f"{prefix}_class_10_count_200m"].tolist() == [5, 0]
        assert result[f"{prefix}_class_12_count_200m"].tolist() == [0, 20]
        assert result[f"{prefix}_class_10_fraction_200m"].tolist() == [0.25, 0.0]
        assert result[f"{prefix}_class_12_fraction_200m"].tolist() == [0.0, 1.0]

    def test_failed_point_gets_null_class_columns(self):
        """A failed point (meta has "error") gets null class columns; other rows are zero-filled."""
        from envoi._output_assembly import _append_stat_columns

        df = pd.DataFrame({"id": ["a", "b", "c"]})
        stats_results = [
            ({"class_10_count": 5, "class_10_fraction": 1.0}, {}),
            ({}, {"error": "RasterioIOError: stub: damaged block"}),
            ({"class_20_count": 2, "class_20_fraction": 1.0}, {}),
        ]
        result, _ = _append_stat_columns(df, "lulc", window_size_m=200, stats_results=stats_results)

        assert list(result.columns) == [
            "id",
            "lulc_class_10_count_200m",
            "lulc_class_10_fraction_200m",
            "lulc_class_20_count_200m",
            "lulc_class_20_fraction_200m",
        ]
        assert result["lulc_class_10_count_200m"].tolist()[0::2] == [5, 0]
        assert result["lulc_class_20_fraction_200m"].tolist()[0::2] == [0.0, 1.0]
        for column in result.columns[1:]:
            assert pd.isna(result.loc[1, column]), column


# ------------------------------------------------------------------
# KNOWN_REDUCERS — make sure the registry sets agree.
# ------------------------------------------------------------------


def test_all_known_reducers_includes_class_reducers():
    # Regression check: the validator's allow-list must include the new
    # categorical reducers, otherwise _validate_reducer_names rejects them
    # before the adapter ever sees them.
    from envoi.reducers import KNOWN_REDUCERS

    assert "class_count" in KNOWN_REDUCERS
    assert "class_fraction" in KNOWN_REDUCERS


# ------------------------------------------------------------------
# Tile naming — sample IDs from a user's CSV must never break file writing
# ------------------------------------------------------------------


class TestTileNaming:
    """Raster tiles are named after the user's ID column.

    Real GBIF occurrenceIDs routinely contain characters that are illegal in a
    filename — "/" is read as a path separator on every OS, and ":" is illegal
    on Windows — so the ID is sanitized on its way into the filename. These
    tests pin the behaviour end to end, through the real extract() pipeline.
    """

    # Character set that is valid on Windows, macOS and Linux alike.
    PORTABLE_TILE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.tif$")

    def _points_with_ids(self, sample_ids):
        """Build an input table placing every point inside the synthetic DEM."""
        return pd.DataFrame(
            {
                "occurrenceID": list(sample_ids),
                # Reuse one in-extent coordinate for every point; this suite is
                # about naming, not about the pixel values.
                "decimalLatitude": [62.98] * len(sample_ids),
                "decimalLongitude": [18.025] * len(sample_ids),
            }
        )

    def _export(self, df, tmp_path, batch_id="naming", **settings):
        extract(
            df,
            {
                "batch_id": batch_id,
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": 200, **settings},
            },
            output_dir=tmp_path,
        )
        return tmp_path / batch_id / "dem_local"

    @pytest.mark.parametrize(
        "hostile_id",
        [
            "http://arctos.database.museum/guid/MSB:Mamm:1",
            "urn:catalog:MO:Tropicos:100123",
            "record 12 ",
            "émile",
            "CON",
            "x" * 300,
        ],
    )
    def test_hostile_ids_export_to_portable_filenames(self, hostile_id, tmp_path):
        """An ID that used to crash or scatter tiles now writes one flat file."""
        tile_dir = self._export(self._points_with_ids([hostile_id]), tmp_path)

        # rglob rather than glob: a "/" in the ID used to create nested
        # directories, so this would still find the tile if that regressed.
        tiles = list(tile_dir.rglob("*.tif"))
        assert len(tiles) == 1, f"expected exactly one tile, found {tiles}"

        tile = tiles[0]
        assert tile.parent == tile_dir, f"tile landed in a subdirectory: {tile}"
        assert self.PORTABLE_TILE_NAME.match(tile.name), f"{tile.name!r} is not portable"
        assert len(tile.name.encode("utf-8")) <= 255

    def test_no_nested_directories_are_created(self, tmp_path):
        """The "/" case specifically: everything stays in one flat folder."""
        tile_dir = self._export(self._points_with_ids(["a/b/c", "d/e"]), tmp_path)

        subdirectories = [path for path in tile_dir.iterdir() if path.is_dir()]
        assert not subdirectories, f"unexpected nested directories: {subdirectories}"

    def test_ids_that_sanitize_alike_do_not_overwrite_each_other(self, tmp_path):
        """`a/b` and `a_b` both reduce to `a_b` — they need distinct files."""
        tile_dir = self._export(self._points_with_ids(["a/b", "a_b"]), tmp_path)
        assert len(list(tile_dir.glob("*.tif"))) == 2

    def test_ids_that_differ_only_in_case_write_two_tiles(self, tmp_path):
        """`Site1` and `site1` get hash suffixes, so both tiles exist on any file system."""
        tile_dir = self._export(self._points_with_ids(["Site1", "site1", "Site2"]), tmp_path)

        manifest = pd.read_csv(tile_dir / "tiles_manifest.csv")
        tile_filename_by_id = dict(zip(manifest["occurrenceID"], manifest["tile_filename"]))
        site1_upper = tile_filename_by_id["Site1"]
        site1_lower = tile_filename_by_id["site1"]

        assert re.fullmatch(r"Site1-[0-9a-f]{8}-dem_local\.tif", site1_upper)
        assert re.fullmatch(r"site1-[0-9a-f]{8}-dem_local\.tif", site1_lower)
        assert site1_upper.casefold() != site1_lower.casefold()
        # An ID without a case collision keeps its historical name.
        assert tile_filename_by_id["Site2"] == "Site2-dem_local.tif"
        assert manifest["exported"].all()
        assert {tile.name for tile in tile_dir.glob("*.tif")} == set(tile_filename_by_id.values())

    def test_clean_ids_keep_their_historical_filenames(self, sample_df, tmp_path):
        """Backward compatibility: unaffected inputs produce unchanged names."""
        tile_dir = self._export(sample_df, tmp_path)

        expected = {f"{sample_id}-dem_local.tif" for sample_id in sample_df["occurrenceID"]}
        assert {tile.name for tile in tile_dir.glob("*.tif")} == expected

    def test_clean_ids_keep_historical_names_with_window_suffix(self, sample_df, tmp_path):
        """The multi-window naming is likewise unchanged for clean IDs."""
        tile_dir = self._export(sample_df, tmp_path, window_size_m=[200, 400])

        assert (tile_dir / f"{sample_df['occurrenceID'][0]}-dem_local-200m.tif").exists()
        assert (tile_dir / f"{sample_df['occurrenceID'][0]}-dem_local-400m.tif").exists()


class TestSampleIdValidation:
    """Blank or duplicated IDs cost the user tiles, so raster mode rejects them."""

    def _points_with_ids(self, sample_ids):
        return pd.DataFrame(
            {
                "occurrenceID": list(sample_ids),
                "decimalLatitude": [62.98] * len(sample_ids),
                "decimalLongitude": [18.025] * len(sample_ids),
            }
        )

    def _raster_config(self):
        return {
            "batch_id": "id_check",
            "datasets": ["dem_local"],
            "settings": {"output_type": "raster", "window_size_m": 200},
        }

    def test_duplicate_ids_raise(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate values"):
            extract(
                self._points_with_ids(["dup", "dup", "unique"]),
                self._raster_config(),
                output_dir=tmp_path,
            )

    def test_blank_ids_raise(self, tmp_path):
        with pytest.raises(ValueError, match="missing or blank"):
            extract(
                self._points_with_ids(["ok", "   "]),
                self._raster_config(),
                output_dir=tmp_path,
            )

    def test_null_ids_raise(self, tmp_path):
        with pytest.raises(ValueError, match="missing or blank"):
            extract(
                self._points_with_ids(["ok", None]),
                self._raster_config(),
                output_dir=tmp_path,
            )

    def test_error_names_the_users_own_column(self, tmp_path):
        """The message must quote the caller's column name, not internal 'id'."""
        df = self._points_with_ids(["dup", "dup"]).rename(columns={"occurrenceID": "my_site_id"})
        with pytest.raises(ValueError, match="my_site_id"):
            extract(
                df,
                self._raster_config(),
                output_dir=tmp_path,
                id_column="my_site_id",
            )

    def test_validation_runs_before_any_tile_is_written(self, tmp_path):
        """Failing fast means no half-finished output folder is left behind."""
        with pytest.raises(ValueError):
            extract(
                self._points_with_ids(["dup", "dup"]),
                self._raster_config(),
                output_dir=tmp_path,
            )
        assert not (tmp_path / "id_check").exists()

    def test_list_checks_ids_before_an_earlier_tabular_output(self, tmp_path):
        """In a list, a raster config with duplicate IDs stops the call before the tabular config runs."""
        tabular_config = {
            "batch_id": "tabular_first",
            "datasets": ["dem_local"],
            "settings": {"output_type": "tabular", "window_size_m": 200, "statistics": ["mean"]},
        }

        with pytest.raises(ValueError, match="Output 'id_check': column 'occurrenceID' has dup"):
            extract(
                self._points_with_ids(["dup", "dup", "unique"]),
                [tabular_config, self._raster_config()],
                output_dir=tmp_path,
            )
        assert list(tmp_path.iterdir()) == []

    def test_tabular_mode_allows_duplicate_ids(self, tmp_path):
        """Tabular output puts IDs in a column, where duplicates are harmless."""
        extract(
            self._points_with_ids(["dup", "dup"]),
            {
                "batch_id": "tab",
                "datasets": ["dem_local"],
                "settings": {
                    "output_type": "tabular",
                    "window_size_m": 200,
                    "statistics": ["mean"],
                },
            },
            output_dir=tmp_path,
        )
        # Both rows survive to the output table — the run is not blocked.
        assert len(pd.read_csv(tmp_path / "tab.csv")) == 2


class TestTileManifest:
    """Every raster export writes a CSV mapping input rows to tile files."""

    def test_manifest_joins_back_to_the_input_table(self, sample_df, tmp_path):
        extract(
            sample_df,
            {
                "batch_id": "manifest",
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        manifest_path = tmp_path / "manifest" / "dem_local" / "tiles_manifest.csv"
        assert manifest_path.exists()

        manifest = pd.read_csv(manifest_path)
        assert list(manifest.columns) == ["occurrenceID", "tile_filename", "exported"]
        assert len(manifest) == len(sample_df)

        # The join key is the caller's own column, so a merge just works.
        merged = sample_df.merge(manifest, on="occurrenceID", how="inner")
        assert len(merged) == len(sample_df)

    def test_manifest_names_the_actual_files_on_disk(self, tmp_path):
        """The point of the manifest: recovering a sanitized filename."""
        df = pd.DataFrame(
            {
                "occurrenceID": ["http://x/guid/A:B"],
                "decimalLatitude": [62.98],
                "decimalLongitude": [18.025],
            }
        )
        extract(
            df,
            {
                "batch_id": "manifest_hostile",
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": 200},
            },
            output_dir=tmp_path,
        )

        tile_dir = tmp_path / "manifest_hostile" / "dem_local"
        manifest = pd.read_csv(tile_dir / "tiles_manifest.csv")

        assert manifest.loc[0, "occurrenceID"] == "http://x/guid/A:B"
        assert (tile_dir / manifest.loc[0, "tile_filename"]).exists()

    def test_multi_window_runs_write_one_manifest_per_window(self, sample_df, tmp_path):
        """Both manifests share a folder, so they carry the window suffix."""
        extract(
            sample_df,
            {
                "batch_id": "manifest_multi",
                "datasets": ["dem_local"],
                "settings": {"output_type": "raster", "window_size_m": [200, 400]},
            },
            output_dir=tmp_path,
        )

        tile_dir = tmp_path / "manifest_multi" / "dem_local"
        assert (tile_dir / "tiles_manifest-200m.csv").exists()
        assert (tile_dir / "tiles_manifest-400m.csv").exists()


class TestConfigPathComponents:
    """batch_id and dataset keys become folder names, so they are validated."""

    @pytest.mark.parametrize("bad_batch_id", ["my/batch", "..", "with space", "CON", "trailing."])
    def test_unsafe_batch_id_is_rejected(self, sample_df, bad_batch_id, tmp_path):
        with pytest.raises(ValueError, match="Invalid batch_id"):
            extract(
                sample_df,
                {
                    "batch_id": bad_batch_id,
                    "datasets": ["dem_local"],
                    "settings": {"output_type": "tabular", "window_size_m": 200},
                },
                output_dir=tmp_path,
            )

    def test_unsafe_dataset_name_is_rejected(self, dem_tif):
        with pytest.raises(ValueError, match="Invalid dataset name"):
            update_catalog(
                {"datasets": {"my/dataset": {"data_source": "local", "path": str(dem_tif)}}}
            )
