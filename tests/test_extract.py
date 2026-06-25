"""Tests for the extract() pipeline using local raster data.

The synthetic DEM and ``sample_df`` fixtures are provided by ``conftest.py``,
so this module no longer references any on-disk fixture paths.
"""

from pathlib import Path
import json
import math
import warnings

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
        assert list(result["gbifID"]) == list(sample_df["gbifID"])

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
            [
                {
                    "gbifID": "OOB",
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
        oob_row = stats_df.loc[stats_df["gbifID"] == "OOB"].iloc[0]
        for band_index in (1, 2, 3):
            for reducer_name in ("mean", "std"):
                value = oob_row[f"multi_band_local_b{band_index}_{reducer_name}_100m"]
                assert pd.isna(
                    value
                ), f"expected NaN at b{band_index}_{reducer_name} for OOB point, got {value}"


# ------------------------------------------------------------------
# Custom input column names
# ------------------------------------------------------------------


class TestCustomColumnNames:
    """The default input column names follow the GBIF / Darwin Core convention
    (``gbifID``, ``decimalLatitude``, ``decimalLongitude``, ``eventDate``),
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
                "gbifID": "id",
                "decimalLatitude": "lat",
                "decimalLongitude": "lon",
                "eventDate": "date",
            }
        )

        outputs = extract(
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

        result = outputs["legacy_names"]
        # The output must round-trip the user's chosen names — not the new GBIF
        # defaults, and not the internal canonical names.
        assert "id" in result.columns
        assert "gbifID" not in result.columns
        assert list(result["id"]) == list(legacy_df["id"])
        # And the actual extraction still ran end-to-end.
        assert "dem_local_mean_100m" in result.columns


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
        df = pd.DataFrame(
            {"gbifID": ["centre"], "decimalLatitude": [62.98], "decimalLongitude": [18.025]}
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
            {"gbifID": ["centre"], "decimalLatitude": [62.98], "decimalLongitude": [18.025]}
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
        """output_file_format='dataframe' returns the stats DataFrame instead of a Path."""
        outputs = extract(
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

        # Sanity: the value in the outputs dict is a DataFrame, not a Path.
        # This is the whole point of the "dataframe" mode — skip disk I/O for
        # the stats output when the caller wants the result in-memory.
        result = outputs["df_test"]
        assert isinstance(
            result, pd.DataFrame
        ), f"expected DataFrame return, got {type(result).__name__}"
        assert len(result) == len(sample_df)
        assert "dem_local_mean_100m" in result.columns
        # Core columns (gbifID/decimalLatitude/decimalLongitude) are
        # preserved on the stats output so the returned DataFrame is
        # independently useful.
        assert "gbifID" in result.columns
        assert list(result["gbifID"]) == list(sample_df["gbifID"])

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
        # Standard deviation of a single repeated value is exactly 0 (the
        # sample-std denominator (n-1) cancels with the zero numerator).
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
        result = _append_stat_columns(df, "lulc", window_size_m=200, stats_results=stats_results)
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
        result = _append_stat_columns(df, "lulc", window_size_m=200, stats_results=stats_results)
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
        result = _append_stat_columns(df, "dem", window_size_m=100, stats_results=stats_results)
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
        result = _append_stat_columns(df, "lulc", window_size_m=200, stats_results=stats_results)
        assert result["lulc_b1_class_10_count_200m"].tolist() == [5, 0]
        assert result["lulc_b1_class_20_count_200m"].tolist() == [0, 3]
        # b2_class_10 only appears in row 0; row 1 is zero-filled.
        assert result["lulc_b2_class_10_count_200m"].tolist() == [4, 0]


# ------------------------------------------------------------------
# _ALL_KNOWN_REDUCERS — make sure the registry sets agree.
# ------------------------------------------------------------------


def test_all_known_reducers_includes_class_reducers():
    # Regression check: the validator's allow-list must include the new
    # categorical reducers, otherwise _validate_reducer_names rejects them
    # before the adapter ever sees them.
    from envoi._config_parsing import _ALL_KNOWN_REDUCERS

    assert "class_count" in _ALL_KNOWN_REDUCERS
    assert "class_fraction" in _ALL_KNOWN_REDUCERS
