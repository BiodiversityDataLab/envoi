"""Unit tests for envoi.adapters.earth_engine internal helpers.

These exercise the pure-Python parsing functions that translate raw
``reduceRegion`` output into stat dicts. They do not hit Earth Engine —
the inputs are hand-built dicts that mimic the shape GEE returns. As such
they do not need credentials and run on every machine, unlike the
end-to-end tests in ``test_gee_features.py``.

The class-count / class-fraction expansion is the most subtle piece, so
most cases here verify that path.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from envoi.adapters.earth_engine._image import _is_mosaic_date_policy, _resolve_date_filter_range
from envoi.adapters.earth_engine._reducers import (
    _circular_aspect_statistics,
    _dedupe_categorical_for_ee,
    _extract_count_from_reduce_result,
    _extract_per_band_counts,
    _parse_multiband_result,
    _parse_reduce_result,
)
from envoi.adapters.earth_engine._tiles import GEE_SYNC_LIMIT_BYTES, _check_tile_size

# ------------------------------------------------------------------
# _dedupe_categorical_for_ee — both class_* names map to one EE call.
# ------------------------------------------------------------------


class TestDedupeCategoricalForEe:
    def test_both_class_reducers_collapse_to_class_count(self):
        # class_count + class_fraction both rely on the same server-side
        # frequencyHistogram reducer. The deduped list contains a single
        # entry so _build_combined_reducer doesn't combine the same reducer
        # twice (which GEE rejects with "Duplicate output name").
        assert _dedupe_categorical_for_ee(["class_count", "class_fraction"]) == ["class_count"]

    def test_class_fraction_alone_becomes_class_count(self):
        # Even if the user only asks for class_fraction, EE still needs the
        # underlying frequencyHistogram (which we map under "class_count").
        # The caller is responsible for stripping the count keys back out
        # of the final stat dict.
        assert _dedupe_categorical_for_ee(["class_fraction"]) == ["class_count"]

    def test_class_count_alone_unchanged(self):
        # Single class_count request — no dedupe needed.
        assert _dedupe_categorical_for_ee(["class_count"]) == ["class_count"]

    def test_continuous_reducers_unchanged(self):
        # Reducers that have nothing to do with frequencyHistogram pass
        # through verbatim.
        assert _dedupe_categorical_for_ee(["mean", "std", "count"]) == ["mean", "std", "count"]

    def test_mixed_request_keeps_other_reducers_and_dedupes(self):
        # A realistic request — class_count + class_fraction alongside
        # mode / count. The class entries collapse, the others stay.
        assert _dedupe_categorical_for_ee(["mode", "class_count", "class_fraction", "count"]) == [
            "mode",
            "class_count",
            "count",
        ]


# ------------------------------------------------------------------
# _parse_reduce_result — single-band parsing, with histogram expansion.
# ------------------------------------------------------------------


class TestParseReduceResultClassCount:
    """class_count expands GEE's nested histogram dict into per-class keys."""

    def test_basic_single_band_histogram_expands_per_class(self):
        # frequencyHistogram returns a dict[str, number] under "{band}_histogram"
        # when combined with other reducers. We expand it into one stat key
        # per class with the form "class_{value}_count".
        result = {"Map_histogram": {"10": 5, "20": 3, "30": 1}}
        parsed = _parse_reduce_result(
            result,
            band_name="Map",
            reducer_names=["class_count"],
            suffixes=["_histogram"],
        )
        assert parsed == {"class_10_count": 5, "class_20_count": 3, "class_30_count": 1}

    def test_histogram_alongside_other_reducers(self):
        # When class_count is combined with mean, both reducers appear in
        # the GEE result. The mean produces a single scalar; the histogram
        # produces multiple class entries — and they must coexist cleanly
        # in the parsed dict.
        result = {
            "Map_histogram": {"10": 5, "20": 3},
            "Map_mean": 15.0,
        }
        parsed = _parse_reduce_result(
            result,
            band_name="Map",
            reducer_names=["class_count", "mean"],
            suffixes=["_histogram", "_mean"],
        )
        assert parsed == {"class_10_count": 5, "class_20_count": 3, "mean": 15.0}

    def test_empty_histogram_produces_no_class_keys(self):
        # A None or empty histogram (out-of-extent point) yields zero class
        # keys — downstream zero-fills those columns alongside successful
        # rows in the same batch.
        result = {"Map_histogram": None}
        parsed = _parse_reduce_result(
            result,
            band_name="Map",
            reducer_names=["class_count"],
            suffixes=["_histogram"],
        )
        assert parsed == {}

    def test_empty_result_omits_class_count_placeholder(self):
        # When the whole reduceRegion result is empty, the regular reducers
        # get a None placeholder. class_count must NOT — it would create a
        # bogus "class_count" column instead of per-class columns.
        parsed = _parse_reduce_result(
            None,
            band_name="Map",
            reducer_names=["class_count", "mean"],
            suffixes=["_histogram", "_mean"],
        )
        assert parsed == {"mean": None}
        assert "class_count" not in parsed

    def test_float_class_codes_are_int_cast(self):
        # Some GEE datasets serialise class codes as JSON floats (e.g.
        # "10.0"). We cast through float→int so the resulting column name
        # ("class_10_count") doesn't carry a stray ".0".
        result = {"Map_histogram": {"10.0": 5, "20.0": 3}}
        parsed = _parse_reduce_result(
            result,
            band_name="Map",
            reducer_names=["class_count"],
            suffixes=["_histogram"],
        )
        assert parsed == {"class_10_count": 5, "class_20_count": 3}


# ------------------------------------------------------------------
# _parse_multiband_result — multi-band parsing, with histogram expansion.
# ------------------------------------------------------------------


class TestParseMultibandResultClassCount:
    """Multi-band class_count emits a {band}_class_{v}_count key per (band, class)."""

    def test_multiband_histogram_expands_per_band_per_class(self):
        # Two bands, each with its own histogram. The parsed keys preserve
        # the band prefix so downstream column naming can keep the bands
        # distinguishable.
        result = {
            "B1_histogram": {"10": 4, "20": 2},
            "B2_histogram": {"10": 6},
        }
        parsed = _parse_multiband_result(
            result,
            reducer_names=["class_count"],
            suffixes=["_histogram"],
        )
        assert parsed == {
            "B1_class_10_count": 4,
            "B1_class_20_count": 2,
            "B2_class_10_count": 6,
        }

    def test_multiband_histogram_alongside_mean(self):
        # Mean entries (one per band) and histogram entries (one per band)
        # must coexist in the parsed dict.
        result = {
            "B1_histogram": {"10": 4, "20": 2},
            "B1_mean": 14.0,
            "B2_histogram": {"10": 6},
            "B2_mean": 10.0,
        }
        parsed = _parse_multiband_result(
            result,
            reducer_names=["class_count", "mean"],
            suffixes=["_histogram", "_mean"],
        )
        assert parsed == {
            "B1_class_10_count": 4,
            "B1_class_20_count": 2,
            "B2_class_10_count": 6,
            "B1_mean": 14.0,
            "B2_mean": 10.0,
        }

    def test_empty_band_histogram_skipped(self):
        # If a band's histogram is empty (no valid pixels for that band),
        # that band contributes zero keys — other bands still expand.
        result = {
            "B1_histogram": {"10": 4},
            "B2_histogram": None,
        }
        parsed = _parse_multiband_result(
            result,
            reducer_names=["class_count"],
            suffixes=["_histogram"],
        )
        assert parsed == {"B1_class_10_count": 4}


# ------------------------------------------------------------------
# Count-only results — "count" is the only window reducer.
# ------------------------------------------------------------------

# Key shapes from the live probe of 2026-10-09. A single reducer is named by
# the band only; a combined reducer gives "{band}_{output}" keys.
_COUNT_ONLY_ONE_BAND = {"elevation": 247}
_COUNT_ONLY_TWO_BANDS = {"elevation": 247, "slope": 239}
_MEAN_AND_COUNT_TWO_BANDS = {
    "elevation_mean": 120.5,
    "elevation_count": 280,
    "slope_mean": 4.25,
    "slope_count": 272,
}


class TestCountOnlyResults:
    """With the count-only flag, bare band keys are the counts of those bands."""

    def test_single_band_count_gives_n_pixels(self):
        """A count-only single-band result gives the band's count as the valid-pixel count."""
        assert (
            _extract_count_from_reduce_result(
                _COUNT_ONLY_ONE_BAND, "elevation", multiband=False, count_only=True
            )
            == 247
        )

    def test_two_band_count_gives_band_count_keys(self):
        """A count-only two-band result gives one {band}_count key per band."""
        parsed = _parse_multiband_result(
            _COUNT_ONLY_TWO_BANDS, reducer_names=["count"], suffixes=["_count"], count_only=True
        )
        assert parsed == {"elevation_count": 247, "slope_count": 239}

    def test_two_band_count_gives_per_band_counts_and_minimum(self):
        """Per-band counts come from the bare keys, and the QC count is the smallest one."""
        assert _extract_per_band_counts(_COUNT_ONLY_TWO_BANDS, count_only=True) == {
            "elevation": 247,
            "slope": 239,
        }
        assert (
            _extract_count_from_reduce_result(
                _COUNT_ONLY_TWO_BANDS, "elevation", multiband=True, count_only=True
            )
            == 239
        )

    def test_combined_result_without_flag_is_unchanged(self):
        """Without the flag, counts come from {band}_count keys, as for combined reducers."""
        assert _parse_multiband_result(
            _MEAN_AND_COUNT_TWO_BANDS, reducer_names=["mean"], suffixes=["_mean"]
        ) == {"elevation_mean": 120.5, "slope_mean": 4.25}
        assert _extract_per_band_counts(_MEAN_AND_COUNT_TWO_BANDS) == {
            "elevation": 280,
            "slope": 272,
        }
        assert (
            _extract_count_from_reduce_result(
                _MEAN_AND_COUNT_TWO_BANDS, "elevation", multiband=True
            )
            == 272
        )
        assert (
            _extract_count_from_reduce_result(
                {"elevation_mean": 120.5, "elevation_count": 280}, "elevation", multiband=False
            )
            == 280
        )

    def test_user_count_with_other_reducer_without_flag_is_unchanged(self):
        """A user "count" combined with "mean" keeps the suffix parsing ({band}_count keys)."""
        parsed = _parse_multiband_result(
            _MEAN_AND_COUNT_TWO_BANDS, reducer_names=["mean", "count"], suffixes=["_mean", "_count"]
        )
        assert parsed == {
            "elevation_mean": 120.5,
            "slope_mean": 4.25,
            "elevation_count": 280,
            "slope_count": 272,
        }


# ------------------------------------------------------------------
# _circular_aspect_statistics — circular mean, std and var of aspect.
# ------------------------------------------------------------------


def _mean_sin_and_cos(directions_deg: list[float]) -> tuple[float, float]:
    """Return the mean sine and mean cosine of directions in degrees (an independent calculation)."""
    radians = [math.radians(direction) for direction in directions_deg]
    mean_sin = sum(math.sin(angle) for angle in radians) / len(radians)
    mean_cos = sum(math.cos(angle) for angle in radians) / len(radians)
    return mean_sin, mean_cos


def _angular_distance_deg(first_deg: float, second_deg: float) -> float:
    """Return the shortest distance between two directions on the circle, in degrees."""
    difference = abs(first_deg - second_deg) % 360.0
    return min(difference, 360.0 - difference)


class TestCircularAspectStatistics:
    """Circular statistics of aspect from the mean sine, mean cosine, and pixel count."""

    def test_directions_around_north_average_to_north(self):
        """350° and 10° give the mean direction 0° (north), not the linear mean 180°."""
        mean_sin, mean_cos = _mean_sin_and_cos([350.0, 10.0])

        result = _circular_aspect_statistics(mean_sin, mean_cos, 2)

        assert 0.0 <= result["mean"] < 360.0
        assert _angular_distance_deg(result["mean"], 0.0) == pytest.approx(0.0, abs=1e-9)
        # R = cos(10°); std = sqrt(-2 ln R) in degrees, var = std squared.
        expected_std = math.degrees(math.sqrt(-2.0 * math.log(math.cos(math.radians(10.0)))))
        assert result["std"] == pytest.approx(expected_std)
        assert result["var"] == pytest.approx(expected_std**2)

    def test_mean_direction_in_each_quadrant(self):
        """The mean direction is in [0, 360) in every quadrant (east 90°, south 180°, west 270°)."""
        for directions, expected_mean in (
            ([80.0, 100.0], 90.0),
            ([170.0, 190.0], 180.0),
            ([260.0, 280.0], 270.0),
            ([300.0, 320.0], 310.0),
        ):
            mean_sin, mean_cos = _mean_sin_and_cos(directions)
            result = _circular_aspect_statistics(mean_sin, mean_cos, 2)
            assert result["mean"] == pytest.approx(expected_mean)

    def test_identical_directions_have_zero_spread(self):
        """Pixels that all face 45° give mean 45°, std 0, and var 0."""
        mean_sin, mean_cos = _mean_sin_and_cos([45.0, 45.0, 45.0])

        result = _circular_aspect_statistics(mean_sin, mean_cos, 3)

        assert result["mean"] == pytest.approx(45.0)
        assert result["std"] == pytest.approx(0.0, abs=1e-6)
        assert result["var"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("count", [0, None])
    def test_window_without_non_flat_pixels_gives_none(self, count):
        """A window with only flat pixels (no count) gives None for mean, std, and var."""
        # Earth Engine gives null means when every pixel is masked.
        assert _circular_aspect_statistics(None, None, count) == {
            "mean": None,
            "std": None,
            "var": None,
        }
        assert _circular_aspect_statistics(0.5, 0.5, count) == {
            "mean": None,
            "std": None,
            "var": None,
        }

    def test_uniform_spread_gives_none(self):
        """Directions spread evenly around the circle (R = 0) have no mean direction."""
        mean_sin, mean_cos = _mean_sin_and_cos([0.0, 90.0, 180.0, 270.0])

        assert _circular_aspect_statistics(mean_sin, mean_cos, 4) == {
            "mean": None,
            "std": None,
            "var": None,
        }
        assert _circular_aspect_statistics(0.0, 0.0, 4) == {"mean": None, "std": None, "var": None}

    def test_resultant_length_slightly_above_one_gives_zero_std(self):
        """Rounding that gives R slightly above 1 gives std 0 and var 0, not NaN or an error."""
        mean_sin, mean_cos = 0.6 * (1.0 + 1e-12), 0.8 * (1.0 + 1e-12)
        assert math.hypot(mean_sin, mean_cos) > 1.0

        result = _circular_aspect_statistics(mean_sin, mean_cos, 5)

        assert result["std"] == 0.0
        assert result["var"] == 0.0
        assert math.copysign(1.0, result["std"]) == 1.0  # 0.0, not -0.0
        assert result["mean"] == pytest.approx(math.degrees(math.atan2(0.6, 0.8)))

    def test_missing_mean_gives_none(self):
        """A missing mean sine or cosine gives None, also with a count."""
        assert _circular_aspect_statistics(None, 0.5, 3)["mean"] is None
        assert _circular_aspect_statistics(0.5, None, 3)["std"] is None


# ------------------------------------------------------------------
# _check_tile_size — pure-Python sync-download size guard.
# ------------------------------------------------------------------


class TestCheckTileSize:
    def test_small_request_passes(self):
        # 500 m window at 10 m resolution = 50×50 = 2500 pixels × 4 bytes ×
        # 1 band = 10 KB. Well under the 32 MB limit, so no exception.
        _check_tile_size(
            window_m=500,
            scale_m=10,
            band_count=1,
            dataset_name="dem_aster",
        )

    def test_oversized_request_raises_with_actionable_message(self):
        # 100 km window at 1 m resolution = 10^10 pixels × 4 bytes ~ 40 GB.
        # Comfortably over the limit; the message must name every knob the
        # user can turn so they don't have to read our source to fix it.
        with pytest.raises(ValueError) as excinfo:
            _check_tile_size(
                window_m=100_000,
                scale_m=1,
                band_count=1,
                dataset_name="dem_glo30",
            )
        message = str(excinfo.value)
        assert "window_size_m" in message
        assert "resample_m" in message
        assert "dem_glo30" in message
        # Sanity-check the reported limit comes from the constant, not a
        # magic number — keeps the assertion honest if we ever raise the cap.
        assert f"{GEE_SYNC_LIMIT_BYTES / 1e6:.0f} MB" in message

    def test_band_count_inflates_estimate(self):
        # A multi-band request that's under the limit at 1 band must trip
        # the guard at 20 bands. Confirms the band_count factor is actually
        # used in the estimate rather than ignored.
        kwargs = {"window_m": 3000, "scale_m": 10, "dataset_name": "multi_band"}
        # 300×300×4 = 360 KB at 1 band → safe.
        _check_tile_size(band_count=1, **kwargs)
        # × 200 bands ~ 72 MB → must raise.
        with pytest.raises(ValueError):
            _check_tile_size(band_count=200, **kwargs)

    def test_estimate_uses_the_exported_pixel_count_with_ties_rounded_up(self):
        # The check must use the tile size that the export requests
        # (geo.tile_pixel_count, ties rounded up). 2896.5 pixels per side
        # gives a 2897 × 2897 tile: 2897² × 4 bytes is just over the 32 MiB
        # limit, while 2896² × 4 bytes (Python's round(), ties to even) is
        # just under it.
        assert 2896**2 * 4 <= GEE_SYNC_LIMIT_BYTES < 2897**2 * 4
        with pytest.raises(ValueError, match="too large"):
            _check_tile_size(window_m=2896.5, scale_m=1, band_count=1, dataset_name="tie_rounding")


class TestIsMosaicDatePolicy:
    """The ``collection_date_policy: mosaic`` switch, read from a catalog entry.

    Some Earth Engine collections tile *space* rather than time: every image is
    a fixed geographic tile of one static product, and its timestamp records
    when that tile was acquired rather than a step in a time series. Copernicus
    DEM GLO-30 2024_1 is the built-in example. Reading those timestamps as a
    time axis makes "most recent image" pick a tile on the other side of the
    planet, so the policy exists to skip date handling entirely.
    """

    def test_mosaic_policy_is_detected(self):
        assert _is_mosaic_date_policy({"collection_date_policy": "mosaic"})

    def test_comparison_is_case_insensitive(self):
        """Catalog YAML is hand-written, so casing should not matter."""
        assert _is_mosaic_date_policy({"collection_date_policy": "Mosaic"})

    @pytest.mark.parametrize("policy", ["nearest", "contains"])
    def test_time_series_policies_are_not_mosaic(self, policy):
        assert not _is_mosaic_date_policy({"collection_date_policy": policy})

    def test_default_is_not_mosaic(self):
        """Omitting the key must leave the existing date behaviour untouched."""
        assert not _is_mosaic_date_policy({})

    def test_unknown_policy_is_not_mosaic(self):
        """An unrecognised value must not silently disable date selection."""
        assert not _is_mosaic_date_policy({"collection_date_policy": "whatever"})


# ------------------------------------------------------------------
# _resolve_date_filter_range — the filterDate range for "contains".
# ------------------------------------------------------------------

# Three 8-day intervals, the shape of the built-in "contains" collections.
_EIGHT_DAY_STARTS = pd.DatetimeIndex(["2021-01-01", "2021-01-09", "2021-01-17"])
_EIGHT_DAY_ENDS = pd.DatetimeIndex(["2021-01-09", "2021-01-17", "2021-01-25"])


class TestResolveDateFilterRangeContains:
    """``filterDate`` is half-open and gets whole days, so the range must
    contain the start day of the selected image, or ``.first()`` is null.
    """

    def test_end_equal_to_start_covers_the_start_day(self):
        """An instantaneous image (end == start) gives a one-day range."""
        starts = pd.DatetimeIndex(["2021-06-15"])
        ends = pd.DatetimeIndex(["2021-06-15"])
        assert _resolve_date_filter_range(pd.Timestamp("2021-06-15"), "contains", starts, ends) == (
            "2021-06-15",
            "2021-06-16",
        )

    def test_end_later_on_the_start_day_covers_the_start_day(self):
        """An image that ends on its start day gives a one-day range."""
        starts = pd.DatetimeIndex(["2021-06-15 00:00:00"])
        ends = pd.DatetimeIndex(["2021-06-15 23:59:59"])
        assert _resolve_date_filter_range(pd.Timestamp("2021-06-15"), "contains", starts, ends) == (
            "2021-06-15",
            "2021-06-16",
        )

    def test_one_day_interval_is_unchanged(self):
        """An image whose end is the next midnight keeps that end."""
        starts = pd.DatetimeIndex(["2021-06-15"])
        ends = pd.DatetimeIndex(["2021-06-16"])
        assert _resolve_date_filter_range(pd.Timestamp("2021-06-15"), "contains", starts, ends) == (
            "2021-06-15",
            "2021-06-16",
        )

    def test_next_start_on_the_same_day_without_ends(self):
        """Without end times, a next start on the same day still gives a one-day range."""
        starts = pd.DatetimeIndex(["2021-06-15 00:00:00", "2021-06-15 12:00:00"])
        assert _resolve_date_filter_range(pd.Timestamp("2021-06-15"), "contains", starts) == (
            "2021-06-15",
            "2021-06-16",
        )

    @pytest.mark.parametrize(
        ("sample_date", "expected_range"),
        [
            # In range: the interval that contains the date.
            ("2021-01-12", ("2021-01-09", "2021-01-17")),
            # On the first start.
            ("2021-01-01", ("2021-01-01", "2021-01-09")),
            # Inside the last interval.
            ("2021-01-20", ("2021-01-17", "2021-01-25")),
            # Before the first start: the first image.
            ("2020-12-01", ("2021-01-01", "2021-01-09")),
            # After the last interval: the last image.
            ("2022-06-01", ("2021-01-17", "2021-01-25")),
        ],
    )
    def test_with_end_times(self, sample_date, expected_range):
        """With end times, the range is the selected image's own interval."""
        assert (
            _resolve_date_filter_range(
                pd.Timestamp(sample_date), "contains", _EIGHT_DAY_STARTS, _EIGHT_DAY_ENDS
            )
            == expected_range
        )

    @pytest.mark.parametrize(
        ("sample_date", "expected_range"),
        [
            # In range: from the containing start to the next start.
            ("2021-01-12", ("2021-01-09", "2021-01-17")),
            ("2021-01-01", ("2021-01-01", "2021-01-09")),
            # The last image ends one day after its start.
            ("2021-01-17", ("2021-01-17", "2021-01-18")),
            ("2020-12-01", ("2021-01-01", "2021-01-09")),
            ("2022-06-01", ("2021-01-17", "2021-01-18")),
        ],
    )
    def test_without_end_times(self, sample_date, expected_range):
        """Without end times, an interval ends at the next start, or one day after the last start."""
        assert (
            _resolve_date_filter_range(pd.Timestamp(sample_date), "contains", _EIGHT_DAY_STARTS)
            == expected_range
        )
