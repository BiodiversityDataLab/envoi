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

import pandas as pd
import pytest

from envoi.adapters.earth_engine._image import _is_mosaic_date_policy, _resolve_date_filter_range
from envoi.adapters.earth_engine._reducers import (
    _dedupe_categorical_for_ee,
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
