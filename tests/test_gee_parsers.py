"""Unit tests for envoi.adapters.gee_adapter internal helpers.

These exercise the pure-Python parsing functions that translate raw
``reduceRegion`` output into stat dicts. They do not hit Earth Engine —
the inputs are hand-built dicts that mimic the shape GEE returns. As such
they do not need credentials and run on every machine, unlike the
end-to-end tests in ``test_gee_features.py``.

The class-count / class-fraction expansion is the most subtle piece, so
most cases here verify that path.
"""

from __future__ import annotations


from envoi.adapters.gee_adapter import (
    _dedupe_categorical_for_ee,
    _parse_multiband_result,
    _parse_reduce_result,
)

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
