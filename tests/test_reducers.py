"""Unit tests for envoi.reducers.

These exercise the reducer functions in isolation — no rasterio, no GEE, no
fixtures. They double as living documentation of the contract that every
adapter relies on (drop NaN/inf, return NaN on empty input, etc.).
"""

from __future__ import annotations

import math

import pytest

from envoi.reducers import (
    CATEGORICAL_ONLY_REDUCERS,
    CONTINUOUS_ONLY_REDUCERS,
    SPECIAL_REDUCERS,
    get_reducer,
    list_reducers,
    make_quantile,
    r_class_count,
    r_class_fraction,
    r_count,
    r_max,
    r_mean,
    r_median,
    r_min,
    r_mode,
    r_std,
    r_sum,
    r_var,
    validate_reducers,
)

# ------------------------------------------------------------------
# Basic reducers — known inputs, known outputs.
# ------------------------------------------------------------------


class TestBasicReducers:
    """Spot-check the numeric output of each registered reducer."""

    def test_mean(self):
        # Plain arithmetic mean of three integers.
        assert r_mean([1.0, 2.0, 3.0]) == 2.0

    def test_median_odd_length(self):
        # Odd-length arrays return the exact middle element.
        assert r_median([1.0, 2.0, 3.0]) == 2.0

    def test_median_even_length(self):
        # Even-length arrays return the mean of the two middle values.
        assert r_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_min(self):
        assert r_min([3.0, 1.0, 2.0]) == 1.0

    def test_max(self):
        assert r_max([3.0, 1.0, 2.0]) == 3.0

    def test_sum(self):
        assert r_sum([1.0, 2.0, 3.0]) == 6.0

    def test_std_is_sample_std(self):
        # ddof=1 — sample std, not population. NumPy defaults to ddof=0,
        # so this is the explicit contract that envoi reducers expose.
        # Pinning the value here catches any accidental flip to ddof=0.
        assert r_std([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_var_is_sample_var(self):
        # ddof=1 sample variance, same reasoning as r_std above.
        assert r_var([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_count_returns_int(self):
        # r_count is the only reducer that returns int (not float). Downstream
        # code relies on the int type, so the type check is part of the contract.
        result = r_count([1.0, 2.0, 3.0])
        assert result == 3
        assert isinstance(result, int)

    def test_count_excludes_non_finite_values(self):
        # NaN and +/- inf are filtered out before counting — same contract as
        # all other reducers.
        assert r_count([1.0, float("nan"), float("inf"), -float("inf"), 2.0]) == 2


# ------------------------------------------------------------------
# NaN / inf / empty-input handling — shared contract across all reducers.
# ------------------------------------------------------------------


class TestNanAndEmptyHandling:
    """Every reducer must drop non-finite values and return NaN on empty input."""

    def test_mean_drops_nan(self):
        # NaN is removed by _finite() before np.mean is called, so the result
        # is the mean of the remaining finite values.
        assert r_mean([1.0, 2.0, float("nan")]) == 1.5

    def test_mean_drops_inf(self):
        # +/- inf is also dropped — if it weren't, mean would also be inf.
        assert r_mean([1.0, 2.0, float("inf"), -float("inf")]) == 1.5

    def test_mean_empty_input_returns_nan(self):
        # Empty arrays must not raise; instead the reducer returns NaN so
        # downstream pipelines can carry the missing-data signal through.
        assert math.isnan(r_mean([]))

    def test_mean_all_nan_returns_nan(self):
        # All values filtered out — equivalent to the empty-input case.
        assert math.isnan(r_mean([float("nan"), float("nan")]))

    # parametrize the same empty-input assertion across every reducer so the
    # contract is enforced uniformly. If a new reducer is added that forgets
    # the _nan_if_empty guard, this is the test that catches it.
    @pytest.mark.parametrize(
        "reducer_callable",
        [r_mean, r_median, r_min, r_max, r_sum, r_std, r_var, r_mode],
    )
    def test_all_reducers_return_nan_on_empty(self, reducer_callable):
        # extract() relies on this: out-of-extent points produce empty
        # windows, and the reducer must return NaN rather than crash.
        assert math.isnan(reducer_callable([]))


# ------------------------------------------------------------------
# Mode — most-frequent value, with documented tie-breaking behaviour.
# ------------------------------------------------------------------


class TestMode:
    """r_mode has special semantics worth pinning down explicitly."""

    def test_mode_returns_most_frequent_value(self):
        # 2.0 appears three times, beating 1.0 and 3.0 which each appear once.
        assert r_mode([1.0, 2.0, 2.0, 2.0, 3.0]) == 2.0

    def test_mode_ties_pick_smallest(self):
        # When two values tie for most-frequent, np.argmax returns the first
        # index in the sorted-unique array, which is the smallest of the tied
        # values. This is the documented behaviour in r_mode's docstring.
        assert r_mode([1.0, 1.0, 2.0, 2.0]) == 1.0

    def test_mode_continuous_no_repeats_returns_smallest(self):
        # With no repeats every count is 1 and np.argmax returns index 0 of
        # the sorted-unique array — i.e. the minimum. This is exactly the
        # situation the typed-statistics warning is meant to flag.
        assert r_mode([3.0, 1.0, 2.0]) == 1.0


# ------------------------------------------------------------------
# Quantile factory.
# ------------------------------------------------------------------


class TestQuantileFactory:
    def test_q50_equals_median_for_odd_length(self):
        # 50th percentile of [1, 2, 3] is the middle value — same as median.
        q50 = make_quantile(0.5)
        assert q50([1.0, 2.0, 3.0]) == 2.0

    def test_q10_linear_interpolation(self):
        # NumPy's default linear interpolation: position = 0.1 * (10 - 1) = 0.9,
        # interpolated between values 1 and 2 -> 1 + 0.9 * (2 - 1) = 1.9.
        # Hard-coded here so a future change to np.percentile's default method
        # would be caught immediately.
        q10 = make_quantile(0.1)
        assert q10(list(range(1, 11))) == pytest.approx(1.9)

    def test_make_quantile_rejects_negative(self):
        # Quantile must be in [0, 1]; negative values are nonsense.
        with pytest.raises(ValueError, match="quantile must be in"):
            make_quantile(-0.1)

    def test_make_quantile_rejects_greater_than_one(self):
        # Quantile > 1 is also rejected at factory time.
        with pytest.raises(ValueError, match="quantile must be in"):
            make_quantile(1.5)

    def test_quantile_function_name_includes_percentile(self):
        # The factory renames the inner function so tracebacks identify the
        # specific quantile — e.g. "r_q25" instead of a generic "_q".
        assert make_quantile(0.25).__name__ == "r_q25"


# ------------------------------------------------------------------
# Public lookup API: get_reducer and the special-reducer guard.
# ------------------------------------------------------------------


class TestGetReducer:
    def test_lookup_returns_registered_function(self):
        # The registry maps "mean" -> r_mean; verify the actual function is
        # returned (not a copy or wrapper).
        assert get_reducer("mean") is r_mean

    def test_case_insensitive_lookup(self):
        # YAML configs can use any casing for reducer names.
        assert get_reducer("MEAN") is r_mean
        assert get_reducer("Mean") is r_mean

    def test_unknown_reducer_raises(self):
        # A missing key produces a clear error pointing to the valid set.
        with pytest.raises(ValueError, match="Unknown reducer"):
            get_reducer("not_a_reducer")

    def test_point_raises_dispatch_error(self):
        # "point" is an adapter-level reducer (samples a single pixel, no
        # window aggregation). If anyone calls get_reducer("point") it means
        # the extract dispatch took a wrong branch — guard with a clear error.
        with pytest.raises(ValueError, match="adapter-level reducer"):
            get_reducer("point")

    def test_quantile_reducer_via_registry(self):
        # Quantile factories are pre-baked into the registry under "q05" etc.
        # so users can reference them by string in configs.
        q10_callable = get_reducer("q10")
        assert q10_callable([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(1.4)


# ------------------------------------------------------------------
# validate_reducers — categorical / continuous compatibility check.
# ------------------------------------------------------------------


class TestValidateReducers:
    def test_continuous_dataset_with_continuous_reducer_passes(self):
        # No warning expected — mean/std on continuous data is the normal case.
        # validate_reducers returns None when nothing is wrong.
        assert (
            validate_reducers(["mean", "std"], data_type="continuous", dataset_name="dem") is None
        )

    def test_categorical_dataset_with_compatible_reducers_passes(self):
        # mode / count / point are well-defined for categorical (class-id) data.
        assert (
            validate_reducers(["mode", "count"], data_type="categorical", dataset_name="lulc")
            is None
        )

    def test_categorical_with_continuous_reducer_returns_warning(self):
        # mean on a categorical land-cover raster is nonsense (averaging
        # class IDs). validate_reducers should return a warning string the
        # caller can record in metadata.
        warning_message = validate_reducers(["mean"], data_type="categorical", dataset_name="lulc")
        assert warning_message is not None
        # The message should name both the data_type and the offending reducer
        # so the user can act on it without consulting the source.
        assert "categorical" in warning_message
        assert "mean" in warning_message

    def test_none_data_type_skips_check(self):
        # Local rasters often omit data_type — auto-inference happens at
        # read time. validate_reducers must be a no-op in that case.
        assert validate_reducers(["mean"], data_type=None, dataset_name="dem") is None

    def test_continuous_only_set_matches_registry(self):
        # Sanity check that every name in CONTINUOUS_ONLY_REDUCERS is also a
        # real registered reducer — catches typos in the constant.
        registry_names = set(list_reducers())
        assert CONTINUOUS_ONLY_REDUCERS <= registry_names

    def test_categorical_only_set_matches_registry(self):
        # Same sanity check for the categorical-only set — every name in
        # CATEGORICAL_ONLY_REDUCERS must be registered, otherwise the warning
        # message can reference reducers users can't actually request.
        registry_names = set(list_reducers())
        assert CATEGORICAL_ONLY_REDUCERS <= registry_names

    def test_continuous_dataset_with_categorical_only_reducer_returns_warning(self):
        # class_count on a DEM is virtually always a user mistake — every
        # floating-point pixel value would become its own "class". Warn so
        # the user notices, but don't hard-fail.
        warning_message = validate_reducers(
            ["class_count"], data_type="continuous", dataset_name="dem"
        )
        assert warning_message is not None
        # The message should name the categorical-only reducer so the user
        # can identify the offending entry without reading the source.
        assert "class_count" in warning_message
        assert "categorical" in warning_message

    def test_none_data_type_with_categorical_only_reducer_returns_warning(self):
        # Local rasters often omit data_type — we treat unknown as continuous
        # for the categorical-only warning so the user still gets the safety
        # net when class_count is asked of an untagged raster.
        warning_message = validate_reducers(
            ["class_fraction"], data_type=None, dataset_name="some_local_raster"
        )
        assert warning_message is not None
        assert "class_fraction" in warning_message


# ------------------------------------------------------------------
# Categorical reducers — dict-valued, one entry per class.
# ------------------------------------------------------------------


class TestClassReducers:
    """class_count and class_fraction return dicts mapping class -> stat.

    They're the only reducers in the registry whose return shape is non-scalar;
    adapter code is expected to detect the dict and expand it into per-class
    stat keys downstream.
    """

    def test_class_count_basic(self):
        # Three class IDs with known counts — checks the unique/return_counts pairing.
        result = r_class_count([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
        assert result == {10: 2, 20: 3, 30: 1}

    def test_class_count_returns_int_values(self):
        # Counts must be Python ints (not numpy ints) so JSON / CSV
        # serialisation downstream is unambiguous and the column dtype
        # is stable.
        result = r_class_count([10.0, 10.0])
        assert isinstance(next(iter(result.values())), int)

    def test_class_count_returns_int_keys(self):
        # Class IDs are cast to int so column names like "class_10_count"
        # don't include a trailing ".0" from the float-typed input array.
        result = r_class_count([10.0, 20.0])
        assert all(isinstance(k, int) for k in result.keys())

    def test_class_count_empty_window(self):
        # Empty input must not raise — must return an empty dict so the
        # adapter emits no class keys for the failing point.
        assert r_class_count([]) == {}

    def test_class_count_drops_non_finite(self):
        # NaN and +/-inf are filtered by _finite() before np.unique runs,
        # same contract as every other reducer.
        result = r_class_count([10.0, 10.0, float("nan"), float("inf"), 20.0])
        assert result == {10: 2, 20: 1}

    def test_class_fraction_basic(self):
        # Same array as class_count_basic — fractions must sum to 1.0.
        result = r_class_fraction([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
        assert result[10] == pytest.approx(2 / 6)
        assert result[20] == pytest.approx(3 / 6)
        assert result[30] == pytest.approx(1 / 6)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_class_fraction_single_class(self):
        # A window with one class gets the full fraction 1.0.
        result = r_class_fraction([10.0, 10.0, 10.0])
        assert result == {10: 1.0}

    def test_class_fraction_empty_window(self):
        # Mirror class_count's empty-window behaviour: empty dict, no error.
        assert r_class_fraction([]) == {}

    def test_class_count_via_registry(self):
        # The registry should expose class_count by name, so adapter code
        # can call get_reducer("class_count") and receive r_class_count.
        reducer_fn = get_reducer("class_count")
        assert reducer_fn is r_class_count

    def test_class_fraction_via_registry(self):
        # Same registry lookup contract for class_fraction.
        reducer_fn = get_reducer("class_fraction")
        assert reducer_fn is r_class_fraction


# ------------------------------------------------------------------
# Registry contents and special-reducers set.
# ------------------------------------------------------------------


class TestRegistry:
    def test_list_reducers_is_sorted(self):
        # The list_reducers contract documents a sorted result.
        names = list_reducers()
        assert names == sorted(names)

    def test_list_reducers_contains_core_set(self):
        # Spot-check that the documented core reducers are all registered.
        # If one disappears the failure points at the missing name directly.
        names = set(list_reducers())
        for core_name in ("mean", "median", "min", "max", "sum", "std", "var", "count", "mode"):
            assert core_name in names, f"core reducer {core_name!r} missing from registry"

    def test_point_is_special_not_in_registry(self):
        # "point" must live ONLY in SPECIAL_REDUCERS — if it leaked into the
        # main registry the get_reducer guard would never trigger.
        assert "point" in SPECIAL_REDUCERS
        assert "point" not in list_reducers()
