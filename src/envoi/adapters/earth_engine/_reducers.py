# src/envoi/adapters/earth_engine/_reducers.py
"""Reducer registry, combined-reducer assembly, and reduceRegion result parsing.

This collects everything that actually runs a server-side ``reduceRegion`` and
turns its raw output dict back into Python keys:

* ``_EE_REDUCER_MAP`` — user-facing name → (GEE factory, output suffix).
* ``_dedupe_categorical_for_ee`` — collapse class_count + class_fraction so
  GEE only runs frequencyHistogram once per call.
* ``_get_ee_reducer`` / ``_build_combined_reducer`` — assemble one combined
  reducer so multiple stats resolve in a single round-trip.
* ``_parse_reduce_result`` / ``_parse_multiband_result`` / ``_parse_point_result``
  / ``_extract_per_band_counts`` / ``_extract_count_from_reduce_result`` —
  unpack the GEE output dict back into ``{reducer_name: value}`` (or
  ``{band}_{reducer_name}: value`` for multi-band images).
* ``_circular_aspect_statistics`` — circular mean, standard deviation, and
  variance of the derived ``aspect`` band from the means of its sine and cosine.
* ``_summarize_band_coverage`` — aggregate per-point band coverage into a
  dataset-level summary for the metadata sidecar.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import ee

# Maps user-facing reducer names to GEE reducer constructors
# and the suffix GEE appends to band names in reduceRegion output.
_EE_REDUCER_MAP = {
    "mean": ("mean", "_mean"),
    "median": ("median", "_median"),
    "mode": ("mode", "_mode"),
    "std": ("stdDev", "_stdDev"),
    "var": ("variance", "_variance"),
    "min": ("min", "_min"),
    "max": ("max", "_max"),
    "count": ("count", "_count"),
    "sum": ("sum", "_sum"),
    # Categorical: frequencyHistogram returns a {class_value: count} dict
    # under the "{band}_histogram" key. The parsing helpers unpack that dict
    # into one stat key per class (e.g. class_10_count).
    # class_fraction is intentionally NOT registered here — it shares the
    # same EE reducer as class_count and is derived in Python from the
    # histogram + the per-band valid-pixel count.
    "class_count": ("frequencyHistogram", "_histogram"),
}


# Matches stat-dict keys produced by the histogram unpack step:
#   single-band: "class_10_count"
#   multi-band:  "<band>_class_10_count" (e.g. "Map_class_10_count")
# Used by the fraction post-process and by the "strip counts if the user
# didn't ask for class_count" cleanup in _fetch_stats_single.
_CLASS_COUNT_KEY_RE = re.compile(r"^(?P<prefix>.*?)class_(?P<value>-?\d+)_count$")


def _dedupe_categorical_for_ee(reducer_names: Sequence[str]) -> list[str]:
    """Collapse ``class_count`` + ``class_fraction`` to a single EE call.

    Both reducers rely on the same server-side ``ee.Reducer.frequencyHistogram``;
    duplicating it in the combined reducer would either error (duplicate
    output name) or waste a server-side pass. The function returns a list
    where ``class_fraction`` has been replaced/removed so only one
    ``class_count`` entry remains. The original list (with ``class_fraction``)
    is preserved by the caller and used later to decide whether the fraction
    keys need to be derived in Python.
    """
    deduped: list[str] = []
    histogram_already_added = False
    for reducer_name in reducer_names:
        if reducer_name in ("class_count", "class_fraction"):
            if not histogram_already_added:
                deduped.append("class_count")
                histogram_already_added = True
            # else: drop the duplicate quietly — histogram already in the call.
        else:
            deduped.append(reducer_name)
    return deduped


def _get_ee_reducer(reducer_name: str) -> tuple[ee.Reducer, str]:
    """Convert a reducer name to an (ee.Reducer, output_suffix) pair.

    Supports standard names (mean, std, …) and percentile shorthands
    in both ``q``-style (q05, q25, q90) and ``p``-style (p10, p50).
    """
    # Standard reducers
    if reducer_name in _EE_REDUCER_MAP:
        factory_name, suffix = _EE_REDUCER_MAP[reducer_name]
        reducer = getattr(ee.Reducer, factory_name)()
        return reducer, suffix

    # Percentiles: q05 / q10 / q25 / q50 / q75 / q90 / q95 or p10 / p50 …
    pct_value = None
    if (
        reducer_name.startswith("q")
        and reducer_name[1:].isdigit()
        or reducer_name.startswith("p")
        and reducer_name[1:].isdigit()
    ):
        pct_value = int(reducer_name[1:])

    if pct_value is not None and 0 < pct_value <= 100:
        reducer = ee.Reducer.percentile([pct_value]).setOutputs([reducer_name])
        return reducer, f"_{reducer_name}"

    raise ValueError(f"Unsupported reducer name: {reducer_name!r}")


def _build_combined_reducer(reducer_names: Sequence[str]) -> tuple[ee.Reducer, list[str]]:
    """Combine multiple reducers into a single ee.Reducer for one reduceRegion call.

    Returns the combined reducer and the list of GEE output suffixes
    (in the same order as *reducer_names*) needed to parse the result.
    """
    first_reducer, first_suffix = _get_ee_reducer(reducer_names[0])
    combined = first_reducer
    suffixes = [first_suffix]

    for reducer_name in reducer_names[1:]:
        next_reducer, suffix = _get_ee_reducer(reducer_name)
        combined = combined.combine(reducer2=next_reducer, sharedInputs=True)
        suffixes.append(suffix)

    return combined, suffixes


def _parse_reduce_result(
    result: dict | None,
    band_name: str,
    reducer_names: Sequence[str],
    suffixes: list[str],
) -> dict[str, float | None]:
    """Parse the output dict from reduceRegion back to {reducer_name: value}.

    GEE keys the output as ``{band}{suffix}`` — e.g. ``elevation_mean``.
    For a single reducer with no combination, GEE may omit the suffix and
    use just the band name.

    Special case: when ``reducer_name == "class_count"`` the GEE value at
    ``{band}_histogram`` is itself a ``dict[str, number]`` (class id → count).
    We expand it into one stat key per class instead of storing the dict
    under a single ``class_count`` key, so downstream column-naming code
    sees ordinary scalar entries.
    """
    result_dict: dict[str, float | None] = {}
    if not result:
        # class_count produces *zero* keys when the window had no data, so the
        # row's stat dict simply lacks any class entries (downstream will
        # zero-fill). Every other reducer gets an explicit None placeholder.
        return {r: None for r in reducer_names if r != "class_count"}

    for reducer_name, suffix in zip(reducer_names, suffixes):
        if reducer_name == "class_count":
            # Histogram lives under "{band}_histogram"; for a lone reducer GEE
            # may omit the suffix, so fall back to the bare band key as well.
            output_key = f"{band_name}{suffix}"
            histogram = result.get(output_key)
            if histogram is None and len(reducer_names) == 1:
                histogram = result.get(band_name)
            if histogram:
                # Class ids come back as JSON strings (e.g. "10"); float→int
                # tolerates legitimate float codes that round to an int.
                for class_str, count_value in histogram.items():
                    result_dict[f"class_{int(float(class_str))}_count"] = int(count_value)
            continue

        # Try with suffix first, then bare band name (single-reducer case)
        output_key = f"{band_name}{suffix}"
        value = result.get(output_key)
        if value is None and len(reducer_names) == 1:
            value = result.get(band_name)
        result_dict[reducer_name] = value

    return result_dict


def _parse_multiband_result(
    result: dict | None,
    reducer_names: Sequence[str],
    suffixes: list[str],
    *,
    count_only: bool = False,
) -> dict[str, float | None]:
    """Parse reduceRegion output for multi-band images.

    Returns a flat dict keyed as ``{band}_{reducer_name}`` for every band
    present in *result*, e.g. ``{"bio01_mean": 27.0, "bio02_mean": 180.5}``.

    The caller (``_fetch_stats_single``) combines a count reducer with the
    user's reducers for QC, unless the user asked for ``"count"``. With two
    or more reducers, Earth Engine emits ``{band}{gee_suffix}`` keys (e.g.
    ``"bio01_mean"``), and this function parses them by suffix.

    Special case: ``class_count`` returns a histogram-dict at
    ``{band}_histogram`` instead of a scalar. Each band's dict is unpacked
    into ``{band}_class_{value}_count`` keys.

    Args:
        result: The window branch of the ``reduceRegion`` output.
        reducer_names: The reducer names, aligned with ``suffixes``.
        suffixes: The Earth Engine output suffixes of the reducers.
        count_only: True when ``count`` is the only window reducer. Earth
            Engine names the output of a single reducer by the band name only
            (``{"elevation": 247, "slope": 247}``), so each key is a band and
            its value is that band's count. The result is then
            ``{band}_count`` for each band.
    """
    if not result:
        return {}

    if count_only:
        return {f"{band}_count": value for band, value in result.items()}

    result_dict: dict[str, float | None] = {}

    for reducer_name, suffix in zip(reducer_names, suffixes):
        if reducer_name == "class_count":
            # For each band's histogram entry, unpack the inner dict into
            # one stat key per class. Empty / None histograms (no data for
            # that band's window) simply produce no keys.
            for output_key, histogram in result.items():
                if not output_key.endswith(suffix):
                    continue
                band = output_key[: -len(suffix)]
                if not histogram:
                    continue
                for class_str, count_value in histogram.items():
                    result_dict[f"{band}_class_{int(float(class_str))}_count"] = int(count_value)
            continue

        # Find every key that ends with this reducer's suffix (e.g. "_mean").
        # Count keys (added by the caller for QC) are skipped here unless
        # the user actually asked for a "count" reducer themselves.
        for output_key, val in result.items():
            if output_key.endswith(suffix):
                band = output_key[: -len(suffix)]
                result_dict[f"{band}_{reducer_name}"] = val

    return result_dict


def _parse_point_result(
    result: dict | None,
    band_name: str,
    multiband: bool,
) -> dict[str, float | None]:
    """Parse a Point-geometry reduceRegion sub-result into _point-keyed stats.

    The point reduction uses ee.Reducer.first() over an ee.Geometry.Point,
    so GEE returns the bare band name(s) as keys with the single sampled
    value(s). We re-key them with a "_point" suffix to match the column-
    naming convention used by the rest of the pipeline:
      - single band  → {"point": value}
      - multi-band   → {"<band>_point": value, ...}

    Returns the stats dict; an empty/None *result* yields a None-valued
    placeholder so downstream callers see a consistent schema.
    """
    if not result:
        # Preserve schema even on empty results — single-band callers expect
        # the "point" key to exist; multi-band callers iterate the dict and
        # are robust to it being empty.
        return {} if multiband else {"point": None}

    if multiband:
        # GEE keys each band's value by the bare band name; just append "_point".
        return {f"{band}_point": value for band, value in result.items()}

    # Single-band: prefer the explicit band_name when present, otherwise
    # fall back to the first value (handles unnamed-band edge cases).
    value = result.get(band_name)
    if value is None:
        value = next(iter(result.values()), None)
    return {"point": value}


def _extract_per_band_counts(result: dict | None, *, count_only: bool = False) -> dict[str, int]:
    """Return {band_name: valid_pixel_count} for every band in a multiband result.

    The combined reducer emits a "{band}_count" entry per band; this helper
    strips the suffix and collects all of them so callers can compute
    per-band coverage or pick the worst-case band.
    With ``count_only`` (``count`` is the only window reducer), Earth Engine
    names each count by the bare band name, so every key is a band.
    Returns an empty dict when result is None or has no count entries.
    """
    if not result:
        return {}
    if count_only:
        return {band: int(val) for band, val in result.items() if val is not None}
    return {
        key[:-6]: int(val)  # strip trailing "_count" (6 chars)
        for key, val in result.items()
        if key.endswith("_count") and val is not None
    }


def _extract_count_from_reduce_result(
    result: dict | None,
    band_name: str,
    multiband: bool,
    *,
    count_only: bool = False,
) -> int:
    """Pull the count-reducer output from a reduceRegion result.

    The combined reducer produces a "{band}_count" entry per band. For
    single-band reductions we look up "{band_name}_count" directly. For
    multi-band reductions we use the minimum count across all bands —
    i.e. the worst-case band — so that coverage reflects the most
    data-sparse band rather than an arbitrarily chosen one.
    With ``count_only`` (``count`` is the only window reducer), Earth Engine
    names each count by the bare band name (``{"elevation": 247}``).
    Returns 0 when the result is missing or has no count entry.
    """
    if not result:
        return 0

    if multiband:
        # Use min so coverage is conservative: the window is "fully valid"
        # only when every band has data at every pixel.
        per_band = _extract_per_band_counts(result, count_only=count_only)
        return min(per_band.values(), default=0)

    count_key = band_name if count_only else f"{band_name}_count"
    val = result.get(count_key)
    return int(val) if val is not None else 0


# ---------------------------------------------------------------------------
# Circular statistics of the derived aspect band
# ---------------------------------------------------------------------------

# Below this mean resultant length R, the directions cancel out (for example
# a uniform spread around the circle), so no mean direction exists. Rounding
# also makes atan2 of two near-zero means meaningless.
_MIN_MEAN_RESULTANT_LENGTH = 1e-12


def _circular_aspect_statistics(
    mean_sin: float | None,
    mean_cos: float | None,
    count: float | None,
) -> dict[str, float | None]:
    """Return the circular mean, standard deviation, and variance of aspect, in degrees.

    Aspect is a direction: 350° and 10° are 20° apart, and their mean is 0°,
    not the linear mean 180°. The adapter reduces the sine and cosine of the
    aspect (in radians) of the window's non-flat pixels on the server. This
    function turns those means into circular statistics:

    * mean: ``atan2(mean_sin, mean_cos)`` in degrees, in [0, 360).
    * std: ``sqrt(-2 ln R)`` in degrees, where ``R = hypot(mean_sin, mean_cos)``
      is the mean resultant length, clamped to at most 1 (rounding can give a
      value slightly above 1, and ``ln R`` must not be positive).
    * var: the square of std, in square degrees.

    Args:
        mean_sin: Mean sine of the aspect of the non-flat pixels, or None.
        mean_cos: Mean cosine of the aspect of the non-flat pixels, or None.
        count: Number of non-flat pixels, or None when Earth Engine gave none.

    Returns:
        ``{"mean": ..., "std": ..., "var": ...}``. All three are None when the
        window has no non-flat pixel (count 0 or missing), when a mean is
        missing, or when R is below ``_MIN_MEAN_RESULTANT_LENGTH``.
    """
    empty_result: dict[str, float | None] = {"mean": None, "std": None, "var": None}
    if not count or mean_sin is None or mean_cos is None:
        return empty_result

    mean_resultant_length = min(math.hypot(mean_sin, mean_cos), 1.0)
    if mean_resultant_length < _MIN_MEAN_RESULTANT_LENGTH:
        return empty_result

    mean_direction_deg = math.degrees(math.atan2(mean_sin, mean_cos)) % 360.0
    # A tiny negative angle modulo 360 rounds to exactly 360.0 in floating
    # point. 360° is north, the same direction as 0°.
    if mean_direction_deg >= 360.0:
        mean_direction_deg = 0.0

    # ln R <= 0, so -2 ln R >= 0. abs() turns the -0.0 of R = 1 into 0.0, so
    # that the output table shows 0.0 and not -0.0.
    std_deg = math.degrees(math.sqrt(abs(-2.0 * math.log(mean_resultant_length))))
    return {"mean": mean_direction_deg, "std": std_deg, "var": std_deg**2}


# ---------------------------------------------------------------------------
# Dataset-level summary helpers  (aggregates over the full point set)
# ---------------------------------------------------------------------------


def _summarize_band_coverage(meta_list: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate per-point band_coverage_pct across all sample points.

    Returns ``{band: {"min": x, "mean": x, "max": x}}`` for each band that
    appeared in at least one point's metadata. Failure-path metas have no
    ``band_coverage_pct`` and are skipped automatically, so the
    summary only reflects points that actually ran. Returns an empty dict for
    single-band datasets or when no point produced band-level coverage data.
    """
    all_band_values: dict[str, list[float]] = {}
    for point_meta in meta_list:
        for band, coverage_pct in point_meta.get("band_coverage_pct", {}).items():
            all_band_values.setdefault(band, []).append(coverage_pct)

    per_band = {
        band: {
            "min": round(min(values), 2),
            "mean": round(sum(values) / len(values), 2),
            "max": round(max(values), 2),
        }
        for band, values in all_band_values.items()
    }

    # If every band's summary is identical, all bands share the same pixel
    # grid and the breakdown adds no information beyond coverage_pct. Return
    # empty so build_dataset_meta omits the key entirely.
    summaries = list(per_band.values())
    if summaries and all(s == summaries[0] for s in summaries[1:]):
        return {}

    return per_band
