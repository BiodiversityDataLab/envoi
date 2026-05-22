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
* ``_summarize_band_coverage`` — aggregate per-point band coverage into a
  dataset-level summary for the metadata sidecar.
"""

from __future__ import annotations

import re
from typing import Sequence

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
    if reducer_name.startswith("q") and reducer_name[1:].isdigit():
        pct_value = int(reducer_name[1:])
    elif reducer_name.startswith("p") and reducer_name[1:].isdigit():
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
) -> dict[str, float | None]:
    """Parse reduceRegion output for multi-band images.

    Returns a flat dict keyed as ``{band}_{reducer_name}`` for every band
    present in *result*, e.g. ``{"bio01_mean": 27.0, "bio02_mean": 180.5}``.

    The caller (``_fetch_stats_single``) arranges for ``{band}_count``
    keys to be present in *result* — either by combining a count reducer
    itself for QC, or by passing through the user's own ``"count"``
    request. Either way the result is a combined-form dict, so GEE emits
    ``{band}{gee_suffix}`` keys (e.g. ``"bio01_mean"``) and we always use
    suffix-based parsing here.

    Special case: ``class_count`` returns a histogram-dict at
    ``{band}_histogram`` instead of a scalar. Each band's dict is unpacked
    into ``{band}_class_{value}_count`` keys.
    """
    if not result:
        return {}

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


def _extract_per_band_counts(result: dict | None) -> dict[str, int]:
    """Return {band_name: valid_pixel_count} for every band in a multiband result.

    The combined reducer emits a "{band}_count" entry per band; this helper
    strips the suffix and collects all of them so callers can compute
    per-band coverage or pick the worst-case band.
    Returns an empty dict when result is None or has no count entries.
    """
    if not result:
        return {}
    return {
        key[:-6]: int(val)  # strip trailing "_count" (6 chars)
        for key, val in result.items()
        if key.endswith("_count") and val is not None
    }


def _extract_count_from_reduce_result(
    result: dict | None,
    band_name: str,
    multiband: bool,
) -> int:
    """Pull the count-reducer output from a reduceRegion result.

    The combined reducer produces a "{band}_count" entry per band. For
    single-band reductions we look up "{band_name}_count" directly. For
    multi-band reductions we use the minimum count across all bands —
    i.e. the worst-case band — so that coverage reflects the most
    data-sparse band rather than an arbitrarily chosen one.
    Returns 0 when the result is missing or has no count entry.
    """
    if not result:
        return 0

    if multiband:
        # Use min so coverage is conservative: the window is "fully valid"
        # only when every band has data at every pixel.
        per_band = _extract_per_band_counts(result)
        return min(per_band.values(), default=0)

    val = result.get(f"{band_name}_count")
    return int(val) if val is not None else 0


# ---------------------------------------------------------------------------
# Dataset-level summary helpers  (aggregates over the full point set)
# ---------------------------------------------------------------------------


def _summarize_band_coverage(meta_list: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate per-point band_coverage_pct across all sample points.

    Returns ``{band: {"min": x, "mean": x, "max": x}}`` for each band that
    appeared in at least one point's metadata. Failure-path metas carry an
    empty ``band_coverage_pct`` dict and are skipped automatically, so the
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
