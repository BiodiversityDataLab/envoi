# src/biodata/adapters/gee_adapter.py
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Sequence

from ..config import load_defaults
from ..metadata import build_tile_crs_zones, summarize_date_info, summarize_tile_export

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import ee
import geemap


try:
    from . import register as _register
except Exception:
    _register = None

logger = logging.getLogger(__name__)

_gee_initialized = False


# ---------------------------------------------------------------------------
# GEE initialization
# ---------------------------------------------------------------------------


def _ensure_gee_init():
    """Initialize GEE once per process, skip if already active."""
    global _gee_initialized
    if _gee_initialized:
        return
    try:
        ee.Number(1).getInfo()
        _gee_initialized = True
    except Exception:
        from ..auth import init_gee

        init_gee()
        _gee_initialized = True


# ---------------------------------------------------------------------------
# Geometry helpers  (UTM zones, pixel-grid snapping)
# ---------------------------------------------------------------------------


def _get_utm_crs(lon: float, lat: float) -> str:
    """Return the EPSG code for the UTM zone covering (lon, lat)."""
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"Invalid WGS84 coordinates: ({lon}, {lat})")
    zone_number = int((lon + 180) / 6) + 1
    base_epsg = 32600 if lat >= 0 else 32700
    return f"EPSG:{base_epsg + zone_number}"


def _get_utm_zone_label(lon: float, lat: float) -> str:
    """Return the UTM zone label like "33N" or "34S" for a lon/lat point."""
    zone_number = int((lon + 180) / 6) + 1
    hemisphere = "N" if lat >= 0 else "S"
    return f"{zone_number}{hemisphere}"


def _snap_to_grid(coord: float, scale: float) -> float:
    """Snap a UTM coordinate (metres) to the nearest pixel grid multiple of scale.

    Ensures GEE export windows align with the pixel grid, preventing
    inconsistent image sizes across points.
    """
    remainder = coord % scale
    if remainder < scale / 2:
        return int(coord / scale) * scale
    return int((coord + scale) / scale) * scale


# ---------------------------------------------------------------------------
# Timestamp helpers  (collection time bounds, nearest-image lookup, date filters)
# ---------------------------------------------------------------------------


def _get_collection_time_bounds(
    collection_id: str,
) -> tuple[pd.DatetimeIndex | None, pd.DatetimeIndex | None]:
    """Fetch start/end timestamps for a GEE ImageCollection.

    Returns (start_times, end_times) as sorted DatetimeIndex objects.
    Returns (start_times, None) when end times are unavailable (e.g. user-uploaded
    assets that omit system:time_end). Returns (None, None) only on total failure.
    One getInfo() round-trip, done once per dataset during __post_init__.
    """
    try:
        image_collection = ee.ImageCollection(collection_id)
        # Fetch both arrays in a single round-trip by wrapping in ee.Dictionary.
        raw = ee.Dictionary(
            {
                "starts": image_collection.aggregate_array("system:time_start"),
                "ends": image_collection.aggregate_array("system:time_end"),
            }
        ).getInfo()
        start_times = raw["starts"]
        end_times = raw["ends"]

        # Drop None entries up front — we still need at least one valid start
        # to do anything useful, regardless of which branch we end up in.
        clean_starts = [int(t) for t in start_times if t is not None]
        if not clean_starts:
            return None, None
        clean_ends = [t for t in end_times if t is not None]

        # Branch 1: end-times are missing or partial → return a start-only
        # index. Build it from clean_starts (sorted + deduped) so it's
        # directly usable for nearest-neighbour lookups downstream.
        if len(clean_ends) != len(clean_starts):
            logger.debug(
                "system:time_end missing for some images in %s; interval-based "
                "date selection will fall back to next-start boundaries.",
                collection_id,
            )
            start_series = (
                pd.to_datetime(pd.Series(clean_starts), unit="ms", origin="unix")
                .drop_duplicates()
                .sort_values()
            )
            return pd.DatetimeIndex(start_series), None

        # Branch 2: every image has a paired (start, end). Build both indices
        # from the same de-duplicated, sorted DataFrame so they stay aligned
        # element-by-element — required for `time_ends[idx]` lookups to map
        # back to the correct start.
        paired_times = sorted(zip(clean_starts, [int(e) for e in clean_ends]))
        bounds_df = pd.DataFrame(paired_times, columns=["start", "end"])
        bounds_df = bounds_df.drop_duplicates().sort_values("start")
        start_index = pd.DatetimeIndex(pd.to_datetime(bounds_df["start"], unit="ms", origin="unix"))
        end_index = pd.DatetimeIndex(pd.to_datetime(bounds_df["end"], unit="ms", origin="unix"))
        return start_index, end_index
    except Exception as e:
        logger.warning("Failed to fetch timestamps for %s: %s", collection_id, e)
        return None, None


def _find_nearest_timestamp(
    timestamps: pd.DatetimeIndex,
    target: pd.Timestamp,
) -> tuple[pd.Timestamp, bool]:
    """Return the timestamp closest to *target*, clamping to range.

    Returns (nearest_timestamp, was_clamped).
    """
    if target <= timestamps.min():
        return timestamps.min(), target < timestamps.min()
    if target >= timestamps.max():
        return timestamps.max(), target > timestamps.max()
    idx = timestamps.get_indexer([target], method="nearest")[0]
    return timestamps[idx], False


def _resolve_date_filter_range(
    date_ts: pd.Timestamp,
    policy: str,
    timestamps: pd.DatetimeIndex | None = None,
    time_ends: pd.DatetimeIndex | None = None,
) -> tuple[str, str]:
    """Return (start, end) date strings for ImageCollection.filterDate().

    When cached timestamps are available, uses them to pin the exact image
    interval. When not (server-side fallback), broadens the window by ±1 day
    so GEE can find the image without a client-side index.

    policy="contains" selects the image whose interval contains date_ts.
    policy="nearest"  selects the image with the closest start timestamp.
    """
    fmt = "%Y-%m-%d"

    if timestamps is None:
        # No cached index — let GEE resolve server-side with a wider window.
        if policy == "contains":
            return date_ts.strftime(fmt), (date_ts + pd.DateOffset(days=1)).strftime(fmt)
        else:
            return (date_ts - pd.DateOffset(days=1)).strftime(fmt), (
                date_ts + pd.DateOffset(days=1)
            ).strftime(fmt)

    if policy == "nearest":
        nearest, _ = _find_nearest_timestamp(timestamps, date_ts)
        return nearest.strftime(fmt), (nearest + pd.DateOffset(days=1)).strftime(fmt)

    # policy == "contains": find the image interval that contains date_ts.
    # Clamp to collection boundaries when date_ts is out of range.
    if date_ts <= timestamps.min():
        idx = 0
    elif date_ts >= timestamps.max():
        idx = len(timestamps) - 1
    else:
        idx = int(timestamps.searchsorted(date_ts, side="right") - 1)
        idx = max(0, min(idx, len(timestamps) - 1))

    selected = timestamps[idx]

    # Use true interval end when available, otherwise fall back to next start.
    if time_ends is not None and len(time_ends) == len(timestamps):
        next_dt = time_ends[idx]
    elif idx + 1 < len(timestamps):
        next_dt = timestamps[idx + 1]
    else:
        next_dt = selected + pd.DateOffset(days=1)

    # filterDate(start, end) is half-open. Some collections have images with
    # system:time_end == system:time_start (instantaneous events); using the
    # raw end would produce an empty filter and .first() would return null.
    # Bump by one second whenever end <= start so the selected image is
    # always inside the range.
    if next_dt <= selected:
        next_dt = selected + pd.Timedelta(seconds=1)

    return selected.strftime(fmt), next_dt.strftime(fmt)


# ---------------------------------------------------------------------------
# Derived bands  (slope, aspect, …)
# ---------------------------------------------------------------------------

# Names that are recognised as *derived* bands. When the user passes a unified
# bands list at the call site (via extract()), names appearing in this set are
# split out and forwarded to the adapter as `derived_bands` rather than `bands`.
# Keep this in sync with the if/elif dispatch inside `_apply_derived_bands`
# below — adding a new derived-band name requires updating both places.
KNOWN_DERIVED_BANDS = frozenset({"slope", "aspect"})


def _apply_derived_bands(img, derived):
    """Compute derived bands and add them alongside the existing bands of `img`.

    `derived` may be either a single band name (e.g. "slope") or a list of names
    (e.g. ["slope", "aspect"]). Each derived band is computed from `img` and
    added to the output via `addBands()`, so the source bands are preserved.

    Raises ValueError if an unknown derived band name is given — silent fallback
    was previously a trap that produced confusing "missing output" bugs.
    """
    # Normalize to a list so callers can pass either form.
    if isinstance(derived, str):
        derived_names = [derived]
    else:
        derived_names = list(derived)

    for name in derived_names:
        if name == "slope":
            img = img.addBands(ee.Terrain.slope(img))
        elif name == "aspect":
            img = img.addBands(ee.Terrain.aspect(img))
        else:
            raise ValueError(f"Unknown derived band '{name}'. Supported: 'slope', 'aspect'.")

    return img


# ---------------------------------------------------------------------------
# Image building  (load → filter → date select → cloud mask → reduce → bands)
# ---------------------------------------------------------------------------


def _build_image(
    dataset_spec: dict,
    date=None,
    geometry=None,
    collection_timestamps: pd.DatetimeIndex | None = None,
    collection_time_ends: pd.DatetimeIndex | None = None,
    *,
    lat: float | None = None,
    lon: float | None = None,
):
    """Build an ee.Image from a dataset_spec config dict.

    Pipeline: load image/collection → bounds filter → date selection →
              cloud_pct filter → cloud mask → reduce → band select →
              derived band.

    For collections, the date handling strategy is:
    - date provided + timestamps cached: find nearest timestamp, filterDate, .first()
    - date provided + no timestamps (fallback): filterDate ±1 day, .first()
    - no date: mosaic (most recent non-masked pixel per position)

    Parameters
    ----------
    geometry : ee.Geometry, optional
        Point or region to spatially constrain the collection via
        ``filterBounds``.  Essential for large tiled collections
        (e.g. satellite embeddings with 97k global tiles).
    collection_timestamps : pd.DatetimeIndex, optional
        Cached start timestamps for the collection.
    collection_time_ends : pd.DatetimeIndex, optional
        Cached end timestamps for the collection (aligned with start times).
    """
    img = None

    if "image" in dataset_spec:
        img = ee.Image(dataset_spec["image"])

    elif "collection" in dataset_spec:
        image_collection = ee.ImageCollection(dataset_spec["collection"])

        if geometry is not None:
            image_collection = image_collection.filterBounds(geometry)

        # Some tiled collections (e.g. satellite embeddings) need UTM-zone
        # filtering to avoid selecting the wrong tile for a point.
        if dataset_spec.get("use_utm_zone") and lat is not None and lon is not None:
            utm_zone = _get_utm_zone_label(lon, lat)
            image_collection = image_collection.filter(ee.Filter.eq("UTM_ZONE", utm_zone))

        # --- Date-based image selection ---
        date_policy = str(dataset_spec.get("collection_date_policy", "nearest")).lower()
        if date is not None:
            date_ts = pd.to_datetime(date)
            start, end = _resolve_date_filter_range(
                date_ts,
                date_policy,
                timestamps=collection_timestamps,
                time_ends=collection_time_ends,
            )
            image_collection = image_collection.filterDate(start, end)
            img = image_collection.first()
        else:
            # No date: mosaic (most recent non-masked pixel per position)
            img = image_collection.mosaic()

    else:
        raise ValueError("dataset_spec must contain 'image' or 'collection'")

    # Compute any derived bands FIRST, before filtering to `bands`. This matters
    # for derivations that need access to source bands the user didn't list
    # (e.g. if the user wanted only the derived output, they shouldn't have to
    # also list the intermediate source bands). For slope/aspect this isn't
    # strictly necessary since they read the first band, but keeping the order
    # consistent means future derivations can freely reference source bands.
    derived = dataset_spec.get("derived_bands")
    if derived:
        img = _apply_derived_bands(img, derived)

    # Build the final output band list. If the user specified both `bands` and
    # `derived_bands`, the result is their concatenation so source bands are
    # included alongside derived ones (e.g. DEM + slope + aspect). If only
    # `derived_bands` is set, keep whatever source bands the image already has
    # plus the derived ones. If neither is set, keep the image as-is.
    bands = dataset_spec.get("bands")
    if bands is not None:
        source_bands = bands if isinstance(bands, list) else [bands]
        derived_bands_list = (
            [derived] if isinstance(derived, str) else list(derived) if derived else []
        )
        img = img.select(source_bands + derived_bands_list)

    return img


# ---------------------------------------------------------------------------
# EE reducer helpers  (combined reducer pattern + result parsing)
# ---------------------------------------------------------------------------

# Maps EDDP/user-facing reducer names to GEE reducer constructors
# and the suffix GEE appends to band names in reduceRegion output.
_GEE_REDUCER_MAP = {
    "mean": ("mean", "_mean"),
    "median": ("median", "_median"),
    "mode": ("mode", "_mode"),
    "std": ("stdDev", "_stdDev"),
    "var": ("variance", "_variance"),
    "min": ("min", "_min"),
    "max": ("max", "_max"),
    "count": ("count", "_count"),
    "sum": ("sum", "_sum"),
}


def _get_ee_reducer(name: str) -> tuple[ee.Reducer, str]:
    """Convert an EDDP reducer name to an (ee.Reducer, output_suffix) pair.

    Supports standard names (mean, std, …) and percentile shorthands
    in both ``q``-style (q05, q25, q90) and ``p``-style (p10, p50).
    """
    # Standard reducers
    if name in _GEE_REDUCER_MAP:
        factory_name, suffix = _GEE_REDUCER_MAP[name]
        reducer = getattr(ee.Reducer, factory_name)()
        return reducer, suffix

    # Percentiles: q05 / q10 / q25 / q50 / q75 / q90 / q95 or p10 / p50 …
    pct_value = None
    if name.startswith("q") and name[1:].isdigit():
        pct_value = int(name[1:])
    elif name.startswith("p") and name[1:].isdigit():
        pct_value = int(name[1:])

    if pct_value is not None and 0 < pct_value <= 100:
        reducer = ee.Reducer.percentile([pct_value]).setOutputs([name])
        return reducer, f"_{name}"

    raise ValueError(f"Unsupported reducer name: {name!r}")


def _build_combined_reducer(reducer_names: Sequence[str]) -> tuple[ee.Reducer, list[str]]:
    """Combine multiple reducers into a single ee.Reducer for one reduceRegion call.

    Returns the combined reducer and the list of GEE output suffixes
    (in the same order as *reducer_names*) needed to parse the result.
    """
    first_reducer, first_suffix = _get_ee_reducer(reducer_names[0])
    combined = first_reducer
    suffixes = [first_suffix]

    for name in reducer_names[1:]:
        next_reducer, suffix = _get_ee_reducer(name)
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
    """
    out: dict[str, float | None] = {}
    if not result:
        return {r: None for r in reducer_names}

    for rname, suffix in zip(reducer_names, suffixes):
        # Try with suffix first, then bare band name (single-reducer case)
        key = f"{band_name}{suffix}"
        val = result.get(key)
        if val is None and len(reducer_names) == 1:
            val = result.get(band_name)
        out[rname] = val

    return out


def _parse_multiband_result(
    result: dict | None,
    reducer_names: Sequence[str],
    suffixes: list[str],
) -> dict[str, float | None]:
    """Parse reduceRegion output for multi-band images.

    Returns a flat dict keyed as ``{band}_{reducer_name}`` for every band
    present in *result*, e.g. ``{"bio01_mean": 27.0, "bio02_mean": 180.5}``.

    The caller (``_fetch_stats_single``) always combines a count reducer
    onto the user-requested reducers so we get a valid-pixel count for QC.
    Because the result is therefore always a combined-form dict, GEE
    always emits ``{band}{gee_suffix}`` keys (e.g. ``"bio01_mean"``),
    so we always use suffix-based parsing here.
    """
    if not result:
        return {}

    out: dict[str, float | None] = {}

    for rname, gee_suffix in zip(reducer_names, suffixes):
        # Find every key that ends with this reducer's suffix (e.g. "_mean").
        # Count keys (added by the caller for QC) are skipped here unless
        # the user actually asked for a "count" reducer themselves.
        for key, val in result.items():
            if key.endswith(gee_suffix):
                band = key[: -len(gee_suffix)]
                out[f"{band}_{rname}"] = val

    return out


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
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class GeeRasterAdapter:
    """Adapter that samples data directly from Google Earth Engine.

    Two extraction modes:

    1. **fetch_stats_batch** — server-side statistics via ``reduceRegion``
       with combined reducers.  Much faster for the common stats use-case.
       When ``"point"`` is included in the requested reducers, an exact-pixel
       sample at each ``(lat, lon)`` is added as a second server-side branch
       and resolved in the same round-trip.
    2. **export_tiles** — download full GeoTIFF tiles via ``geemap``.

    For ImageCollections, the adapter automatically selects the nearest
    image to each point's date. When no date is provided, the most
    recent image is used.
    """

    spec: Dict[str, Any]
    _static_image: Any = field(default=None, init=False, repr=False)
    _needs_per_point_date: bool = field(default=False, init=False, repr=False)
    _native_proj: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def __post_init__(self):
        _ensure_gee_init()

        # dataset_spec holds extra config (bands, date windows, derivatives, etc.)
        # Always auto-detect asset type from GEE, then merge with user config.
        dataset_spec = dict(self.spec.get("dataset_spec") or {})
        if (
            self.spec.get("path")
            and "image" not in dataset_spec
            and "collection" not in dataset_spec
        ):
            asset_id = self.spec["path"]
            try:
                asset_info = ee.data.getAsset(asset_id)
            except Exception as e:
                raise ValueError(
                    f"GEE asset not found: '{asset_id}'.\n"
                    f"Check the path in your catalog and that your service account has access to it.\n"
                    f"Original error: {e}"
                ) from e
            asset_type = asset_info.get("type")
            if asset_type == "IMAGE_COLLECTION":
                dataset_spec["collection"] = asset_id
            else:
                dataset_spec["image"] = asset_id
            logger.debug("Auto-detected %s as %s", asset_id, asset_type)
        if self.spec.get("bands") and "bands" not in dataset_spec:
            dataset_spec["bands"] = self.spec["bands"]
        if self.spec.get("derived_bands") and "derived_bands" not in dataset_spec:
            dataset_spec["derived_bands"] = self.spec["derived_bands"]

        # Native scale resolution priority:
        #   1. catalog override `native_scale_m` under dataset_spec (set explicitly
        #      when GEE auto-detection is unreliable, e.g. for composite collections
        #      stored in EPSG:4326)
        #   2. None → fall back to GEE's projection().nominalScale() at
        #      query time (see _get_scale / _get_scale_value)
        self.scale = dataset_spec.get("native_scale_m")
        self.crs = self.spec.get("crs", "EPSG:4326")
        self.max_workers = self.spec.get("max_workers", load_defaults()["max_workers"])
        self._dataset_spec = dataset_spec

        is_collection = "collection" in dataset_spec
        self._collection_timestamps = None
        self._collection_time_ends = None

        # Cache the native projection from the source — composite images
        # (mosaic, mean, etc.) lose per-band projection info, so we grab it
        # from the first image of a collection or the image's first band.
        if is_collection:
            self._native_proj = (
                ee.ImageCollection(dataset_spec["collection"]).first().select(0).projection()
            )
            # Fetch available timestamps for automatic date selection.
            start_times, end_times = _get_collection_time_bounds(dataset_spec["collection"])
            self._collection_timestamps = start_times
            self._collection_time_ends = end_times
            # Collections with timestamps use per-point date selection;
            # the static image is built lazily in _get_image when no date
            # is provided (most-recent fallback).
            self._needs_per_point_date = self._collection_timestamps is not None
            if not self._needs_per_point_date:
                # Timestamp fetch failed — fall back to static mosaic
                self._static_image = _build_image(dataset_spec)
        elif "image" in dataset_spec:
            self._native_proj = ee.Image(dataset_spec["image"]).select(0).projection()
            self._static_image = _build_image(dataset_spec)
            self._needs_per_point_date = False

        # If the user didn't supply a native_scale_m override, sanity-check
        # that GEE's auto-detected scale isn't the EPSG:4326 default. Some
        # composite collections (Landsat composites, Dynamic World, …) lose
        # their native projection metadata and fall back to WGS84, in which
        # case nominalScale() returns ~111319 m (one degree at the equator).
        # We catch that here so the user gets a clear error at run start
        # rather than silently producing 100 km tiles.
        if self.scale is None and self._native_proj is not None:
            self._validate_auto_detected_scale()

    def _validate_auto_detected_scale(self) -> None:
        """Raise if GEE auto-detection returned the EPSG:4326 default scale.

        Reads the projection's CRS and nominal scale once via getInfo() and
        caches the resolved scale so downstream calls don't pay the cost again.
        """
        # Fetch projection CRS and nominal scale in a single round-trip by
        # wrapping both server-side expressions in one ee.Dictionary.
        try:
            raw = ee.Dictionary(
                {
                    "proj_info": self._native_proj,
                    "scale": self._native_proj.nominalScale(),
                }
            ).getInfo()
            proj_info = raw.get("proj_info") or {}
            native_scale = float(raw["scale"])
        except Exception:
            # If GEE refuses to evaluate the projection, leave detection to
            # the first sampling call rather than masking the underlying error.
            return

        # Cache the resolved scale so _get_scale_value() doesn't re-fetch it.
        self._cached_native_scale = native_scale

        crs = (proj_info or {}).get("crs", "")
        # The GEE default is exactly 111319.49079327357 m, but allow a small
        # tolerance to also catch close-but-not-identical values (e.g. when
        # the asset reports a slightly different default-equivalent scale).
        looks_like_default = crs.upper() == "EPSG:4326" and abs(native_scale - 111319.49) < 1.0

        if looks_like_default:
            asset_id = self._dataset_spec.get(
                "collection", self._dataset_spec.get("image", "<unknown>")
            )
            raise ValueError(
                f"Dataset '{asset_id}': GEE returned the EPSG:4326 default scale "
                f"(~{native_scale:.0f} m), which means the asset's native projection "
                f"metadata is unavailable. Add `native_scale_m: <true_resolution>` "
                f"under `dataset_spec` in the catalog entry to fix this "
                f"(e.g. `native_scale_m: 30` for Landsat composites)."
            )

    # ------------------------------------------------------------------
    # Image resolution  (which ee.Image to sample, plus its band/scale)
    # ------------------------------------------------------------------

    def _get_image(self, date=None, lat=None, lon=None) -> ee.Image:
        """Return the ee.Image to sample, building per-point if needed.

        For IMAGE assets the pre-built static image is always returned.
        For ImageCollections:
        - date provided → select nearest image to that date
        - no date + point coords → select most recent image for that point
        - no date + no coords → use most recent image (cached)
        """
        # Per-point branch has priority for collections that need spatial
        # filtering (e.g. tiled DEM collections). When coordinates are given,
        # always build a fresh per-point image — never reuse `_static_image`,
        # which may have been built earlier for a no-coords call and would
        # therefore point at an arbitrary global tile that doesn't cover the
        # current sample. Reusing it caused `img.sample(point)` to return
        # empty props, so the "point" reducer silently dropped its columns
        # for tiled IMAGE_COLLECTIONs (issue with dem_glo30 + "point" stat).
        if self._needs_per_point_date and lat is not None and lon is not None:
            target_date = (
                self._collection_timestamps.max() if date is None else pd.to_datetime(date)
            )
            geom = ee.Geometry.Point([lon, lat])
            return _build_image(
                self._dataset_spec,
                date=target_date,
                geometry=geom,
                collection_timestamps=self._collection_timestamps,
                collection_time_ends=self._collection_time_ends,
                lat=lat,
                lon=lon,
            )

        # IMAGE assets or already-cached no-coords fallback.
        # Invariant: for IMAGE_COLLECTIONs, _static_image is only ever set by
        # the most-recent-fallback branch below — so reusing it here for a
        # no-coords call returns the same most-recent global image, which is
        # safe for band-name probing and global tile-export, but never for
        # per-point sampling (the per-point branch above runs first).
        if date is None and self._static_image is not None:
            return self._static_image

        # Collection with no date and no coords: build (and cache) a global
        # most-recent fallback. Used by tile-export and band-name probing
        # paths that don't have a specific point in mind.
        if date is None and self._needs_per_point_date:
            most_recent = self._collection_timestamps.max()
            logger.info(
                "No date/geometry provided for collection %s; using most recent image (%s).",
                self._dataset_spec.get("collection"),
                most_recent.strftime("%Y-%m-%d"),
            )
            self._static_image = _build_image(
                self._dataset_spec,
                date=most_recent,
                collection_timestamps=self._collection_timestamps,
                collection_time_ends=self._collection_time_ends,
            )
            return self._static_image

        # Per-point date selection without per-point bounds requirement
        # (e.g. an IMAGE_COLLECTION whose timestamps fetch failed but a
        # date is still provided).
        dt = pd.to_datetime(date)
        geom = ee.Geometry.Point([lon, lat]) if lat is not None else None
        return _build_image(
            self._dataset_spec,
            dt,
            geometry=geom,
            collection_timestamps=self._collection_timestamps,
            collection_time_ends=self._collection_time_ends,
            lat=lat,
            lon=lon,
        )

    def _get_band_name(self, img: ee.Image) -> str:
        """Get the first band name from the image (needed for result parsing)."""
        band = self._dataset_spec.get("bands")
        derived = self._dataset_spec.get("derived_bands")

        # Normalize both settings to lists so we can handle the combined case.
        source_bands = (
            [band] if isinstance(band, str) else list(band) if isinstance(band, list) else []
        )
        derived_bands_list = (
            [derived] if isinstance(derived, str) else list(derived) if derived else []
        )
        combined = source_bands + derived_bands_list

        if combined:
            # Populate band cache from the config — no GEE API call needed.
            if not hasattr(self, "_cached_band_name"):
                self._cached_band_name = combined[0]
                self._cached_band_names = combined
                self._cached_band_count = len(combined)
            return self._cached_band_name
        # Fallback: ask GEE (costs one getInfo call, cached after first use)
        if not hasattr(self, "_cached_band_name"):
            try:
                names = img.bandNames().getInfo()
                self._cached_band_name = names[0] if names else "value"
                self._cached_band_names = names  # full list for metadata
                self._cached_band_count = len(names)
            except Exception:
                self._cached_band_name = "value"
                self._cached_band_names = []
                self._cached_band_count = 1
        return self._cached_band_name

    def _get_scale(self, img):
        """Return user-specified scale or fall back to the image's native scale."""
        if self.scale is not None:
            return self.scale
        if self._native_proj is not None:
            return self._native_proj.nominalScale()
        return img.select(0).projection().nominalScale()

    def _get_scale_value(self, img) -> float:
        """Return scale as a plain Python float (metres). Fetches from GEE once and caches."""
        if self.scale is not None:
            return float(self.scale)
        if not hasattr(self, "_cached_native_scale"):
            proj = self._native_proj or img.select(0).projection()
            self._cached_native_scale = float(proj.nominalScale().getInfo())
        return self._cached_native_scale

    # ------------------------------------------------------------------
    # Region & labels
    # ------------------------------------------------------------------

    def _make_region(self, lat: float, lon: float, window_m: int) -> ee.Geometry:
        """Build a meter-accurate square region using the point's UTM zone.

        ``window_m == 0`` returns the bare point geometry, which yields a
        single-pixel reduction at (lat, lon). Negative values are rejected
        so a bad config (e.g. accidental sign flip in a YAML) surfaces as a
        clear error rather than silently degrading to point-only output.
        """
        if window_m < 0:
            raise ValueError(f"window_m must be >= 0, got {window_m}")
        point = ee.Geometry.Point([lon, lat])
        if window_m == 0:
            return point
        utm = _get_utm_crs(lon, lat)
        return point.buffer(window_m / 2, proj=ee.Projection(utm)).bounds(maxError=1)

    def _src_label(self) -> str:
        dataset_config = self._dataset_spec
        return f"gee://{dataset_config.get('image', dataset_config.get('collection', 'unknown'))}"

    # ------------------------------------------------------------------
    # Date metadata  (per-point image timestamps for the meta sidecar)
    # ------------------------------------------------------------------

    def _make_date_info(
        self,
        image_start: pd.Timestamp,
        *,
        end_index: int | None,
        clamped: bool,
        source: str,
    ) -> dict:
        """Build the date metadata dict for one resolved image timestamp.

        ``end_index`` is the position of the matching end-time inside
        ``_collection_time_ends``; pass ``None`` when no end-time should be
        emitted. The ``image_time_end`` key is included only when the index
        is valid for the cached end-times array.
        """
        # Always include the start-of-image timestamp; this is the only
        # mandatory field.
        info = {
            "date_clamped": clamped,
            "date_source": source,
            "image_time_start": image_start.strftime("%Y-%m-%d"),
        }
        # Only emit image_time_end when we have a cached end-times array
        # AND the requested index is in-bounds. Some collections expose
        # mismatched-length start/end arrays, so the bounds check matters.
        end_times = self._collection_time_ends
        if end_times is not None and end_index is not None and 0 <= end_index < len(end_times):
            info["image_time_end"] = end_times[end_index].strftime("%Y-%m-%d")
        return info

    def _resolve_date_info(self, date=None) -> dict:
        """Compute date metadata for a single point fetch.

        Returns a dict with image_time_start, image_time_end, date_clamped,
        date_source to be merged into the per-point meta dict.
        """
        if self._collection_timestamps is None:
            return {}

        timestamps = self._collection_timestamps
        last_index = len(timestamps) - 1

        if date is None:
            # No-date fallback: most recent image (last entry in the
            # timestamps array).
            return self._make_date_info(
                timestamps.max(),
                end_index=last_index,
                clamped=False,
                source="most_recent_no_date",
            )

        dt = pd.to_datetime(date)
        policy = str(self._dataset_spec.get("collection_date_policy", "nearest")).lower()

        if policy == "contains":
            # "contains" policy: pick the image whose time interval contains
            # the sample date. When the date falls outside the collection's
            # full range, clamp to the nearest boundary image.
            if dt <= timestamps.min():
                return self._make_date_info(
                    timestamps.min(),
                    end_index=0,
                    clamped=True,
                    source="clamped_to_nearest",
                )
            if dt >= timestamps.max():
                return self._make_date_info(
                    timestamps.max(),
                    end_index=last_index,
                    clamped=True,
                    source="clamped_to_nearest",
                )

            # In-range: find the latest start-time that's still <= dt.
            idx = int(timestamps.searchsorted(dt, side="right") - 1)
            idx = max(0, min(idx, last_index))
            return self._make_date_info(
                timestamps[idx],
                end_index=idx,
                clamped=False,
                source="contains_sample_date",
            )

        # Default "nearest" policy: pick whichever image timestamp is closest
        # to the sample date, even if the date is outside the range.
        nearest, was_clamped = _find_nearest_timestamp(timestamps, dt)
        nearest_index = int(timestamps.get_indexer([nearest])[0])
        return self._make_date_info(
            nearest,
            end_index=nearest_index,
            clamped=was_clamped,
            source="clamped_to_nearest" if was_clamped else "nearest_to_sample",
        )

    # ------------------------------------------------------------------
    # Empty-result builders  (failure-path skeletons matching success shape)
    # ------------------------------------------------------------------

    def _empty_result(self, window_m: int):
        vals = np.array([])
        meta = {
            "in_extent": False,
            "n_pixels": 0,
            "had_nodata": False,
            "coverage_pct": 0.0,
            "window_m": int(window_m),
            "raster_crs": self.crs,
            "transform": None,
            "dtype": None,
            "nodata": None,
            "src_path": self._src_label(),
            "window_arr": None,
        }
        return vals, meta

    def _empty_stats_result(
        self,
        window_m: int,
        reducer_names: Sequence[str],
        *,
        want_point: bool = False,
    ):
        """Return a (stats, meta) tuple matching the merged success schema.

        Used when a fetch fails so callers see the same dict shape they'd
        get on a successful call. Includes a "point" / "{band}_point" key
        when the original request asked for the point reducer, regardless
        of which window reducers were combined alongside it.
        """
        # Window-side: one None value per requested non-point reducer.
        stats: dict[str, float | None] = {r: None for r in reducer_names}

        # Point-side: mirror the success-path shape — multi-band emits one
        # "{band}_point" key per band; single-band emits a bare "point" key.
        if want_point:
            spec_band = self._dataset_spec.get("bands")
            band_count = getattr(self, "_cached_band_count", 1)
            multiband = not isinstance(spec_band, str) and band_count > 1
            if multiband:
                # We may not have the full band list cached here; emit one
                # placeholder per cached band name, falling back to a single
                # "point" key when band names aren't known yet.
                band_names = getattr(self, "_cached_band_names", None) or []
                if band_names:
                    for band in band_names:
                        stats[f"{band}_point"] = None
                else:
                    stats["point"] = None
            else:
                stats["point"] = None

        _, meta = self._empty_result(window_m)
        # Emit an empty band_coverage_pct for multiband failure paths so
        # callers always see a consistent schema regardless of success/failure.
        spec_band = self._dataset_spec.get("bands")
        band_count = getattr(self, "_cached_band_count", 1)
        if not isinstance(spec_band, str) and band_count > 1:
            meta["band_coverage_pct"] = {}
        return stats, meta

    # ------------------------------------------------------------------
    # Mode 1: Server-side stats  (reduceRegion with combined reducers)
    # ------------------------------------------------------------------

    def _fetch_stats_single(
        self,
        lat: float,
        lon: float,
        window_m: int,
        combined_reducer: "ee.Reducer | None",
        reducer_names: Sequence[str],
        suffixes: list[str],
        *,
        want_point: bool = False,
        date=None,
    ):
        """Compute server-side stats for a single point via reduceRegion.

        Two reductions are issued together as a single ``ee.Dictionary``
        and resolved with one ``getInfo()`` round-trip:

        - **window** — runs only when *combined_reducer* is provided. The
          user's reducers are combined with ``ee.Reducer.count()`` for QC,
          and ``reduceRegion`` is applied over the buffered square region.
        - **point** — runs only when *want_point* is True. ``Reducer.first()``
          is applied over an ``ee.Geometry.Point`` at the exact (lat, lon)
          to fetch the value of the pixel containing the point.

        When both branches run, GEE evaluates them in parallel server-side
        so callers pay one round-trip for both. Returns a single
        ``(stats_dict, meta_dict)`` tuple with merged keys:
        - window keys: ``{reducer}`` (single band) or ``{band}_{reducer}`` (multi-band)
        - point  keys: ``"point"``  (single band) or ``{band}_point``     (multi-band)
        """
        img = self._get_image(date, lat=lat, lon=lon)
        region = self._make_region(lat, lon, window_m)

        utm_crs = _get_utm_crs(lon, lat)
        native_m = self._get_scale_value(img)

        # If the window is smaller than the native pixel size, expand the
        # region so at least one pixel center falls inside it. Only affects
        # the window branch — the point branch always uses the exact point.
        # Track the *effective* window size so the coverage calculation
        # below uses the size GEE actually reduced over, not the user's
        # original window_m. Otherwise total_cells would be derived from
        # window_m (=1 cell here) while valid_count reflects the larger
        # expanded region (~4 cells), inflating coverage past 100%.
        effective_window_m = window_m
        if window_m < native_m:
            effective_window_m = int(native_m * 2)
            region = self._make_region(lat, lon, effective_window_m)

        # Resolve band name (also populates _cached_band_count on first call)
        band_name = self._get_band_name(img)
        band_count = getattr(self, "_cached_band_count", 1)

        # Use multi-band mode when: no band specified (auto-detect all bands), OR
        # a list of bands was specified. A single named band keeps simple {reducer} naming.
        spec_band = self._dataset_spec.get("bands")
        multiband = not isinstance(spec_band, str) and band_count > 1

        if multiband:
            img_to_reduce = img
        else:
            img_to_reduce = img.select(band_name)

        # Build up to two server-side reductions into a single ee.Dictionary.
        # GEE evaluates them in parallel and we pay one HTTP round-trip total.
        branches: dict[str, "ee.Dictionary"] = {}

        if combined_reducer is not None:
            # Window branch: user reducers + count for QC. The combined reducer
            # produces `{band}{suffix}` keys including a `{band}_count` entry.
            # Skip adding the QC count when the user already requested "count"
            # — combining a second count reducer with the same output name
            # causes GEE to raise "Duplicate output name: 'count'".
            if "count" not in reducer_names:
                full_reducer = combined_reducer.combine(
                    reducer2=ee.Reducer.count(), sharedInputs=True
                )
            else:
                full_reducer = combined_reducer
            branches["window"] = img_to_reduce.reduceRegion(
                reducer=full_reducer,
                geometry=region,
                scale=native_m,
                crs=utm_crs,
                bestEffort=True,
            )

        if want_point:
            # Point branch: exact pixel value at lat/lon. One pixel only —
            # no need for bestEffort. Output keys are bare band names.
            branches["point"] = img_to_reduce.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=ee.Geometry.Point([lon, lat]),
                scale=native_m,
                crs=utm_crs,
            )

        # Single round-trip — both branches resolved server-side.
        full_result = ee.Dictionary(branches).getInfo() or {}

        # ---- parse window branch ----
        window_stats: dict[str, float | None] = {}
        valid_count = 0
        per_band_counts: dict[str, int] = {}
        if combined_reducer is not None:
            window_raw = full_result.get("window") or {}
            if multiband:
                window_stats = _parse_multiband_result(window_raw, reducer_names, suffixes)
                # Collect per-band counts for individual coverage reporting.
                per_band_counts = _extract_per_band_counts(window_raw)
            else:
                window_stats = _parse_reduce_result(window_raw, band_name, reducer_names, suffixes)
            valid_count = _extract_count_from_reduce_result(window_raw, band_name, multiband)

        # ---- parse point branch ----
        point_stats: dict[str, float | None] = {}
        if want_point:
            point_raw = full_result.get("point") or {}
            point_stats = _parse_point_result(point_raw, band_name, multiband)

        # Merged stats dict — window keys first, then point keys.
        stats = {**window_stats, **point_stats}

        # ---- QC meta ----
        # When the window branch ran we have a real valid-pixel count from
        # GEE; otherwise we fall back to point-only semantics (n_pixels is
        # 1 if the point sample returned a value, else 0).
        if combined_reducer is not None:
            total_cells = max(1, round((max(effective_window_m, native_m) / native_m) ** 2))
            coverage_pct = 100.0 * (valid_count / total_cells) if total_cells else 0.0
            # Clamp to 100% — the geometric total can be off by a cell or two,
            # so a fully-valid window can otherwise read as 101%.
            coverage_pct = min(coverage_pct, 100.0)
            has_values = any(v is not None for v in window_stats.values())
            n_pixels = int(valid_count)
            had_nodata = valid_count < total_cells
        else:
            # Point-only request — n_pixels is 1 if the sample returned a value, else 0.
            any_pt = any(v is not None for v in point_stats.values())
            n_pixels = 1 if any_pt else 0
            coverage_pct = 100.0 if any_pt else 0.0
            has_values = any_pt
            had_nodata = not any_pt

        # Build per-band coverage breakdown for multiband datasets. Each entry
        # shows what fraction of window cells had valid data for that band —
        # useful when bands carry independent nodata masks (e.g. a derived band
        # may introduce NaNs that source bands don't have).
        band_coverage_pct: dict[str, float] = {}
        if per_band_counts and combined_reducer is not None:
            band_coverage_pct = {
                band: min(round(100.0 * count / total_cells, 2), 100.0)
                for band, count in per_band_counts.items()
            }

        meta = {
            "in_extent": has_values,
            "n_pixels": n_pixels,
            "had_nodata": had_nodata,
            "coverage_pct": coverage_pct if has_values else 0.0,
            **({"band_coverage_pct": band_coverage_pct} if band_coverage_pct else {}),
            "window_m": int(window_m),
            "raster_crs": self.crs,
            "region_crs": utm_crs,
            "transform": None,
            "dtype": "float64",
            "nodata": None,
            "src_path": self._src_label(),
            "window_arr": None,
            **self._resolve_date_info(date),
        }
        return stats, meta

    def fetch_stats_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        reducer_names: Sequence[str],
        *,
        dates: Sequence | None = None,
        progress_desc: str | None = None,
    ) -> List[tuple[dict[str, float | None], dict]]:
        """Compute server-side statistics for many points in parallel (Mode 1).

        Issues one ``reduceRegion`` call per point — or, when ``"point"`` is
        in *reducer_names*, a single ``ee.Dictionary`` containing both the
        window reduction and a Point-geometry reduction. GEE evaluates both
        branches server-side in one ``getInfo()`` round-trip, so the cost
        is one HTTP call per point regardless of whether the user asked
        for window stats, point sampling, or both.

        Output keys per point:
          - window stats: ``{reducer}`` (single band) or ``{band}_{reducer}`` (multi-band)
          - point sample: ``"point"``  (single band) or ``{band}_point``     (multi-band)

        Returns a list of ``(stats_dict, meta_dict)`` tuples — one per point.
        """
        # Split "point" out of reducer_names — it doesn't go through the
        # combined window reducer; it gets its own server-side branch.
        reducer_names = list(reducer_names)
        want_point = "point" in reducer_names
        window_reducers = [r for r in reducer_names if r != "point"]

        n = len(lats)
        lats = list(lats)
        lons = list(lons)
        date_list = list(dates) if dates is not None else [None] * n

        # Build the window reducer once when window stats are requested;
        # leave it None for point-only runs.
        if window_reducers:
            combined_reducer, suffixes = _build_combined_reducer(window_reducers)
        else:
            combined_reducer, suffixes = None, []

        # Warm the band cache once on the main thread so workers don't race
        # to populate it (and we avoid an extra getInfo from inside a worker).
        if not hasattr(self, "_cached_band_count"):
            try:
                self._get_band_name(self._get_image())
            except Exception:
                pass

        # Single ThreadPool: each worker handles both branches for its point.
        # Progress bar advances as each per-point future completes — gives the
        # user visible feedback for what is otherwise a long, opaque GEE call.
        results: List = [None] * n
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._fetch_stats_single,
                    lat,
                    lon,
                    window_m,
                    combined_reducer,
                    window_reducers,
                    suffixes,
                    want_point=want_point,
                    date=date,
                ): i
                for i, (lat, lon, date) in enumerate(zip(lats, lons, date_list))
            }
            with tqdm(total=n, desc=progress_desc or "GEE stats", unit="pt") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.warning("GEE stats fetch failed for point %d: %s", idx, e)
                        results[idx] = self._empty_stats_result(
                            window_m, window_reducers, want_point=want_point
                        )
                    pbar.update(1)

        return results

    # ------------------------------------------------------------------
    # Mode 2: Image export  (geemap → GeoTIFF)
    # ------------------------------------------------------------------

    def _export_single(
        self,
        lat: float,
        lon: float,
        window_m: int,
        output_path: Path,
        date=None,
        resample_m: float | None = None,
    ):
        """Export a GeoTIFF tile for one point with pixel-grid-snapped window.

        Snapping the centre coordinate to the nearest pixel grid multiple
        ensures all exported tiles have identical dimensions.

        If resample_m is set, the tile is exported at that resolution instead
        of the native image resolution — all tiles will be exactly
        round(window_m / resample_m) × round(window_m / resample_m) pixels.
        """
        img = self._get_image(date, lat=lat, lon=lon)
        # Use resample_m as the export scale when provided; fall back to native.
        scale_m = float(resample_m) if resample_m is not None else self._get_scale_value(img)
        utm = _get_utm_crs(lon, lat)

        # Project to UTM, snap to pixel grid, compute window corners
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
        cx, cy = transformer.transform(lon, lat)
        cx = _snap_to_grid(cx, scale_m)
        cy = _snap_to_grid(cy, scale_m)

        # Snap half-window to nearest whole pixel count so output dimensions
        # are consistent across points (window_m is approximate, not exact).
        half_pixels = max(1, round(window_m / 2 / scale_m))
        half_m = half_pixels * scale_m
        region = ee.Geometry.Rectangle(
            [cx - half_m, cy - half_m, cx + half_m, cy + half_m],
            proj=ee.Projection(utm),
            geodesic=False,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # verbose=False suppresses geemap's per-tile "Downloading data from..."
        # and "Data downloaded to..." prints, which otherwise interleave with
        # and disrupt the tqdm progress bar in export_tiles.
        geemap.ee_export_image(
            img,
            filename=str(output_path),
            scale=scale_m,
            region=region,
            crs=utm,
            verbose=False,
        )
        return output_path

    def export_tiles(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        output_dir: str | Path,
        *,
        ids: Sequence[str] | None = None,
        dates: Sequence | None = None,
        dataset_name: str = "dataset",
        resample_m: float | None = None,
        filename_suffix: str | None = None,
        progress_desc: str | None = None,
    ) -> tuple[list[Path | None], list[dict]]:
        """Export GeoTIFF tiles for many points in parallel (Mode 2).

        If resample_m is set, all tiles are exported at that resolution so they
        are exactly round(window_m / resample_m) × round(window_m / resample_m) pixels.

        ``filename_suffix`` is inserted before the .tif extension so multi-
        window runs can place every window's tiles in the same folder
        without overwriting one another.

        Returns ``(paths, meta_list)`` — one entry per input point in each
        list. ``paths[i]`` is ``None`` when the export for that point failed.
        """
        output_dir = Path(output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        n = len(lats)
        date_list = list(dates) if dates is not None else [None] * n
        id_list = list(ids) if ids is not None else [str(i) for i in range(n)]
        results: List = [None] * n

        # Warm the band-name cache so build_dataset_meta can read it later.
        try:
            self._get_band_name(self._get_image())
        except Exception:
            pass

        meta_list = [self._resolve_date_info(d) for d in date_list]

        # Suffix wrangling: when caller passes "200m", filenames become
        # "<id>-<dataset>-200m.tif". When suffix is None we keep the
        # historical "<id>-<dataset>.tif" naming so single-window callers
        # are completely unaffected.
        suffix_part = f"-{filename_suffix}" if filename_suffix else ""

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for i, (lat, lon, date, sample_id) in enumerate(zip(lats, lons, date_list, id_list)):
                output_path = output_dir / f"{sample_id}-{dataset_name}{suffix_part}.tif"
                future = executor.submit(
                    self._export_single, lat, lon, window_m, output_path, date, resample_m
                )
                future_to_idx[future] = i

            # Progress bar advances as each tile finishes downloading from GEE.
            with tqdm(total=n, desc=progress_desc or "GEE tiles", unit="tile") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.warning("GEE export failed for point %d: %s", idx, e)
                        results[idx] = None
                    pbar.update(1)

        return results, meta_list

    # ------------------------------------------------------------------
    # Dataset metadata  (sidecar JSON for one dataset's run)
    # ------------------------------------------------------------------

    def build_dataset_meta(
        self,
        spec: dict,
        meta_list: list | None = None,
        exported_paths: list | None = None,
        quality: dict | None = None,
        lats: Sequence[float] | None = None,
        lons: Sequence[float] | None = None,
    ) -> dict:
        """Build per-dataset metadata using this adapter's cached state.

        Includes GEE-specific fields like asset_type, collection date range,
        and per-tile UTM zones (when lats/lons are supplied for the raster
        path). Quality stats and date-selection info are added when present.
        """
        # Static dataset info from the catalog spec.
        meta: Dict[str, Any] = {
            "data_source": spec.get("data_source"),
            "path": spec.get("path"),
        }
        if spec.get("data_type"):
            meta["data_type"] = spec["data_type"]

        # Asset type: IMAGE vs IMAGE_COLLECTION — determined during __post_init__.
        if "collection" in self._dataset_spec:
            meta["asset_type"] = "IMAGE_COLLECTION"
        else:
            meta["asset_type"] = "IMAGE"

        # Native CRS (set in __post_init__).
        meta["native_crs"] = str(self.crs)

        # Native spatial resolution — prefer the catalog override when set
        # (GEE's nominalScale() returns the projection unit size, which is
        # misleading for datasets stored in geographic CRS, e.g. ~111km for
        # EPSG:4326 even when the true resolution is 30 m).
        catalog_scale = spec.get("dataset_spec", {}).get("native_scale_m")
        if catalog_scale is not None:
            meta["native_spatial_resolution_m"] = float(catalog_scale)
        elif self._native_proj is not None:
            try:
                meta["native_spatial_resolution_m"] = round(
                    float(self._native_proj.nominalScale().getInfo()), 2
                )
            except Exception:
                pass

        # Band names — populated on first fetch via _get_band_name().
        band_names = getattr(self, "_cached_band_names", None)
        if band_names:
            meta["band_names"] = band_names

        # Tile CRS: GEE exports each tile in the UTM zone of its centre point,
        # so we list all unique zones touched by the input sample set.
        if lats is not None and lons is not None:
            zones = build_tile_crs_zones(lats, lons)
            if zones:
                meta["tile_crs"] = zones

        # Collection date range: available when timestamps were fetched.
        timestamps = getattr(self, "_collection_timestamps", None)
        if timestamps is not None and len(timestamps) > 0:
            meta["collection_date_range"] = [
                timestamps.min().strftime("%Y-%m-%d"),
                timestamps.max().strftime("%Y-%m-%d"),
            ]

        # Pass-through catalog field for dataset description.
        dataset_info = spec.get("dataset_information")
        if dataset_info:
            meta["dataset_information"] = dataset_info

        # Per-point date-selection summary (nearest/clamped/most-recent counts).
        if meta_list:
            date_info = summarize_date_info(meta_list)
            if date_info is not None:
                meta["date_info"] = date_info

        # QC/coverage stats accumulated during processing. Make a copy so we
        # don't mutate the caller's dict when adding the tile-export summary.
        quality = dict(quality or {})
        # Tile-export summary: populated only when this call is for the raster path.
        if exported_paths is not None:
            n_points = len(lats) if lats is not None else len(exported_paths)
            quality["tiles"] = summarize_tile_export(exported_paths, n_points)
        if quality:
            meta["quality"] = quality

        return meta


# Module-level side effect: registers this adapter under the "earth_engine"
# data_source so config.py can dispatch to it without an explicit import.
# The try/except around the `_register` import at the top of the file makes
# this no-op if the registry isn't available (avoids circular-import pain).
if _register is not None:
    _register("earth_engine", GeeRasterAdapter)
