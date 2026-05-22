# src/envoi/adapters/earth_engine/_image.py
"""Everything needed to construct the ee.Image that the adapter will sample.

This collects, in one place:

* GEE SDK initialisation (one-time per process) and session-pool tuning.
* Pixel-grid snapping for export window alignment.
* Collection timestamp fetching and nearest / contains date selection.
* Derived-band registration (slope, aspect, ...) and computation.
* The central ``_build_image`` pipeline: load → spatial filter → date select →
  derive bands → final band select.

The ``GeeRasterAdapter`` class composes these as building blocks.
"""

from __future__ import annotations

import logging

import ee
import pandas as pd

from ...geo import get_utm_zone_label

logger = logging.getLogger(__name__)

# Flag flipped to True the first time _ensure_gee_init() actually initialises
# the SDK. Kept at module scope so repeated calls across threads / batches are
# cheap no-ops after the first.
_gee_initialized = False


# ---------------------------------------------------------------------------
# GEE initialization
# ---------------------------------------------------------------------------


def _patch_ee_session_pool(pool_size: int) -> None:
    """Raise the urllib3 connection-pool size on GEE's shared requests.Session.

    All Earth Engine API calls (including ``getInfo()``) go through one shared
    ``requests.Session`` stored at ``ee.data._get_state().requests_session``.
    The default urllib3 pool size is 10, so when more than 10 worker threads
    are active in parallel, the pool overflows and urllib3 logs:

        WARNING:urllib3.connectionpool:Connection pool is full,
        discarding connection: earthengine.googleapis.com

    The warning itself is harmless (the request still succeeds), but each
    overflowed thread pays the cost of a fresh TCP handshake. Mounting a
    larger pool eliminates both the noise and the handshake overhead.

    ``_get_state()`` is a private EE API — if the EE SDK ever moves this,
    we silently skip the patch and the warnings reappear, but nothing breaks.
    """
    try:
        from requests.adapters import HTTPAdapter

        session = ee.data._get_state().requests_session
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        session.mount("https://", adapter)
    except Exception as e:
        # ``_get_state()`` is private EE API — if the EE SDK ever moves it,
        # this patch will fail and the urllib3 "Connection pool is full"
        # warnings described in the docstring will reappear. Log at WARNING
        # (not DEBUG) so the failure stays visible: the runtime keeps
        # working, but the user sees noisier output until we update the
        # patch. Matches the visibility level used elsewhere for "couldn't
        # do auto-detect" cases (e.g. catalog._inspect_raster).
        logger.warning(
            "Could not patch EE session pool size (%s); urllib3 warnings may persist.", e
        )


def _ensure_gee_init():
    """Initialize GEE once per process, skip if already active."""
    global _gee_initialized
    if _gee_initialized:
        return
    try:
        ee.Number(1).getInfo()
        _gee_initialized = True
    except Exception:
        from ...auth import init_gee

        init_gee()
        _gee_initialized = True
    # Size the connection pool generously so parallel workers don't overflow
    # the urllib3 default of 10. Covers the default max_workers (20) and most
    # user overrides without retuning. 50 idle TCP connections is trivial.
    _patch_ee_session_pool(50)


# ---------------------------------------------------------------------------
# Geometry helpers  (pixel-grid snapping)
# UTM helpers live in ``geo.get_utm_crs`` / ``geo.get_utm_zone_label``
# so both adapters share one implementation.
# ---------------------------------------------------------------------------


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
    *,
    bounds_geometry: ee.Geometry | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[pd.DatetimeIndex | None, pd.DatetimeIndex | None]:
    """Fetch start/end timestamps for a GEE ImageCollection.

    Returns (start_times, end_times) as sorted DatetimeIndex objects.
    Returns (start_times, None) when end times are unavailable (e.g. user-uploaded
    assets that omit system:time_end). Returns (None, None) only on total failure.
    One getInfo() round-trip, done once per dataset.

    Parameters
    ----------
    bounds_geometry : ee.Geometry, optional
        Restrict the timestamp index to images whose footprint intersects this
        geometry. Critical for per-tile per-pass collections (e.g. DynamicWorld
        has ~10 million images globally — without a spatial filter, aggregating
        every system:time_start exceeds GEE's compute budget and times out).
        Passing a ``MultiPoint`` of the batch's sample coordinates narrows the
        result to just the tiles we actually care about.
    date_range : (start, end) tuple of pd.Timestamp, optional
        Additional pre-filter on the collection's time interval. Combined with
        ``bounds_geometry`` so even busy collections (Sentinel-2 over a few
        tiles for many years) stay well within budget.
    """
    try:
        image_collection = ee.ImageCollection(collection_id)
        # Narrow the collection BEFORE aggregating timestamps. Order matters
        # only for clarity: GEE optimises filter chains internally, but doing
        # the cheaper spatial filter first keeps the chain readable. Without
        # these filters, ``aggregate_array`` materialises every image's
        # timestamp — fine for ~1k-image collections, fatal for the per-pass
        # ones (DynamicWorld, raw Sentinel-2, …) which contain millions.
        if bounds_geometry is not None:
            image_collection = image_collection.filterBounds(bounds_geometry)
        if date_range is not None:
            range_start, range_end = date_range
            # filterDate is half-open; add a day on the end so an exact
            # match at range_end is still included.
            image_collection = image_collection.filterDate(
                range_start.strftime("%Y-%m-%d"),
                (range_end + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
            )
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
    time_starts: pd.DatetimeIndex,
    target: pd.Timestamp,
) -> tuple[pd.Timestamp, bool]:
    """Return the timestamp closest to *target*, clamping to range.

    Returns (nearest_timestamp, was_clamped).
    """
    if target <= time_starts.min():
        return time_starts.min(), target < time_starts.min()
    if target >= time_starts.max():
        return time_starts.max(), target > time_starts.max()
    idx = time_starts.get_indexer([target], method="nearest")[0]
    return time_starts[idx], False


def _resolve_date_filter_range(
    date_timestamp: pd.Timestamp,
    policy: str,
    time_starts: pd.DatetimeIndex | None = None,
    time_ends: pd.DatetimeIndex | None = None,
) -> tuple[str, str]:
    """Return (start, end) date strings for ImageCollection.filterDate().

    When cached timestamps are available, uses them to pin the exact image
    interval. When not (server-side fallback), broadens the window so GEE
    can resolve the image without a client-side index — asymmetrically:
        - policy="nearest"  → [date - 1d, date + 1d]
        - policy="contains" → [date,      date + 1d]

    policy="contains" selects the image whose interval contains date_timestamp.
    policy="nearest"  selects the image with the closest start timestamp.
    """
    date_format = "%Y-%m-%d"

    if time_starts is None:
        # No cached index — let GEE resolve server-side with a wider window.
        if policy == "contains":
            return date_timestamp.strftime(date_format), (
                date_timestamp + pd.DateOffset(days=1)
            ).strftime(date_format)
        else:
            return (date_timestamp - pd.DateOffset(days=1)).strftime(date_format), (
                date_timestamp + pd.DateOffset(days=1)
            ).strftime(date_format)

    if policy == "nearest":
        nearest, _ = _find_nearest_timestamp(time_starts, date_timestamp)
        return nearest.strftime(date_format), (nearest + pd.DateOffset(days=1)).strftime(
            date_format
        )

    # --- policy == "contains": find the image interval that contains date_timestamp. ---
    # Clamp to collection boundaries when date_timestamp is out of range.
    if date_timestamp <= time_starts.min():
        idx = 0
    elif date_timestamp >= time_starts.max():
        idx = len(time_starts) - 1
    else:
        idx = int(time_starts.searchsorted(date_timestamp, side="right") - 1)
        idx = max(0, min(idx, len(time_starts) - 1))

    start_date = time_starts[idx]

    # Use true interval end when available, otherwise fall back to next start.
    if time_ends is not None and len(time_ends) == len(time_starts):
        end_date = time_ends[idx]
    elif idx + 1 < len(time_starts):
        end_date = time_starts[idx + 1]
    else:
        end_date = start_date + pd.DateOffset(days=1)

    # filterDate(start, end) is half-open. Some collections have images with
    # system:time_end == system:time_start (instantaneous events); using the
    # raw end would produce an empty filter and .first() would return null.
    # Bump by one second whenever end <= start so the selected image is
    # always inside the range.
    if end_date <= start_date:
        end_date = start_date + pd.Timedelta(seconds=1)

    return start_date.strftime(date_format), end_date.strftime(date_format)


# ---------------------------------------------------------------------------
# Derived bands  (slope, aspect, …)
# ---------------------------------------------------------------------------

# Dispatch table for derived bands. Maps the user-facing derived-band name to
# the GEE function that computes it from a source image. Add a new derived
# band by inserting one entry here — KNOWN_DERIVED_BANDS and the runtime
# dispatch inside `_apply_derived_bands` both read from this dict, so no
# further edits are needed.
_DERIVED_BAND_DISPATCH = {
    "slope": ee.Terrain.slope,
    "aspect": ee.Terrain.aspect,
}

# Names that are recognised as *derived* bands. When the user passes a unified
# bands list at the call site (via extract()), names appearing in this set are
# split out and forwarded to the adapter as `derived_bands` rather than `bands`.
# Derived directly from the dispatch table so the two can never drift apart.
KNOWN_DERIVED_BANDS = frozenset(_DERIVED_BAND_DISPATCH)


def _apply_derived_bands(image, derived):
    """Compute derived bands and add them alongside the existing bands of `image`.

    `derived` may be either a single band name (e.g. "slope") or a list of names
    (e.g. ["slope", "aspect"]). Each derived band is computed from `image` via
    the dispatch table above and added to the output via `addBands()`, so the
    source bands are preserved.

    Raises ValueError if an unknown derived band name is given — silent fallback
    was previously a trap that produced confusing "missing output" bugs.
    """
    # Normalize to a list so callers can pass either form.
    if isinstance(derived, str):
        derived_names = [derived]
    else:
        derived_names = list(derived)

    for derived_band_name in derived_names:
        # Look up the compute function once per name; an unknown name short-
        # circuits with a clear error before any GEE-side work is done.
        compute = _DERIVED_BAND_DISPATCH.get(derived_band_name)
        if compute is None:
            raise ValueError(
                f"Unknown derived band '{derived_band_name}'. "
                f"Supported: {', '.join(sorted(KNOWN_DERIVED_BANDS))}."
            )
        image = image.addBands(compute(image))

    return image


# ---------------------------------------------------------------------------
# Image building  (load → filter → date select → cloud mask → reduce → bands)
# ---------------------------------------------------------------------------


def _build_image(
    dataset_spec: dict,
    date=None,
    geometry=None,
    collection_time_starts: pd.DatetimeIndex | None = None,
    collection_time_ends: pd.DatetimeIndex | None = None,
    *,
    lat: float | None = None,
    lon: float | None = None,
):
    """Build an ee.Image from a dataset_spec config dict.

    Pipeline:
        1. Load image (single asset) or ImageCollection.
        2. For collections only: filterBounds → optional UTM-zone filter
           → date selection (filterDate + .first(), or mosaic when no date).
        3. Compute derived bands (slope, aspect, …) and addBands them.
        4. Final band select — drop source bands when only derived
           bands were requested, otherwise keep both.

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
    collection_time_starts : pd.DatetimeIndex, optional
        Cached start timestamps for the collection.
    collection_time_ends : pd.DatetimeIndex, optional
        Cached end timestamps for the collection (aligned with start times).
    """
    image = None

    if "image" in dataset_spec:
        image = ee.Image(dataset_spec["image"])

    elif "collection" in dataset_spec:
        image_collection = ee.ImageCollection(dataset_spec["collection"])

        if geometry is not None:
            image_collection = image_collection.filterBounds(geometry)

        # Some tiled collections (e.g. satellite embeddings) need UTM-zone
        # filtering to avoid selecting the wrong tile for a point.
        if dataset_spec.get("use_utm_zone") and lat is not None and lon is not None:
            utm_zone = get_utm_zone_label(lon, lat)
            image_collection = image_collection.filter(ee.Filter.eq("UTM_ZONE", utm_zone))

        # --- Date-based image selection ---
        date_policy = str(dataset_spec.get("collection_date_policy", "nearest")).lower()
        if date is not None:
            date_ts = pd.to_datetime(date)
            start, end = _resolve_date_filter_range(
                date_ts,
                date_policy,
                time_starts=collection_time_starts,
                time_ends=collection_time_ends,
            )
            image_collection = image_collection.filterDate(start, end)
            image = image_collection.first()
        else:
            # No date: mosaic (most recent non-masked pixel per position)
            image = image_collection.mosaic()

    else:
        raise ValueError("dataset_spec must contain 'image' or 'collection'")

    # Compute derived bands BEFORE selecting `bands`. This allows derived bands
    # to read any source band they need internally — the user doesn't have to
    # list those source bands in `bands` just to make the derivation work.
    derived_bands = dataset_spec.get("derived_bands")
    if derived_bands:
        image = _apply_derived_bands(image, derived_bands)

    # Determine the final output bands.
    bands = dataset_spec.get("bands")
    derived_bands_list = (
        [derived_bands]
        if isinstance(derived_bands, str)
        else list(derived_bands) if derived_bands else []
    )

    if bands is not None:
        # Both source and derived specified — include all of them.
        source_bands = bands if isinstance(bands, list) else [bands]
        image = image.select(source_bands + derived_bands_list)
    elif derived_bands_list:
        # Only derived bands specified — drop source bands from the output.
        image = image.select(derived_bands_list)
    # else: neither specified — return the image as-is.

    return image
