# src/envoi/adapters/earth_engine/_image.py
"""Everything needed to construct the ee.Image that the adapter will sample.

This collects, in one place:

* GEE SDK initialisation (only when Earth Engine is not yet initialized) and
  session-pool tuning (once per process).
* Collection timestamp fetching and nearest / contains date selection.
* Derived-band registration (slope, aspect, ...) and computation.
* The central ``_build_image`` pipeline: load → spatial filter → date select →
  derive bands → final band select.

The ``GeeRasterAdapter`` class composes these as building blocks.
"""

from __future__ import annotations

import logging
import math

import ee
import pandas as pd

from ...geo import get_utm_zone_label

logger = logging.getLogger(__name__)

# Set to True the first time _ensure_gee_init() patches the session pool, so
# that the patch runs only once per process. Kept at module scope so repeated
# calls across adapters and batches skip it.
_session_pool_patched = False


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


def _ensure_gee_init() -> None:
    """Initialize Earth Engine when it is not initialized, and enlarge its connection pool once.

    ``ee.data.is_initialized()`` reads local state and sends no request.
    An Earth Engine session that is already initialized (by the user, or by an
    earlier run) is kept as it is. Only when Earth Engine is not initialized does
    this call ``auth.init_gee()``, which reads envoi's service-account credentials.

    The session-pool patch runs once per process, after initialization.

    Raises:
        FileNotFoundError: Earth Engine is not initialized and ``init_gee()``
            finds no credentials file.
        RuntimeError: Earth Engine is not initialized and refuses the credentials.
    """
    global _session_pool_patched

    # Initialize only when needed. A test request (and a new initialization
    # when it fails) would replace a working session after a temporary
    # network error, and with it the user's own credentials and project.
    if not ee.data.is_initialized():
        from ...auth import init_gee

        init_gee()

    # Size the connection pool generously so parallel workers don't overflow
    # the urllib3 default of 10. Covers the default max_workers (20) and most
    # user overrides without retuning. 50 idle TCP connections is trivial.
    # Earth Engine creates the shared session during initialization, so the
    # patch must run after it.
    if not _session_pool_patched:
        _patch_ee_session_pool(50)
        _session_pool_patched = True


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

    Returns (start_times, end_times) as sorted DatetimeIndex objects. The start
    times are unique: when several images have the same start, end_times holds
    the latest of their ends.
    Returns (start_times, None) when end times are unavailable (e.g. user-uploaded
    assets that omit system:time_end). Returns (None, None) when no image with a
    start time is left after the filters (for example, no image of the
    collection intersects ``bounds_geometry``).
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

    Raises
    ------
    RuntimeError
        Earth Engine could not return the timestamps. The exception chains
        the original error. Every point of the batch needs the timestamps to
        select its image, so the caller lets this error stop the run.
    """
    # Request the timestamps. Only the Earth Engine calls are inside the try,
    # so an error in the local processing below is not reported as an Earth
    # Engine error.
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
    # Broad catch that re-raises a clear error: Earth Engine reports unrelated
    # failures (network, quota, unknown asset) with one exception type.
    # Without the timestamps, the adapter cannot select the image for each
    # point's sample date. Continuing would silently give every point
    # statistics from a mosaic of the most recent images, or from a search
    # within one day of its sample date, so the run stops here.
    except Exception as error:
        raise RuntimeError(
            f"Earth Engine could not return the image dates of collection "
            f"'{collection_id}'. envoi needs these dates to select the image for "
            f"each sample point, so the run stops. Check the collection ID and your "
            f"network connection, then run the extraction again. "
            f"Original error: {error}"
        ) from error
    start_times = raw["starts"]
    end_times = raw["ends"]

    # Pair each start with the end at the same position, and only then drop
    # the pairs without a start. This keeps the pairs correct when Earth
    # Engine returns a null value for an image without the property. When
    # Earth Engine instead leaves such an image out of a list, the positions
    # of the two lists do not match. Lists of different lengths show this:
    # there are then no reliable pairs, so use no end times. Known limitation:
    # one image without a start and another image without an end give two
    # lists of the same length, and their pairs can then be wrong.
    if len(end_times) == len(start_times):
        paired_times = list(zip(start_times, end_times))
    else:
        paired_times = [(start, None) for start in start_times]
    # Drop pairs without a start — we still need at least one valid start
    # to do anything useful, regardless of which branch we end up in.
    paired_times = [(start, end) for start, end in paired_times if start is not None]
    clean_starts = [int(start) for start, _ in paired_times]
    if not clean_starts:
        return None, None

    # Branch 1: end-times are missing or partial → return a start-only
    # index. Build it from clean_starts (sorted + deduped) so it's
    # directly usable for nearest-neighbour lookups downstream.
    if any(end is None for _, end in paired_times):
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
    # from the same DataFrame, with one row per start and sorted by start,
    # so they stay aligned element-by-element — required for
    # `time_ends[idx]` lookups to map back to the correct start.
    # Two images can have the same start and different ends. A duplicate
    # start makes the index non-unique, and the nearest-image lookup
    # (`get_indexer`) then raises. Keep the latest end of each start, so
    # the interval covers every image with that start.
    bounds_df = pd.DataFrame({"start": clean_starts, "end": [int(end) for _, end in paired_times]})
    bounds_df = bounds_df.groupby("start", as_index=False, sort=True)["end"].max()
    start_index = pd.DatetimeIndex(pd.to_datetime(bounds_df["start"], unit="ms", origin="unix"))
    end_index = pd.DatetimeIndex(pd.to_datetime(bounds_df["end"], unit="ms", origin="unix"))
    return start_index, end_index


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


def _is_mosaic_date_policy(dataset_spec: dict) -> bool:
    """True when a collection's images tile space rather than time.

    Declared in the catalog as ``collection_date_policy: mosaic``. Such a
    collection is one static product cut into geographic tiles; the per-image
    timestamp records when a tile was acquired, not a step in a time series.
    Date selection is meaningless for it — picking "the nearest image in time"
    returns a tile somewhere else entirely — so the adapter spatially filters
    and mosaics instead, ignoring dates completely.

    Shared by the adapter and :func:`_build_image` so both agree on how the
    catalog value is spelled and compared.
    """
    return str(dataset_spec.get("collection_date_policy", "nearest")).lower() == "mosaic"


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
    A date outside the collection's range selects the first or the last image.
    The returned range always covers at least one calendar day.
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

    # filterDate(start, end) is half-open, and both ends are whole days. Some
    # collections have images with system:time_end == system:time_start
    # (instantaneous events), or an end later on the same day as the start.
    # Formatted as days, start and end are then the same string: the filter
    # is empty and .first() returns null, so every statistic is null. Use the
    # next day as the end in that case, so the range always covers the
    # start day and the selected image is always inside it.
    if end_date.normalize() <= start_date.normalize():
        end_date = start_date + pd.DateOffset(days=1)

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
    circular_aspect: bool = False,
):
    """Build an ee.Image from a dataset_spec config dict.

    Pipeline:
        1. Load image (single asset) or ImageCollection.
        2. For collections only: filterBounds → optional UTM-zone filter
           → date selection (filterDate + .first(), or mosaic when no date).
        3. Compute derived bands (slope, aspect, …) and addBands them.
        4. Final band select — drop source bands when only derived
           bands were requested, otherwise keep both.

    With ``circular_aspect=True``, step 3 returns a companion image for the
    circular statistics of the derived ``aspect`` band instead: two bands,
    ``sin`` and ``cos``, of the aspect in radians. Aspect and slope come from
    the same image (and so the same first band) as the derived bands. Flat
    pixels (slope 0) are masked: Earth Engine gives them aspect 0 (north), so
    they would pull the mean towards north although they face no direction.

    For collections, the date handling strategy is:
    - date provided + timestamps cached: find nearest timestamp, filterDate, .first()
    - date provided + no timestamps (fallback): filterDate ±1 day, .first()
    - no date: mosaic (most recent non-masked pixel per position)

    When the date filter leaves no image at the point, the image is the first
    image of the collection with every pixel masked, so its statistics are
    null (Earth Engine decides this in the point's request).

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
        if _is_mosaic_date_policy(dataset_spec):
            # Space-tiled collection: dates carry no meaning here, so mosaic
            # whatever filterBounds left us regardless of any date passed in.
            # The guard is belt-and-braces — the adapter never passes a date
            # for these — but it keeps the two modules from drifting apart.
            #
            # mosaic() discards the source projection, leaving the result on
            # the EPSG:4326 1-degree default. Reductions are unaffected (they
            # take an explicit scale), but neighbourhood algorithms are not:
            # ee.Terrain.slope/aspect silently return null on a projectionless
            # image. Stamping the collection's native projection back on keeps
            # derived bands working, and matches what the single-image path
            # gets for free from .first().
            native_projection = (
                ee.ImageCollection(dataset_spec["collection"]).first().select(0).projection()
            )
            image = image_collection.mosaic().setDefaultProjection(native_projection)
        elif date is not None:
            date_ts = pd.to_datetime(date)
            start, end = _resolve_date_filter_range(
                date_ts,
                date_policy,
                time_starts=collection_time_starts,
                time_ends=collection_time_ends,
            )
            image_collection = image_collection.filterDate(start, end)
            # When no image covers the point on the selected dates (for
            # example an ocean point of a tiled land product), the filtered
            # collection is empty, and .first() gives a null image. Every
            # request on a null image raises ("Parameter 'image' is required"),
            # so the point would be a failed point. Use the first image of the
            # whole collection with every pixel masked instead: it has the
            # collection's bands, so the band selection and the derived bands
            # below work, and the point gets null statistics and in_extent
            # False, as a point outside a mosaic does. Earth Engine evaluates
            # the condition in the point's own request, so no request is added.
            masked_placeholder = ee.Image(
                ee.ImageCollection(dataset_spec["collection"]).first()
            ).updateMask(0)
            image = ee.Image(
                ee.Algorithms.If(
                    image_collection.size().gt(0),
                    image_collection.first(),
                    masked_placeholder,
                )
            )
        else:
            # No date: mosaic (most recent non-masked pixel per position)
            image = image_collection.mosaic()

    else:
        raise ValueError("dataset_spec must contain 'image' or 'collection'")

    # Circular aspect companion image: sine and cosine of the aspect, without
    # flat pixels. Uses the same dispatch functions, on the same image, as the
    # derived bands below.
    if circular_aspect:
        aspect_radians = _DERIVED_BAND_DISPATCH["aspect"](image).multiply(math.pi / 180.0)
        not_flat_mask = _DERIVED_BAND_DISPATCH["slope"](image).gt(0)
        return ee.Image.cat(
            aspect_radians.sin().rename("sin"),
            aspect_radians.cos().rename("cos"),
        ).updateMask(not_flat_mask)

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
