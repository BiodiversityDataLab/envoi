# src/envoi/adapters/earth_engine/_image.py
"""Everything needed to construct the ee.Image that the adapter will sample.

This collects, in one place:

* GEE SDK initialisation (only when Earth Engine is not yet initialized) and
  session-pool tuning (once per process).
* Collection timestamp fetching and nearest / contains date selection,
  with the per-point fallback that selects an own image near the centre day.
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

# The value that the timestamp request gives for a missing system:time_start
# or system:time_end. aggregate_array leaves out the images without the
# property (seen live on 2026-10-12), so a list of starts and a list of ends
# would no longer pair by position. With this value, both lists keep one
# entry for each image. It is text, so it cannot be a time: times are
# numbers, and they are negative before 1970 (TerraClimate starts in 1958),
# so a negative number cannot mark a missing time.
_MISSING_TIME = "missing"


def _image_start_and_end(image):
    """Return a feature with the start and end time of ``image`` (for ``ImageCollection.map``).

    The values are ``system:time_start`` and ``system:time_end`` in
    milliseconds since 1970, or ``_MISSING_TIME`` when the image does not have
    the property. Earth Engine evaluates the condition for each image on the
    server, in the request that reads the times.
    """
    property_names = image.propertyNames()
    start = ee.Algorithms.If(
        property_names.contains("system:time_start"),
        image.get("system:time_start"),
        _MISSING_TIME,
    )
    end = ee.Algorithms.If(
        property_names.contains("system:time_end"),
        image.get("system:time_end"),
        _MISSING_TIME,
    )
    return ee.Feature(None, {"start": start, "end": end})


def _get_collection_time_bounds(
    collection_id: str,
    *,
    bounds_geometry: ee.Geometry | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[pd.DatetimeIndex | None, pd.DatetimeIndex | None]:
    """Fetch start/end timestamps for a GEE ImageCollection.

    Returns (start_times, end_times) as sorted DatetimeIndex objects. Each end
    time is the end of an image with that start time: the request pairs the
    two times of each image. The start times are unique: when several images
    have the same start, end_times holds the latest of their ends.
    An image without system:time_start is not in the index.
    Returns (start_times, None) when an image with a start has no
    system:time_end (e.g. user-uploaded assets that omit it). Returns
    (None, None) when no image with a start time is left after the filters
    (for example, no image of the collection intersects ``bounds_geometry``).
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
        # Map each image to its start and end time, with _MISSING_TIME for a
        # missing time, so that the two lists below have one entry for each
        # image, in the same order. Two plain aggregate_array calls on the
        # image properties would leave out the missing values, and the two
        # lists would pair the start of one image with the end of another.
        image_times_collection = image_collection.map(_image_start_and_end)
        # Fetch both lists in a single round-trip by wrapping in ee.Dictionary.
        raw = ee.Dictionary(
            {
                "starts": image_times_collection.aggregate_array("start"),
                "ends": image_times_collection.aggregate_array("end"),
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

    # Pair each start with the end at the same position: both are the times
    # of one image. strict=True: lists of different lengths would mean that
    # the positions do not match, and every pair after the gap would be
    # wrong, so the run stops instead. Then drop the images without a start:
    # we still need at least one valid start to do anything useful,
    # regardless of which branch we end up in.
    paired_times = [
        (start, end)
        for start, end in zip(start_times, end_times, strict=True)
        if start != _MISSING_TIME
    ]
    clean_starts = [int(start) for start, _ in paired_times]
    if not clean_starts:
        return None, None

    # Branch 1: an image with a start has no end → return a start-only
    # index. Build it from clean_starts (sorted + deduped) so it's
    # directly usable for nearest-neighbour lookups downstream.
    if any(end == _MISSING_TIME for _, end in paired_times):
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
    # start would make the index non-unique and the interval lookups by
    # position ambiguous. Keep the latest end of each start, so the
    # interval covers every image with that start.
    bounds_df = pd.DataFrame({"start": clean_starts, "end": [int(end) for _, end in paired_times]})
    bounds_df = bounds_df.groupby("start", as_index=False, sort=True)["end"].max()
    start_index = pd.DatetimeIndex(pd.to_datetime(bounds_df["start"], unit="ms", origin="unix"))
    end_index = pd.DatetimeIndex(pd.to_datetime(bounds_df["end"], unit="ms", origin="unix"))
    return start_index, end_index


def _find_nearest_timestamp(
    time_starts: pd.DatetimeIndex,
    target: pd.Timestamp,
) -> tuple[pd.Timestamp, bool]:
    """Return the image start nearest to *target* in whole UTC days, and whether it was clamped.

    Dates compare as whole days (the day rule): the nearest start is the one
    with the fewest days between its start day and the day of *target*, and
    of two equally near starts the earlier one. The date is clamped only when
    its day is before the start day of the first image or after the start day
    of the last image. A sample date has day precision, so an image that
    starts later on the sample day is on that day, not after it.

    Returns (nearest_timestamp, was_clamped).
    """
    start_days = time_starts.normalize()
    target_day = target.normalize()
    was_clamped = bool(target_day < start_days.min() or target_day > start_days.max())
    day_gaps = abs((start_days - target_day).days)
    # Of the starts with the smallest gap, take the earliest.
    nearest = time_starts[day_gaps == day_gaps.min()].min()
    return nearest, was_clamped


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
    Dates compare as whole days in UTC (the day rule): an image that starts at
    any time on the day of date_timestamp starts on or before that day. The
    image is the one with the latest start before the next day. A day before
    the first image's start day selects the first image, and a day after the
    last interval the last image.
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
    # Day rule: compare whole days. The sample date is 00:00 of its day, so
    # comparing the times would give the previous image (or a clamp to the
    # first image) for an image that starts later on the sample day, for
    # example at 10:00. Select the latest start before the next day instead.
    # A day before the first start day gives no such start: clamp to the
    # first image. A day after the last interval selects the last image.
    next_day = date_timestamp.normalize() + pd.DateOffset(days=1)
    idx = int(time_starts.searchsorted(next_day, side="left")) - 1
    idx = max(0, idx)

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
    # is empty and .first() returns null, so the point's request fails. Use the
    # next day as the end in that case, so the range always covers the
    # start day and the selected image is always inside it.
    if end_date.normalize() <= start_date.normalize():
        end_date = start_date + pd.DateOffset(days=1)

    return start_date.strftime(date_format), end_date.strftime(date_format)


# ---------------------------------------------------------------------------
# Per-point fallback  (an own image near the centre day, selected on the server)
# ---------------------------------------------------------------------------

# Milliseconds in one day. The fallback compares image starts as whole days
# in UTC (the day rule).
_DAY_MS = 86_400_000

# The date windows of the fallback, in whole days before and after the centre
# day: first 32 days, then 366 days. A point whose own images are all outside
# both windows stays a failed point. A search of the whole collection took
# 12 to 27 s per point in live checks (2026-10-12); the 32-day window took
# about 1.6 s and the 366-day window about 4.3 s. See the decision log,
# "Image selection per point".
_FALLBACK_WINDOWS_DAYS = (32, 366)

# The image property that holds the sort key of the "nearest" fallback.
_FALLBACK_SORT_KEY = "envoi_fallback_sort_key"

# The sort key of the "nearest" fallback is
# day gap × _DAY_GAP_KEY_FACTOR + (image start − window start) in milliseconds.
# The factor is larger than the longest window, (2 × 366 + 1) days ≈ 6.3e10 ms,
# so the day gap decides first, and an earlier start wins between equal gaps.
# One key avoids two sorts: Earth Engine does not promise a stable sort. The
# largest key, about 3.7e13, is below 2**53, so a double holds it exactly.
_DAY_GAP_KEY_FACTOR = 1e11


def _epoch_milliseconds(timestamp: pd.Timestamp) -> int:
    """Return a UTC time without time zone in milliseconds since 1970, as ``system:time_*``."""
    return (timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)


def _resolve_centre_day(
    date_timestamp: pd.Timestamp, time_starts: pd.DatetimeIndex
) -> pd.Timestamp:
    """Return the centre day of a point: its date, limited to the batch index, at 00:00 UTC.

    The date is limited to the range from the first start to the latest start
    of the batch index. The caller passes the latest start for a point
    without a sample date.
    """
    centre_date = min(max(date_timestamp, time_starts.min()), time_starts.max())
    return centre_date.normalize()


def _nearest_fallback_step(own_images, centre_ms: int, window_days: int):
    """Return ``(window_images, image)`` for one window of the "nearest" fallback.

    ``window_images`` are the own images that start in
    ``[C − window_days, C + window_days + 1)`` days, with C the centre day.
    ``image`` is the one with the fewest whole days between its start day and
    C, and the earliest start between equal gaps. Both are server-side objects.
    """
    window_start_ms = centre_ms - window_days * _DAY_MS
    window_images = own_images.filterDate(window_start_ms, centre_ms + (window_days + 1) * _DAY_MS)
    centre_day_number = centre_ms // _DAY_MS

    def with_sort_key(image):
        start_ms = ee.Number(image.get("system:time_start"))
        day_gap = start_ms.divide(_DAY_MS).floor().subtract(centre_day_number).abs()
        sort_key = day_gap.multiply(_DAY_GAP_KEY_FACTOR).add(start_ms.subtract(window_start_ms))
        return image.set(_FALLBACK_SORT_KEY, sort_key)

    nearest_image = window_images.map(with_sort_key).sort(_FALLBACK_SORT_KEY).first()
    return window_images, nearest_image


def _contains_fallback_steps(own_images, centre_ms: int) -> list[tuple]:
    """Return the ``(candidate_images, image)`` steps of the "contains" fallback, in order.

    1. The own images that contain the centre day C (day rule): the latest start.
    2. For each window W (32, then 366 days): the latest start in
       ``[C − W, C + 1)`` days, then the earliest start in ``[C + 1, C + 1 + W)``.
    """
    next_day_ms = centre_ms + _DAY_MS

    # Day rule: an image contains day C when it starts before C + 1 day and
    # ends after the start of C. An image without an end, or with an end on
    # its start day, ends at its start day + 1 day. So an image that starts
    # on day C contains C, whatever its end, and an image that starts before
    # C contains C only when its end is after the start of C. Filter.gt is
    # false for an image without system:time_end, so such an image contains
    # only its start day.
    containing_images = own_images.filter(
        ee.Filter.And(
            ee.Filter.lt("system:time_start", next_day_ms),
            ee.Filter.Or(
                ee.Filter.gte("system:time_start", centre_ms),
                ee.Filter.gt("system:time_end", centre_ms),
            ),
        )
    )
    steps = [(containing_images, containing_images.sort("system:time_start", False).first())]

    for window_days in _FALLBACK_WINDOWS_DAYS:
        images_before = own_images.filterDate(centre_ms - window_days * _DAY_MS, next_day_ms)
        images_after = own_images.filterDate(next_day_ms, next_day_ms + window_days * _DAY_MS)
        steps.append((images_before, images_before.sort("system:time_start", False).first()))
        steps.append((images_after, images_after.sort("system:time_start").first()))
    return steps


def _build_fallback_image(own_images, centre_day: pd.Timestamp, policy: str):
    """Return the server-side fallback image of one point: an own image near the centre day.

    ``own_images`` are the point's images after ``filterBounds`` and the
    optional ``UTM_ZONE`` filter. ``policy`` is "nearest" or "contains": the
    catalog check rejects other values, and a "mosaic" collection does not
    select by date. The rules are in ``_nearest_fallback_step`` ("nearest")
    and ``_contains_fallback_steps`` ("contains").

    The result is null on the server when no own image is in any window: the
    point's request then fails, so the point stays a failed point. Building
    it sends no request. Earth Engine evaluates it only inside the point's
    own request, and only when the fast path has no image.
    """
    centre_ms = _epoch_milliseconds(centre_day)
    if policy == "contains":
        steps = _contains_fallback_steps(own_images, centre_ms)
    else:
        steps = [
            _nearest_fallback_step(own_images, centre_ms, window_days)
            for window_days in _FALLBACK_WINDOWS_DAYS
        ]

    # Chain the steps from the last to the first, so that a step is used only
    # when every step before it has no image. limit(1) lets Earth Engine stop
    # at the first image of a step.
    selected_image = None
    for candidate_images, candidate_image in reversed(steps):
        selected_image = ee.Algorithms.If(
            candidate_images.limit(1).size().gt(0), candidate_image, selected_image
        )
    return selected_image


def _count_gap_days(
    sample_date: str | pd.Timestamp, image_start_ms: int, image_end_ms: int | None, policy: str
) -> int:
    """Return the number of whole days between a sample date and an image (the day rule).

    Dates compare as whole days in UTC. No request: the times are numbers
    that Earth Engine already returned.

    Args:
        sample_date: The point's sample date (text ``YYYY-MM-DD`` or a
            timestamp). Only its day counts.
        image_start_ms: ``system:time_start`` of the image, in milliseconds
            since 1970.
        image_end_ms: ``system:time_end`` of the image in milliseconds, or
            None when the image has no end.
        policy: The ``collection_date_policy``, "nearest" or "contains".

    Returns:
        For "nearest": the number of days between the sample day and the
        image's start day. For "contains": 0 when the image contains the
        sample day; otherwise the number of days from the sample day to the
        nearest day that the image contains: its start day for an image after
        the sample day, and its last day for an image before it. An image
        without an end, or with an end on its start day, contains only its
        start day.
    """
    one_day = pd.Timedelta(days=1)
    sample_day = pd.Timestamp(sample_date).normalize()
    start_day = pd.to_datetime(image_start_ms, unit="ms").normalize()
    if policy != "contains":
        return abs((start_day - sample_day).days)

    # Day rule: the image ends at least at the end of its start day.
    end = start_day + one_day
    if image_end_ms is not None:
        end = max(end, pd.to_datetime(image_end_ms, unit="ms"))

    if start_day > sample_day:
        return (start_day - sample_day).days
    if end > sample_day:
        # The image starts on or before the sample day and ends after the
        # start of it: it contains the sample day.
        return 0
    # The image ends at or before the start of the sample day. Its last day is
    # the last day that starts before its end: an end at 00:00 does not
    # contain the day that starts at that time.
    last_day = end.ceil("D") - one_day
    return (sample_day - last_day).days


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
    """Build an ee.Image from a dataset_spec config dict, and the request for its dates.

    Pipeline:
        1. Load image (single asset) or ImageCollection.
        2. For collections only: filterBounds → optional UTM-zone filter
           → date selection (filterDate + .first(), the per-point fallback,
           or mosaic when no date).
        3. Compute derived bands (slope, aspect, …) and addBands them.
        4. Final band select — drop source bands when only derived
           bands were requested, otherwise keep both.

    With ``circular_aspect=True``, step 3 returns a companion image for the
    circular statistics of the derived ``aspect`` band instead: two bands,
    ``sin`` and ``cos``, of the aspect in radians. Aspect and slope come from
    the same image (and so the same first band) as the derived bands. Flat
    pixels (slope 0) are masked: Earth Engine gives them aspect 0 (north), so
    they would pull the mean towards north although they face no direction.
    The same arguments select the same image as without the flag.

    For collections, the date handling strategy is:
    - date provided + timestamps cached (the batch index): the fast path
      selects the point's own image in the range of the batch-selected image
      (``_resolve_date_filter_range``; for "contains" it ends at most one day
      after the centre day). With a point ``geometry``, a point without an
      own image in that range gets the per-point fallback: an own image near
      the centre day, selected on the server (``_build_fallback_image``).
    - date provided + no timestamps (fallback): filterDate ±1 day, .first()
    - no date: mosaic (most recent non-masked pixel per position)
    Own images are the images at the point after the optional UTM-zone filter.

    When the collection has no image at the point ``geometry`` on any date
    (before the UTM-zone and date filters), the image is the first image of
    the collection that has every configured source band (``bands``; the
    first image of the collection when no source band is configured), with
    every pixel masked. The point gets the values of a point without data:
    null statistics, except ``count``, which is 0 (Earth Engine decides this
    in the point's request). When the point has images, but no own image is
    selected (none in the date range or the fallback windows, or none in the
    point's UTM zone), the image is null, and the point's request fails.

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

    Returns
    -------
    (image, image_dates)
        ``image`` is the ee.Image to sample. ``image_dates`` is a server-side
        value for the point's request: ``{"start": ..., "end": ...,
        "fallback": ...}``, the ``system:time_start`` and ``system:time_end``
        (milliseconds, ``None`` when the image has no end) of the image that
        the date selection chose, and ``fallback``, True when the per-point
        fallback chose it (the fast-path range had no own image of the point)
        and False for the fast path. It is null when the collection has no
        image at the point (the masked image). ``image_dates`` is ``None``
        when there is no per-point date selection (an image asset, a mosaic,
        no point geometry, or no batch index). Building both sends no request.
    """
    image = None
    image_dates = None

    if "image" in dataset_spec:
        image = ee.Image(dataset_spec["image"])

    elif "collection" in dataset_spec:
        image_collection = ee.ImageCollection(dataset_spec["collection"])

        if geometry is not None:
            image_collection = image_collection.filterBounds(geometry)
        # Every image at the point, on any date, before the selection filters
        # below. The date branch checks it for an image at the point.
        bounds_collection = image_collection

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

            # The per-point fallback needs a point and the batch index. Without
            # a point there is no location to search, and without the index
            # there is no centre day (the ±1 day range above is already a
            # search on the server).
            use_point_fallback = geometry is not None and collection_time_starts is not None
            if use_point_fallback:
                centre_day = _resolve_centre_day(date_ts, collection_time_starts)
                # "contains": the batch-selected range can end weeks after the
                # centre day when another point's image starts later. This
                # point's own image that starts after the centre day would then
                # be selected, although it does not contain the date. End the
                # range at most at the end of the centre day.
                if date_policy == "contains":
                    capped_end = min(pd.Timestamp(end), centre_day + pd.Timedelta(days=1))
                    end = capped_end.strftime("%Y-%m-%d")

            # Fast path: the point's own image in the batch-selected range.
            dated_collection = image_collection.filterDate(start, end)
            image = dated_collection.first()

            # Collections whose images have different dates at different
            # places (for example Sentinel-2 tiles) can have no own image of
            # the point in that range, although the point has images. Then
            # select an own image near the centre day on the server, in the
            # point's own request. Earth Engine evaluates only the branch of
            # If that it needs, so a point with an image in the fast-path
            # range does not pay for the fallback.
            if use_point_fallback:
                fast_path_has_image = dated_collection.limit(1).size().gt(0)
                fallback_image = _build_fallback_image(image_collection, centre_day, date_policy)
                image = ee.Image(ee.Algorithms.If(fast_path_has_image, image, fallback_image))

            # When the collection has no image at the point on any date (for
            # example an ocean point of a tiled land product), .first() gives
            # a null image. Every request on a null image raises ("Parameter
            # 'image' is required"), so the point would be a failed point
            # although it only has no data. Use the first image of the whole
            # collection with every pixel masked instead, so the point gets
            # the values of a point without data, as a point outside a mosaic
            # does. The image must have the configured source bands, or the
            # band selection below fails ("Band ... not found") and the point
            # is a failed point again: the images of some collections have
            # different sets of bands. So take the first image that has every
            # configured source band. Without configured source bands (all
            # bands, or only derived bands), take the first image.
            # The check uses the collection after filterBounds only, before
            # the UTM-zone filter and the date filter, on purpose. When the
            # point has images, but no own image is selected (none in the
            # date range or the fallback windows, or none in the point's UTM
            # zone because of a wrong use_utm_zone setting), the image stays
            # null and the point is a failed point with an error. A problem
            # with the image selection then stays visible and is not reported
            # as missing data.
            # Earth Engine evaluates the condition in the point's own request,
            # so no request is added, and limit(1) lets it stop at the first
            # image. Without a point geometry, there is no location to check,
            # and the condition would only count the images of the whole
            # collection, so it is left out.
            if geometry is not None:
                placeholder_collection = ee.ImageCollection(dataset_spec["collection"])
                configured_bands = dataset_spec.get("bands") or []
                if isinstance(configured_bands, str):
                    configured_bands = [configured_bands]
                for band in configured_bands:
                    placeholder_collection = placeholder_collection.filter(
                        ee.Filter.listContains("system:band_names", band)
                    )
                masked_placeholder = ee.Image(placeholder_collection.first()).updateMask(0)
                point_has_images = bounds_collection.limit(1).size().gt(0)

                # The dates of the selected image, from the image itself, so
                # that the QC columns name the image that the point actually
                # used (the fallback image can differ from the batch-selected
                # image). The masked image has no dates of its own: its point
                # has no image, so the value is null, with the same condition.
                # "fallback" tells whether the per-point fallback selected the
                # image (the fast-path range had no own image of the point).
                # The server decides it with the condition of the If above, so
                # the adapter does not have to guess it from the dates.
                if use_point_fallback:
                    image_dates = ee.Algorithms.If(
                        point_has_images,
                        ee.Dictionary(
                            {
                                "start": image.get("system:time_start"),
                                "end": image.get("system:time_end"),
                                "fallback": ee.Algorithms.If(fast_path_has_image, False, True),
                            }
                        ),
                        None,
                    )
                image = ee.Image(ee.Algorithms.If(point_has_images, image, masked_placeholder))
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
        circular_aspect_image = ee.Image.cat(
            aspect_radians.sin().rename("sin"),
            aspect_radians.cos().rename("cos"),
        ).updateMask(not_flat_mask)
        return circular_aspect_image, image_dates

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

    if bands:
        # Both source and derived specified — include all of them.
        source_bands = bands if isinstance(bands, list) else [bands]
        image = image.select(source_bands + derived_bands_list)
    elif derived_bands_list:
        # Only derived bands specified — drop source bands from the output.
        # An empty `bands` list (a per-call band list of only derived bands,
        # such as ["aspect"]) means no source bands, so it comes here too.
        image = image.select(derived_bands_list)
    # else: neither specified — return the image as-is.

    return image, image_dates
