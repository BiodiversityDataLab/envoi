# src/envoi/adapters/earth_engine/adapter.py
"""``GeeRasterAdapter`` — the Earth Engine implementation of BaseAdapter.

The free-function helpers used by this class live in the sibling modules:

* ``_image`` — GEE init, timestamp resolution, derived bands,
  per-point ``ee.Image`` construction (``_build_image``).
* ``_reducers`` — combined-reducer assembly + result parsing + per-band
  coverage summary.
* ``_tiles`` — sync-download size guard + retrying tile downloader.

Keeping those out of this file leaves the class as the orchestration story:
how each batch fans out per-point work, what each per-point fetch composes,
and how the per-dataset metadata is assembled.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ee
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..._filenames import build_tile_filenames
from ...catalog import load_defaults
from ...geo import build_tile_crs_zones, get_utm_crs, tile_pixel_count, tile_top_left
from ...metadata import summarize_date_info, summarize_tile_export
from ...progress import ProgressStepCallback, emit_progress_step
from ...qc import format_point_error
from ...reducers import KNOWN_REDUCERS
from ..base import _BAND_NAMING_KEY, _PER_BAND_NAMING, BaseAdapter
from ._image import (
    KNOWN_DERIVED_BANDS,
    _build_image,
    _count_gap_days,
    _ensure_gee_init,
    _find_nearest_timestamp,
    _get_collection_time_bounds,
    _is_mosaic_date_policy,
)
from ._reducers import (
    _CLASS_COUNT_KEY_RE,
    _build_combined_reducer,
    _circular_aspect_statistics,
    _dedupe_categorical_for_ee,
    _extract_count_from_reduce_result,
    _extract_per_band_counts,
    _parse_multiband_result,
    _parse_point_result,
    _parse_reduce_result,
    _summarize_band_coverage,
)
from ._tiles import _check_tile_size, _download_tile_via_url

try:
    from .. import register as _register
except Exception:
    _register = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class GeeRasterAdapter(BaseAdapter):
    """Adapter that samples data directly from Google Earth Engine.

    Two extraction modes:

    1. **fetch_stats_batch** — server-side statistics via ``reduceRegion``
       with combined reducers.  Much faster for the common stats use-case.
       When ``"point"`` is included in the requested reducers, an exact-pixel
       sample at each ``(lat, lon)`` is added as a second server-side branch
       and resolved in the same round-trip.
    2. **export_tiles** — download full GeoTIFF tiles via
       ``ee.Image.getDownloadURL`` + ``requests`` (no ``geemap`` dependency).

    For ImageCollections, the adapter selects an image per point based
    on the catalog's ``collection_date_policy``: ``"nearest"`` (default)
    picks the image whose start timestamp is closest to the sample
    date; ``"contains"`` picks the image whose interval covers the
    sample date. When no date is provided, the most recent image is
    used.
    """

    # Capability declarations (see BaseAdapter). No annotations: in this
    # dataclass, an annotation would make them fields.
    output_types = frozenset({"tabular", "raster"})
    supported_reducers = KNOWN_REDUCERS
    computable_derived_bands = KNOWN_DERIVED_BANDS

    spec: dict[str, Any]
    _static_image: Any = field(default=None, init=False, repr=False)
    _needs_per_point_date: bool = field(default=False, init=False, repr=False)
    _native_proj: Any = field(default=None, init=False, repr=False)
    # True when the bands come from a per-call band list (the mark
    # _BAND_NAMING_KEY in the spec): statistic keys are then named after their
    # bands, also for one band.
    _band_named_keys: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def __post_init__(self):
        _ensure_gee_init()

        # dataset_spec holds extra config (bands, derived bands, asset
        # type override, native scale override, etc.). Always auto-detect
        # asset type from GEE, then merge with user config.
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
        # An empty `bands` list (a per-call list of only derived bands) is not
        # copied, so the image keeps only the derived bands (see _build_image).
        if self.spec.get("bands") and "bands" not in dataset_spec:
            dataset_spec["bands"] = self.spec["bands"]
        if self.spec.get("derived_bands") and "derived_bands" not in dataset_spec:
            dataset_spec["derived_bands"] = self.spec["derived_bands"]
        self._band_named_keys = self.spec.get(_BAND_NAMING_KEY) == _PER_BAND_NAMING

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
        self._is_collection = is_collection
        self._collection_time_starts = None
        self._collection_time_ends = None
        # Lazy-init flag for the collection timestamp index. The fetch is
        # deferred to the first ``fetch_stats_batch`` / ``export_tiles`` call
        # so it can be filtered by the actual batch's bounds and date range —
        # without that filter, per-tile per-pass collections (DynamicWorld
        # and similar) hit GEE's compute budget and time out.
        self._timestamps_loaded = False

        # Cache the native projection from the source — composite images
        # (mosaic, mean, etc.) lose per-band projection info, so we grab it
        # from the first image of a collection or the image's first band.
        if is_collection:
            self._native_proj = (
                ee.ImageCollection(dataset_spec["collection"]).first().select(0).projection()
            )
            # Assume per-point date selection. The lazy timestamp fetch flips
            # this to False (and builds a static-mosaic fallback) when no
            # image of the collection intersects the batch's points. A failed
            # timestamp request stops the run instead.
            self._needs_per_point_date = True

            # ...unless the catalog declares the collection has no meaningful
            # time axis. Some collections tile *space*, not time: every image
            # is a fixed geographic tile of one static product, and its
            # timestamp records when that tile happened to be acquired rather
            # than marking a step in a time series. Copernicus DEM GLO-30
            # 2024_1 is the built-in example — 4133 distinct timestamps
            # globally, 15 of them at a single sample point. Treating those as
            # a time axis makes "most recent image" select a tile on the other
            # side of the planet, so every statistic comes back null. Declaring
            # the policy skips date handling entirely and mosaics the tiles
            # covering each point, which is what such a product actually means.
            if _is_mosaic_date_policy(dataset_spec):
                self._needs_per_point_date = False
                # Suppress the lazy timestamp fetch: there is no time index to
                # build, and for a global tiled collection the aggregate_array
                # call is expensive enough to be worth skipping outright.
                self._timestamps_loaded = True
                # Global mosaic for the no-coordinates callers (band-name
                # probing, scale detection). Constructing it is lazy on GEE's
                # side — nothing is computed until something reduces over a
                # region — so this costs nothing here.
                self._static_image, _ = _build_image(dataset_spec)
        elif "image" in dataset_spec:
            self._native_proj = ee.Image(dataset_spec["image"]).select(0).projection()
            self._static_image, _ = _build_image(dataset_spec)
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

    def _ensure_collection_timestamps(
        self,
        lats: Sequence[float] | None = None,
        lons: Sequence[float] | None = None,
        dates: Sequence | None = None,  # kept for API symmetry; see body for why unused.
    ) -> None:
        """Fetch the collection's start/end timestamps on first use.

        Called by the public batch methods (``fetch_stats_batch``,
        ``export_tiles``) before any per-point work. Idempotent — subsequent
        calls return immediately once the index is cached, or once the fetch
        found no image at the batch's points.

        Filters the collection by the batch's bounding geometry BEFORE the
        ``aggregate_array`` call so per-tile per-pass collections (e.g.
        DynamicWorld with ~10M images globally) return a tractable few-
        thousand-image index instead of timing out server-side. The ``dates``
        argument is accepted but intentionally NOT used as a filter — see
        the comment below.

        Raises ``RuntimeError`` (from ``_get_collection_time_bounds``) when
        Earth Engine cannot return the timestamps. Every point needs them,
        so the error stops the run.
        """
        # Non-collections (single IMAGE assets) have no timestamp axis, and
        # repeat calls after a completed fetch are no-ops.
        if not self._is_collection or self._timestamps_loaded:
            return

        # Build a MultiPoint geometry from the batch's sample coordinates.
        # GEE's filterBounds keeps any image whose footprint intersects at
        # least one of the points, so for tiled collections this is the
        # difference between scanning millions of images and scanning the
        # handful of tiles we actually care about. Skipped (no spatial
        # filter) when the caller didn't pass coordinates — falls back to
        # the unfiltered behaviour the adapter used to have.
        bounds_geometry = None
        if lats is not None and lons is not None and len(lats) > 0:
            point_pairs = [[float(lon), float(lat)] for lat, lon in zip(lats, lons)]
            bounds_geometry = ee.Geometry.MultiPoint(point_pairs)

        # NOTE on date_range: we deliberately do NOT pre-filter by the
        # batch's input dates here. Two reasons:
        #   1. Many collections expose `system:time_start` values that have
        #      nothing to do with the input sample date (e.g. GLO30 reports
        #      2006-01-01 for every tile while users typically query with
        #      modern dates). A naive ``[date-1y, date+1y]`` filter excludes
        #      every tile and breaks the per-point fetch.
        #   2. Out-of-range date tests (date = 2099) deliberately exercise
        #      the clamping path — but a date filter built from that date
        #      would itself produce an empty collection, defeating the test.
        # filterBounds alone is enough for every catalog entry we ship: it
        # already bounds DynamicWorld to a few thousand timestamps for a
        # typical batch, which fits comfortably inside GEE's compute budget.
        # If a future per-pass collection ever turns out to need date
        # narrowing too, ``_get_collection_time_bounds`` accepts ``date_range``
        # for the caller to pass in explicitly.
        collection_id = self._dataset_spec["collection"]
        start_times, end_times = _get_collection_time_bounds(
            collection_id,
            bounds_geometry=bounds_geometry,
        )
        self._collection_time_starts = start_times
        self._collection_time_ends = end_times
        # An empty index means that no image of the collection (with a start
        # time) intersects the batch's points. Flip to the static-mosaic
        # fallback so per-point work doesn't crash trying to read a None
        # timestamp index. A failed request does not reach this line: it
        # raises in _get_collection_time_bounds and stops the run.
        self._needs_per_point_date = start_times is not None
        if not self._needs_per_point_date:
            self._static_image, _ = _build_image(self._dataset_spec)
        self._timestamps_loaded = True

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
        # Cache the resolved CRS so build_dataset_meta can report the asset's
        # actual projection rather than the spec/default fallback. Empty
        # string is preserved as-is so the build_dataset_meta truthiness
        # check falls back to self.crs in that case.
        self._cached_native_crs = crs
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

    def _get_image(
        self, date=None, lat=None, lon=None, *, circular_aspect: bool = False
    ) -> tuple[ee.Image, Any | None]:
        """Return the ee.Image to sample, building per-point if needed, and its date request.

        For IMAGE assets the pre-built static image is always returned,
        regardless of date or coordinates.

        For ImageCollections:
        - date provided + coords → per-point image picked via
          ``collection_date_policy`` (nearest/contains), with the per-point
          fallback of ``_build_image``
        - date provided + no coords → server-side date filter only
          (no spatial selection — used when timestamps weren't cached
          or for global probes)
        - no date + coords → per-point most-recent image
        - no date + no coords → most-recent image, cached on first call

        With ``circular_aspect=True``, returns the companion image for the
        circular aspect statistics (``sin`` and ``cos`` bands, see
        ``_build_image``), selected by the same rules. The same date and
        coordinates select the same image as without the flag. This image is
        built for each call and never read from or written to
        ``_static_image``. Building it sends no request: Earth Engine
        evaluates it only in the point's ``reduceRegion`` request.

        Returns:
            ``(image, image_dates)``. ``image_dates`` is the server-side value
            for the dates of the selected image (see ``_build_image``), for a
            collection with per-point date selection and coordinates. It is
            ``None`` in every other case.
        """
        # Per-point branch has priority for collections that need spatial
        # filtering (e.g. tiled DEM collections). When coordinates are given,
        # always build a fresh per-point image — never reuse `_static_image`,
        # which may have been built earlier for a no-coords call and would
        # therefore point at an arbitrary global tile that doesn't cover the
        # current sample. Reusing it caused `img.sample(point)` to return
        # empty props, so the "point" reducer silently dropped its columns
        # for tiled IMAGE_COLLECTIONs (issue with dem_glo30 + "point" stat).
        # Space-tiled collections (collection_date_policy: mosaic) ignore dates
        # entirely: spatially filter to the tiles covering this point and
        # mosaic them. Built fresh per point rather than cached, for the same
        # reason as the per-point-date branch below — a cached global mosaic
        # would be an arbitrary tile that need not cover this sample.
        if _is_mosaic_date_policy(self._dataset_spec) and lat is not None and lon is not None:
            return _build_image(
                self._dataset_spec,
                geometry=ee.Geometry.Point([lon, lat]),
                lat=lat,
                lon=lon,
                circular_aspect=circular_aspect,
            )

        if self._needs_per_point_date and lat is not None and lon is not None:
            target_date = (
                self._collection_time_starts.max() if date is None else pd.to_datetime(date)
            )
            geom = ee.Geometry.Point([lon, lat])
            return _build_image(
                self._dataset_spec,
                date=target_date,
                geometry=geom,
                collection_time_starts=self._collection_time_starts,
                collection_time_ends=self._collection_time_ends,
                lat=lat,
                lon=lon,
                circular_aspect=circular_aspect,
            )

        # IMAGE assets or already-cached no-coords fallback.
        # Invariant: for IMAGE_COLLECTIONs, _static_image is only ever set by
        # the most-recent-fallback branch below — so reusing it here for a
        # no-coords call returns the same most-recent global image, which is
        # safe for band-name probing and global tile-export, but never for
        # per-point sampling (the per-point branch above runs first).
        # The circular aspect image is never cached: it continues to the
        # branches below, which build it from the same dataset spec.
        if not circular_aspect and date is None and self._static_image is not None:
            return self._static_image, None

        # Collection with no date and no coords: build (and cache) a global
        # most-recent fallback. Used by tile-export and band-name probing
        # paths that don't have a specific point in mind. Without a point,
        # there is no per-point date request.
        if date is None and self._needs_per_point_date:
            most_recent = self._collection_time_starts.max()
            logger.info(
                "No date/geometry provided for collection %s; using most recent image (%s).",
                self._dataset_spec.get("collection"),
                most_recent.strftime("%Y-%m-%d"),
            )
            most_recent_image, _ = _build_image(
                self._dataset_spec,
                date=most_recent,
                collection_time_starts=self._collection_time_starts,
                collection_time_ends=self._collection_time_ends,
                circular_aspect=circular_aspect,
            )
            if not circular_aspect:
                self._static_image = most_recent_image
            return most_recent_image, None

        # Final fallback — a date was provided but neither the per-point
        # branch (no coords) nor the static-image branch (date is None)
        # could handle it. Two real paths reach here:
        #   1. Collection whose timestamp index is empty (no image intersects
        #      the batch's points) and date is not None (server-side date
        #      filter with no client-side index).
        #   2. Collection with timestamps but no coords and date is not
        #      None (per-point date selection without a spatial filter).
        # The circular aspect image also reaches here without a date, in
        # place of the static image. date None stays None (pd.to_datetime
        # would give NaT, which _build_image reads as a date), so
        # _build_image loads an IMAGE asset as the static image does, and
        # mosaics a collection without date selection.
        dt = pd.to_datetime(date) if date is not None else None
        geom = ee.Geometry.Point([lon, lat]) if lat is not None else None
        return _build_image(
            self._dataset_spec,
            dt,
            geometry=geom,
            collection_time_starts=self._collection_time_starts,
            collection_time_ends=self._collection_time_ends,
            lat=lat,
            lon=lon,
            circular_aspect=circular_aspect,
        )

    def _get_band_name(self, img: ee.Image) -> str:
        """Get the first band name from the image (needed for result parsing).

        Side effect: populates ``_cached_band_name``, ``_cached_band_names``,
        and ``_cached_band_count`` on the first successful call. Other methods
        (``_fetch_stats_single``, ``_guard_tile_size``, ``build_dataset_meta``)
        rely on these caches being warm.

        When the bands are not in the config, this asks Earth Engine for the
        band names. If that request fails, the Earth Engine exception
        propagates and nothing is cached, so the next call asks again.
        """
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
                self._cached_band_names = combined
                self._cached_band_count = len(combined)
                # Set last: worker threads read this attribute as "cache is filled".
                self._cached_band_name = combined[0]
            return self._cached_band_name
        # Fallback: ask GEE (costs one getInfo call, cached after first success).
        # A failed request raises and caches nothing. Caching a fallback name
        # would make every later point of the batch parse its result with a
        # band name and band count that do not match the image.
        if not hasattr(self, "_cached_band_name"):
            names = img.bandNames().getInfo()
            self._cached_band_names = names  # full list for metadata
            self._cached_band_count = len(names)
            # Set last: worker threads read this attribute as "cache is filled".
            self._cached_band_name = names[0] if names else "value"
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
        utm = get_utm_crs(lon, lat)
        return point.buffer(window_m / 2, proj=ee.Projection(utm)).bounds(maxError=1)

    def _src_label(self) -> str:
        dataset_config = self._dataset_spec
        return f"gee://{dataset_config.get('image', dataset_config.get('collection', 'unknown'))}"

    # ------------------------------------------------------------------
    # Date metadata  (per-point image timestamps for the meta sidecar)
    # ------------------------------------------------------------------

    def _resolve_date_info(self, date, image_times: dict | None) -> dict:
        """Return the date metadata of one point, for its meta dict.

        ``date_clamped`` and ``date_source`` compare the sample date with the
        batch index (see ``_resolve_date_source``). ``image_time_start`` and
        ``image_time_end`` (``%Y-%m-%d``) name the image that the point's
        request actually selected, which the per-point fallback can choose
        instead of the batch-selected image. They come from ``image_times``.

        Args:
            date: The point's sample date, or None in a run without dates.
            image_times: The ``image_dates`` value of ``_build_image`` that
                Earth Engine returned for the point: ``{"start": ..., "end":
                ..., "fallback": ...}``, the times in milliseconds since 1970,
                with ``"end"`` None when the image has no end, and
                ``"fallback"`` True when the per-point fallback selected the
                image. None when the collection has no image at the point (the
                masked image).

        Returns:
            The four date keys, without ``image_time_end`` when the image has
            no end. An empty dict when there is no batch index (image assets,
            mosaic collections, and collections without an image at the
            batch's points), and when ``image_times`` is None: a point without
            an image has no date fields, and ``date_info`` does not count it.

            One internal key is added when the per-point fallback selected the
            image, the point has a sample date, and the date is not clamped:
            ``fallback_image_gap_days``, the whole days between the sample
            date and the image (see ``_count_gap_days``). ``extract()`` warns
            when it is large. It stays in memory: no output writes it.
        """
        if self._collection_time_starts is None or image_times is None:
            return {}

        date_clamped, date_source = self._resolve_date_source(date)
        # Every selection filter of _build_image uses system:time_start, so
        # the selected image always has a start.
        date_info = {
            "date_clamped": date_clamped,
            "date_source": date_source,
            "image_time_start": pd.to_datetime(image_times["start"], unit="ms").strftime(
                "%Y-%m-%d"
            ),
        }
        if image_times.get("end") is not None:
            date_info["image_time_end"] = pd.to_datetime(image_times["end"], unit="ms").strftime(
                "%Y-%m-%d"
            )

        # The gap between the sample date and an image that the per-point
        # fallback selected, for the warning about images far from the date.
        # Only for a point with a sample date that is not clamped: without a
        # date there is no gap, and a clamped date is outside the dates of
        # the batch's images, which date_clamped already reports.
        if image_times.get("fallback") and date is not None and not date_clamped:
            policy = str(self._dataset_spec.get("collection_date_policy", "nearest")).lower()
            date_info["fallback_image_gap_days"] = _count_gap_days(
                date, image_times["start"], image_times.get("end"), policy
            )
        return date_info

    def _resolve_date_source(self, date) -> tuple[bool, str]:
        """Return ``(date_clamped, date_source)`` of a point, from its date and the batch index.

        No request: the batch index is in memory. ``date_source`` is
        ``most_recent_no_date`` without a date, ``clamped_to_nearest`` for a
        date outside the collection's range, and otherwise
        ``contains_sample_date`` ("contains") or ``nearest_to_sample``.

        Uses attributes:
            _collection_time_starts: The batch index start times (not None).
            _collection_time_ends: The batch index end times, or None.
        """
        time_starts = self._collection_time_starts

        if date is None:
            # No-date run: the most recent image of the batch index.
            return False, "most_recent_no_date"

        dt = pd.to_datetime(date)
        policy = str(self._dataset_spec.get("collection_date_policy", "nearest")).lower()

        if policy == "contains":
            # "contains" policy: the date is clamped when it falls outside the
            # collection's full range. The range starts at the first image's
            # start day and ends at the end of the last image interval, so a
            # date on the first start day or inside the last interval is not
            # clamped. Dates compare as whole days (the day rule): an image
            # that starts at 10:00 on the sample day starts on or before that
            # day. The range is the one that _resolve_date_filter_range
            # selects from.
            sample_day = dt.normalize()
            if sample_day < time_starts.min().normalize():
                return True, "clamped_to_nearest"

            # The last interval ends at the system:time_end of the image with
            # the latest start. Without end times, it ends at the end of that
            # start's day (the day rule), as the date range of
            # _resolve_date_filter_range does: a later day is clamped, also
            # when the image starts at 10:00.
            # That function's date range also always covers the whole start
            # day of the image, so an image whose end is on its start day (or
            # equal to its start) is selected for a date on that day. Use the
            # same day here, or such a date would be reported as clamped.
            time_ends = self._collection_time_ends
            last_start = time_starts.max()
            if time_ends is not None and len(time_ends) == len(time_starts):
                last_interval_end = time_ends[len(time_starts) - 1]
            else:
                last_interval_end = last_start.normalize() + pd.DateOffset(days=1)
            last_interval_end = max(
                last_interval_end, last_start.normalize() + pd.DateOffset(days=1)
            )
            if sample_day >= last_interval_end:
                return True, "clamped_to_nearest"
            return False, "contains_sample_date"

        # Default "nearest" policy: the date is clamped when it is before the
        # first start or after the latest start of the batch index.
        _, was_clamped = _find_nearest_timestamp(time_starts, dt)
        return was_clamped, "clamped_to_nearest" if was_clamped else "nearest_to_sample"

    # ------------------------------------------------------------------
    # Empty-result builders  (results of a failed point)
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

    def _empty_stats_result(self, window_m: int, error: Exception):
        """Return the ``(stats, meta)`` result of a failed point.

        The stats dict has no keys. ``_append_stat_columns`` then gives the
        failed point a null value in every statistic column that the other
        points of the batch create, and the failed point creates no column of
        its own. (Placeholder keys would not match the success schema: a
        multi-band point has ``{band}_{reducer}`` keys, and the band names are
        not always known when a request fails.)

        The meta dict has the QC values of a point outside the extent, plus the
        error text (``error``) and the exception (``exception``), so that
        ``extract()`` can report the point and stop the run when every point
        failed. ``exception`` stays in memory and is never written to JSON.
        """
        _, meta = self._empty_result(window_m)
        meta["error"] = format_point_error(error)
        meta["exception"] = error
        return {}, meta

    # ------------------------------------------------------------------
    # Mode 1: Server-side stats  (reduceRegion with combined reducers)
    # ------------------------------------------------------------------

    def _fetch_stats_single(
        self,
        lat: float,
        lon: float,
        window_m: int,
        combined_reducer: ee.Reducer | None,
        reducer_names: Sequence[str],
        suffixes: list[str],
        *,
        want_point: bool = False,
        date=None,
        original_reducer_names: Sequence[str] | None = None,
    ):
        """Compute server-side stats for a single point via reduceRegion.

        Up to three reductions are issued together as a single ``ee.Dictionary``
        and resolved with one ``getInfo()`` round-trip:

        - **window** — runs only when *combined_reducer* is provided. The
          user's reducers are combined with ``ee.Reducer.count()`` for QC,
          and ``reduceRegion`` is applied over the buffered square region.
        - **aspect_circular** — runs only when the derived ``aspect`` band is
          reduced and ``mean``, ``std``, or ``var`` is a window reducer. It
          reduces the sine and cosine of the aspect of the non-flat pixels
          over the same region, and the circular values replace the linear
          ``mean``, ``std``, and ``var`` of aspect
          (see ``_circular_aspect_statistics``).
        - **point** — runs only when *want_point* is True. ``Reducer.first()``
          is applied over an ``ee.Geometry.Point`` at the exact (lat, lon)
          to fetch the value of the pixel containing the point.

        A fourth branch, **image_dates**, is no reduction: for collections
        with per-point date selection, it returns the start and end of the
        image that the date selection chose, and whether the per-point
        fallback chose it, for the date fields of the meta dict (see
        ``_resolve_date_info``).

        GEE evaluates the branches in parallel server-side, so callers pay
        one round-trip for all of them. Returns a single
        ``(stats_dict, meta_dict)`` tuple with merged keys:
        - window keys: ``{reducer}`` (single band) or ``{band}_{reducer}`` (multi-band)
        - point  keys: ``"point"``  (single band) or ``{band}_point``     (multi-band)
        A per-call band list gives the multi-band keys, also for one band.
        For a band whose window has no valid pixel, ``sum`` is None and
        ``count`` is 0.

        ``original_reducer_names`` carries the *un-deduped* user request
        (i.e. with ``class_fraction`` still present if asked for). It is
        used only by the categorical post-process: ``class_fraction`` is
        derived from the histogram counts after parsing, and the count
        keys are stripped if the user only asked for fractions.
        """
        # Fall back to the deduped list when the caller didn't pass the
        # original — preserves the old single-reducer test entry points.
        if original_reducer_names is None:
            original_reducer_names = reducer_names
        # image_dates: the dates of the selected image, for collections with
        # per-point date selection (None otherwise). See _build_image.
        img, image_dates = self._get_image(date, lat=lat, lon=lon)
        region = self._make_region(lat, lon, window_m)

        utm_crs = get_utm_crs(lon, lat)
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

        # Use multi-band mode ({band}_{reducer} keys) when the bands come from
        # a per-call band list, also with one band, so that the column names
        # say which band the user selected. Otherwise use it when no band is
        # specified (auto-detect all bands) or a list of bands is specified,
        # and the image has more than one band. A single named band, and a
        # catalog default with one band, keep simple {reducer} naming.
        spec_band = self._dataset_spec.get("bands")
        multiband = self._band_named_keys or (not isinstance(spec_band, str) and band_count > 1)

        if multiband:
            img_to_reduce = img
        else:
            img_to_reduce = img.select(band_name)

        # With "count" as the only window reducer, no QC count is combined
        # with it, and Earth Engine names the output of a single reducer by
        # the band name only ({"elevation": 247}), not "{band}_count".
        count_only = list(reducer_names) == ["count"]

        # Aspect is a direction, so its mean, std and var must be circular
        # statistics: the linear mean of 350° and 10° is 180° (south), not 0°
        # (north). They come from a companion image of the sine and cosine of
        # the aspect (see _build_image), reduced in the same request. Only the
        # derived aspect band qualifies: it is reduced when the run is
        # multi-band (_build_image keeps every derived band, and a per-call
        # band list makes even the aspect band alone multi-band) or when it
        # is the single band.
        derived_bands = self._dataset_spec.get("derived_bands") or []
        derived_band_names = [derived_bands] if isinstance(derived_bands, str) else derived_bands
        aspect_is_reduced = "aspect" in derived_band_names and (multiband or band_name == "aspect")
        circular_reducer_names = [
            reducer_name for reducer_name in ("mean", "std", "var") if reducer_name in reducer_names
        ]
        want_circular_aspect = (
            combined_reducer is not None and aspect_is_reduced and bool(circular_reducer_names)
        )

        # Build up to three server-side reductions into a single ee.Dictionary.
        # GEE evaluates them in parallel and we pay one HTTP round-trip total.
        branches: dict[str, ee.Dictionary] = {}

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

        if want_circular_aspect:
            # Circular aspect branch: mean sine, mean cosine, and the number of
            # non-flat pixels, over the same region and grid as the window
            # branch. Output keys: sin_mean, sin_count, cos_mean, cos_count.
            # The same date and coordinates select the same image as above.
            circular_aspect_image, _ = self._get_image(date, lat=lat, lon=lon, circular_aspect=True)
            branches["aspect_circular"] = circular_aspect_image.reduceRegion(
                reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.count(), sharedInputs=True),
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

        if image_dates is not None:
            # Dates branch: system:time_start and system:time_end of the image
            # that the date selection chose, or null for a point where the
            # collection has no image. No extra request: it is a branch of
            # the same dictionary.
            branches["image_dates"] = image_dates

        # Single round-trip — all branches resolved server-side.
        full_result = ee.Dictionary(branches).getInfo() or {}

        # ---- parse window branch ----
        window_stats: dict[str, float | None] = {}
        valid_count = 0
        per_band_counts: dict[str, int] = {}
        if combined_reducer is not None:
            window_raw = full_result.get("window") or {}
            if multiband:
                window_stats = _parse_multiband_result(
                    window_raw, reducer_names, suffixes, count_only=count_only
                )
                # Collect per-band counts for individual coverage reporting.
                per_band_counts = _extract_per_band_counts(window_raw, count_only=count_only)
            else:
                window_stats = _parse_reduce_result(window_raw, band_name, reducer_names, suffixes)
            valid_count = _extract_count_from_reduce_result(
                window_raw, band_name, multiband, count_only=count_only
            )

            # ---- sum of a window without valid pixels ----
            # Earth Engine gives sum 0, not null, for a band whose window has
            # no valid pixel (seen live on 2026-10-09). A 0 would look like
            # data, and the local adapter gives no value for the sum of no
            # pixels. So the sum of a band with a pixel count of 0 becomes
            # null. count stays 0: it is the true number of valid pixels.
            if "sum" in reducer_names:
                if multiband:
                    for band, band_count in per_band_counts.items():
                        if band_count == 0 and f"{band}_sum" in window_stats:
                            window_stats[f"{band}_sum"] = None
                elif valid_count == 0:
                    window_stats["sum"] = None

            # ---- categorical post-process ----
            # We always ask GEE for `class_count` (frequencyHistogram) when
            # the user wants either class_count or class_fraction; fractions
            # are derived locally from the histogram's own per-band pixel
            # total. (Using the histogram's own sum — rather than a parallel
            # ee.Reducer.count() result — guarantees fractions sum to 1.0
            # per row by construction. The two can disagree on some GEE
            # datasets because the masks they honour aren't always identical.)
            user_wants_fraction = "class_fraction" in original_reducer_names
            user_wants_count = "class_count" in original_reducer_names
            if user_wants_fraction:
                # Group the just-parsed class_*_count entries by their prefix
                # (single-band → "", multi-band → "<band>_") and sum each
                # group to get the denominator for that band's fractions.
                class_totals_by_prefix: dict[str, int] = {}
                for stat_key, count_value in window_stats.items():
                    match = _CLASS_COUNT_KEY_RE.match(stat_key)
                    if match is None or count_value is None:
                        continue
                    prefix = match.group("prefix")
                    class_totals_by_prefix[prefix] = (
                        class_totals_by_prefix.get(prefix, 0) + count_value
                    )

                # Iterate a snapshot so adding fraction keys mid-loop is safe.
                for stat_key in list(window_stats):
                    match = _CLASS_COUNT_KEY_RE.match(stat_key)
                    if match is None:
                        continue
                    prefix = match.group("prefix")
                    class_value = match.group("value")
                    count_value = window_stats[stat_key]
                    if count_value is None:
                        continue
                    total = class_totals_by_prefix.get(prefix, 0)
                    if total > 0:
                        window_stats[f"{prefix}class_{class_value}_fraction"] = count_value / total

            # If the user asked only for class_fraction, drop the
            # class_*_count keys we generated internally to derive fractions.
            if not user_wants_count and user_wants_fraction:
                for stat_key in list(window_stats):
                    if _CLASS_COUNT_KEY_RE.match(stat_key) is not None:
                        del window_stats[stat_key]

        # ---- parse point branch ----
        point_stats: dict[str, float | None] = {}
        if want_point:
            point_raw = full_result.get("point") or {}
            point_stats = _parse_point_result(point_raw, band_name, multiband)

        # Merged stats dict — window keys first, then point keys.
        stats = {**window_stats, **point_stats}

        # ---- circular aspect statistics ----
        # Replace the linear aspect mean, std and var that the user asked for
        # with the circular values: "mean"/"std"/"var" when aspect is the
        # single band, "aspect_mean"/... when the run is multi-band (also for
        # a per-call band list of aspect alone). Only keys that exist are
        # replaced, so the set of columns does not change. The other
        # statistics of aspect stay linear. The QC values below read
        # window_stats, which keeps the linear values, so QC does not change.
        if want_circular_aspect:
            circular_raw = full_result.get("aspect_circular") or {}
            circular_values = _circular_aspect_statistics(
                circular_raw.get("sin_mean"),
                circular_raw.get("cos_mean"),
                circular_raw.get("sin_count"),
            )
            key_prefix = "aspect_" if multiband else ""
            for reducer_name in circular_reducer_names:
                stat_key = f"{key_prefix}{reducer_name}"
                if stat_key in stats:
                    stats[stat_key] = circular_values[reducer_name]

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
            # The window has data when at least one band has a valid pixel.
            # Use the pixel counts, not the statistics: on a window without
            # valid pixels, Earth Engine gives null for most statistics, but
            # 0 for count (and for sum, which the step above sets to null), so
            # a non-null statistic does not show data.
            # The window branch always has a count for each band: the QC
            # count, or the user's count.
            if multiband:
                has_values = any(count > 0 for count in per_band_counts.values())
            else:
                has_values = valid_count > 0
            n_pixels = int(valid_count)
            had_nodata = valid_count < total_cells
        else:
            # Point-only request — n_pixels is 1 if the sample returned a value, else 0.
            any_pt = any(v is not None for v in point_stats.values())
            n_pixels = 1 if any_pt else 0
            coverage_pct = 100.0 if any_pt else 0.0
            has_values = any_pt
            had_nodata = not any_pt

        # Date metadata: the dates of the image that the point used. Without
        # per-point date selection, the point has no date fields.
        date_info = (
            self._resolve_date_info(date, full_result.get("image_dates"))
            if image_dates is not None
            else {}
        )

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
            **date_info,
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
        disable_progress: bool = False,
        progress_callback: ProgressStepCallback | None = None,
    ) -> list[tuple[dict[str, float | None], dict]]:
        """Compute server-side statistics for many points in parallel (Mode 1).

        Each point is resolved with one server-side ``ee.Dictionary``
        containing up to three reductions: a window reduction (whenever
        any non-point reducer was requested), a circular aspect reduction
        (whenever the derived ``aspect`` band is reduced with ``mean``,
        ``std``, or ``var``), and a point reduction (whenever ``"point"``
        is in *reducer_names*). GEE evaluates the branches in parallel
        server-side and we pull them back together with one ``getInfo()``
        round-trip, so the cost is one HTTP call per point regardless of
        which combination the user requested. See ``_fetch_stats_single``.

        Output keys per point:
          - window stats: ``{reducer}`` (single band) or ``{band}_{reducer}`` (multi-band)
          - point sample: ``"point"``  (single band) or ``{band}_point``     (multi-band)
          A per-call band list gives the multi-band keys, also for one band.

        Returns a list of ``(stats_dict, meta_dict)`` tuples — one per point.
        A point whose request failed gets an empty stats dict, and its meta
        dict has the error text (``error``) and the exception (``exception``).
        See ``_empty_stats_result``.
        """
        # Split "point" out of reducer_names — it doesn't go through the
        # combined window reducer; it gets its own server-side branch.
        reducer_names = list(reducer_names)
        want_point = "point" in reducer_names
        # `original_window_reducers` keeps the user-requested list (incl. any
        # class_fraction) so _fetch_stats_single can decide what to derive
        # locally. `ee_window_reducers` is the deduped list fed to the
        # server-side combined reducer — class_fraction is merged into
        # class_count because they share frequencyHistogram in EE.
        original_window_reducers = [r for r in reducer_names if r != "point"]
        ee_window_reducers = _dedupe_categorical_for_ee(original_window_reducers)

        n = len(lats)
        lats = list(lats)
        lons = list(lons)
        date_list = list(dates) if dates is not None else [None] * n

        # A batch without points needs no request. Without this return, the
        # timestamp request below would have no points to filter the
        # collection by, and would ask for the dates of every image of the
        # collection, which can exceed Earth Engine's limits.
        if n == 0:
            emit_progress_step(progress_callback, 0, 0)
            return []

        # Fetch the collection's timestamp index now that we know the
        # batch's spatial extent and date range — without this the fetch
        # would happen unfiltered in __post_init__ and time out for very
        # large per-tile per-pass collections.
        self._ensure_collection_timestamps(lats, lons, date_list)

        # Build the window reducer once when window stats are requested;
        # leave it None for point-only runs.
        if ee_window_reducers:
            combined_reducer, suffixes = _build_combined_reducer(ee_window_reducers)
        else:
            combined_reducer, suffixes = None, []

        # Warm the band cache once on the main thread, so the workers find it
        # filled and send no band-name request of their own. Broad catch for a
        # harmless best-effort failure: a failed request is not cached, so each
        # worker then asks again, and a worker whose request also fails gives
        # a failed point.
        if not hasattr(self, "_cached_band_count"):
            try:
                batch_image, _ = self._get_image()
                self._get_band_name(batch_image)
            except Exception:
                pass

        # Single ThreadPool: each worker handles both branches for its point.
        # Progress bar advances as each per-point future completes — gives the
        # user visible feedback for what is otherwise a long, opaque GEE call.
        results: list = [None] * n
        emit_progress_step(progress_callback, 0, n)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._fetch_stats_single,
                    lat,
                    lon,
                    window_m,
                    combined_reducer,
                    ee_window_reducers,
                    suffixes,
                    want_point=want_point,
                    date=date,
                    original_reducer_names=original_window_reducers,
                ): i
                for i, (lat, lon, date) in enumerate(zip(lats, lons, date_list))
            }
            with tqdm(
                total=n,
                desc=progress_desc or "GEE stats",
                unit="pt",
                disable=disable_progress,
            ) as pbar:
                # `completed` counts finished points (1-based) for the progress
                # callback; futures arrive in completion order, not input order.
                for completed, future in enumerate(as_completed(future_to_idx), start=1):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    # Broad catch that replaces one point's failed result with
                    # an empty result (docs/coding_guidelines.md, "Failures for
                    # single points"): Earth Engine reports unrelated failures
                    # with one exception type. The result keeps the error, so
                    # extract() reports the point, and stops the run when every
                    # point failed.
                    except Exception as error:
                        logger.warning("GEE stats fetch failed for point %d: %s", idx, error)
                        results[idx] = self._empty_stats_result(window_m, error)
                    pbar.update(1)
                    emit_progress_step(progress_callback, completed, n)

        return results

    # ------------------------------------------------------------------
    # Mode 2: Image export  (ee.Image.getDownloadURL → GeoTIFF)
    # ------------------------------------------------------------------

    def _export_single(
        self,
        lat: float,
        lon: float,
        window_m: int,
        output_path: Path,
        date=None,
        resample_m: float | None = None,
    ) -> tuple[Path, dict]:
        """Export a GeoTIFF tile for one point, on a pixel grid in its UTM zone.

        The tile has n × n pixels, with n = :func:`~envoi.geo.tile_pixel_count`
        (``window_m`` / pixel size, rounded to the nearest whole number with
        ties rounded up, at least 1). The pixel size is ``resample_m`` when it
        is set, otherwise the image's native scale. The tile edges lie on
        multiples of the pixel size (:func:`~envoi.geo.tile_top_left`): for
        odd n, the pixel that contains the point is the centre pixel; for even
        n, the tile centre is the grid line nearest to the point. The local
        adapter uses the same grid when ``resample_m`` is set.

        For a collection with per-point date selection, one small request
        before the download returns the dates of the image that the point's
        selection chose (``image_dates`` of ``_build_image``). When this
        request fails, the error propagates, so the tile is a failed tile, and
        a tile with the same name from an earlier run is removed, as for a
        failed download.

        Returns:
            ``(output_path, date_info)``. ``date_info`` holds the point's date
            fields (see ``_resolve_date_info``). It is empty without per-point
            date selection, and for a point where the collection has no image.
        """
        output_path = Path(output_path)
        img, image_dates = self._get_image(date, lat=lat, lon=lon)

        # Dates of the selected image, so that the tile meta names the image
        # that the tile shows. The download request cannot return them.
        date_info: dict = {}
        if image_dates is not None:
            try:
                selected_image_times = image_dates.getInfo()
            # Broad catch that re-raises the error unchanged after a cleanup:
            # Earth Engine reports unrelated failures with one exception type.
            # export_tiles makes the tile a failed tile. Remove a tile with
            # the same name from an earlier run, so that the folder matches
            # the tile manifest, which marks the tile as not exported.
            except Exception:
                with contextlib.suppress(OSError):
                    output_path.unlink(missing_ok=True)
                raise
            date_info = self._resolve_date_info(date, selected_image_times)

        # Use resample_m as the export scale when provided; fall back to native.
        scale_m = float(resample_m) if resample_m is not None else self._get_scale_value(img)
        utm = get_utm_crs(lon, lat)

        # Project the point to its UTM zone (x first), then compute the tile's
        # pixel count and its corners on the pixel grid.
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
        point_x_m, point_y_m = transformer.transform(lon, lat)
        n_pixels = tile_pixel_count(window_m, scale_m)
        tile_left_m, tile_top_m = tile_top_left(point_x_m, point_y_m, n_pixels, scale_m)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Request the exact pixel grid (transform and dimensions), not a region
        # and a scale. With a region, Earth Engine adds every pixel that the
        # region touches, so rounding at the region edges can add a row or a
        # column (seen live on 2026-10-09 with the 30.92 m native scale of
        # the GLO-30 DEM: 7 × 7 pixels, or 7 rows of 6 pixels, instead of 6 × 6).
        # export_tiles has already checked the tile size once for the batch.
        _download_tile_via_url(
            img,
            crs=utm,
            crs_transform=[scale_m, 0.0, tile_left_m, 0.0, -scale_m, tile_top_m],
            dimensions=f"{n_pixels}x{n_pixels}",
            output_path=output_path,
        )
        return output_path, date_info

    # ------------------------------------------------------------------
    # Mode 2 helper: pre-flight size check
    # ------------------------------------------------------------------

    def _guard_tile_size(
        self,
        img: ee.Image,
        *,
        window_m: float,
        scale_m: float,
        dataset_name: str,
    ) -> None:
        """Raise ValueError if the requested tile would exceed GEE's sync limit.

        Thin wrapper around :func:`_check_tile_size` — it only exists to
        fetch the band count off the GEE image (which requires a live
        adapter), then hands the pure-Python size check to the module-level
        function so the latter stays unit-testable without GEE creds.

        ``export_tiles`` calls it once per batch, before any tile download.
        Every tile of a batch has the same size and the same bands.
        """
        # _get_band_name populates _cached_band_count as a side effect. It is
        # usually warm already (export_tiles primes it), but a failed request
        # is not cached, so this call asks Earth Engine again. If it fails
        # again, the exception propagates and stops the run: without the band
        # count, the size of every tile is unknown.
        self._get_band_name(img)
        band_count = getattr(self, "_cached_band_count", 1) or 1

        _check_tile_size(
            window_m=window_m,
            scale_m=scale_m,
            band_count=band_count,
            dataset_name=dataset_name,
        )

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
        disable_progress: bool = False,
        progress_callback: ProgressStepCallback | None = None,
    ) -> tuple[list[Path | None], list[dict]]:
        """Export GeoTIFF tiles for many points in parallel (Mode 2).

        Each tile has n × n pixels, where n is ``window_m`` divided by the
        pixel size, rounded to the nearest whole number with ties rounded up
        (at least 1). If resample_m is set, it is the pixel size, so every tile
        has the same size. For odd n, the pixel that contains the point is the
        centre pixel. See :meth:`_export_single` for the pixel grid.

        ``filename_suffix`` is inserted before the .tif extension so multi-
        window runs can place every window's tiles in the same folder
        without overwriting one another.

        Returns ``(paths, meta_list)`` — one entry per input point in each
        list. ``paths[i]`` is ``None`` when the export for that point failed.
        ``meta_list[i]`` holds the point's date information (the dates of the
        image in the tile, see :meth:`_export_single`). For a failed export, it
        holds only the error text (``error``) and the exception (``exception``).

        For collections with per-point date selection, each tile sends one
        small date request before its download, in its worker thread.

        Raises ``ValueError`` before any tile download when the tiles would
        exceed GEE's synchronous download limit. A failed request for the band
        names or the native scale also propagates before any download.
        """
        output_dir = Path(output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        n = len(lats)
        lats = list(lats)
        lons = list(lons)
        date_list = list(dates) if dates is not None else [None] * n
        id_list = list(ids) if ids is not None else [str(i) for i in range(n)]
        results: list = [None] * n

        # A batch without points needs no request (see fetch_stats_batch).
        if n == 0:
            emit_progress_step(progress_callback, 0, 0)
            return [], []

        # Fetch the collection's timestamp index now that we know the
        # batch's spatial extent and date range — see fetch_stats_batch
        # for why this is deferred from __post_init__ to here.
        self._ensure_collection_timestamps(lats, lons, date_list)

        # The image without point coordinates (see _get_image). The band-name
        # request and the tile-size check below use it.
        batch_image, _ = self._get_image()

        # Warm the band-name cache so build_dataset_meta can read it later.
        # Broad catch for a harmless best-effort failure: a failed request is
        # not cached, and the tile-size check below asks again.
        try:
            self._get_band_name(batch_image)
        except Exception:
            pass

        # Check the tile size once, before any download. Every tile of the
        # batch has the same size and bands, so an oversized request would
        # make every tile fail, and the run would end without tiles.
        # This needs the band count and the export scale. In a raster run, a
        # failed request for either one stops the run here: it affects every
        # tile. Do not catch the ValueError of the size check.
        export_scale_m = (
            float(resample_m) if resample_m is not None else self._get_scale_value(batch_image)
        )
        self._guard_tile_size(
            batch_image, window_m=window_m, scale_m=export_scale_m, dataset_name=dataset_name
        )

        # One meta dict per point. Each worker returns the date fields of its
        # tile, because only the point's own request knows which image the
        # date selection chose.
        meta_list: list[dict] = [{} for _ in range(n)]

        # Build every tile filename up front. Sample IDs come from the user's
        # input table and can contain anything (URL-style occurrenceIDs with
        # "/" in them, "urn:catalog:..." IDs with ":"), so they are sanitized
        # into a character set that is valid on Windows, macOS and Linux
        # alike. When the caller passes "200m", filenames become
        # "<id>-<dataset>-200m.tif"; with no suffix we keep the historical
        # "<id>-<dataset>.tif" naming. IDs that were already safe are left
        # untouched, unless two of them differ only in letter case (then both
        # get a hash suffix, so they cannot overwrite each other on Windows
        # and macOS).
        tile_filenames = build_tile_filenames(id_list, dataset_name, filename_suffix)

        emit_progress_step(progress_callback, 0, n)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for i, (lat, lon, date) in enumerate(zip(lats, lons, date_list)):
                output_path = output_dir / tile_filenames[i]
                future = executor.submit(
                    self._export_single, lat, lon, window_m, output_path, date, resample_m
                )
                future_to_idx[future] = i

            # Progress bar advances as each tile finishes downloading from GEE.
            with tqdm(
                total=n,
                desc=progress_desc or "GEE tiles",
                unit="tile",
                disable=disable_progress,
            ) as pbar:
                # `completed` counts finished points (1-based) for the progress
                # callback; futures arrive in completion order, not input order.
                for completed, future in enumerate(as_completed(future_to_idx), start=1):
                    idx = future_to_idx[future]
                    try:
                        results[idx], meta_list[idx] = future.result()
                    # Broad catch that replaces one point's failed tile with an
                    # empty result (no path; docs/coding_guidelines.md,
                    # "Failures for single points"): Earth Engine reports
                    # unrelated failures with one exception type. A failed
                    # date request or download comes here. The point's meta
                    # dict keeps the error, so extract() reports the tile, and
                    # stops the run when every tile failed.
                    except Exception as error:
                        logger.warning("GEE export failed for point %d: %s", idx, error)
                        results[idx] = None
                        meta_list[idx]["error"] = format_point_error(error)
                        meta_list[idx]["exception"] = error
                    pbar.update(1)
                    emit_progress_step(progress_callback, completed, n)

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
        meta: dict[str, Any] = {
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

        # Native CRS — prefer the projection resolved from the asset itself
        # (cached during __post_init__'s scale validation). Fall back to the
        # spec value (or the EPSG:4326 default) only when GEE couldn't
        # resolve the projection — in which case self.crs is the best we have.
        meta["native_crs"] = getattr(self, "_cached_native_crs", None) or str(self.crs)

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
        time_starts = getattr(self, "_collection_time_starts", None)
        if time_starts is not None and len(time_starts) > 0:
            meta["collection_date_range"] = [
                time_starts.min().strftime("%Y-%m-%d"),
                time_starts.max().strftime("%Y-%m-%d"),
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

        # Per-band coverage summary across all points, for multi-band datasets.
        # Reported as {band: {min, mean, max}} so users can spot bands with
        # systematically lower data density without inspecting the QC CSV.
        if meta_list:
            band_coverage_summary = _summarize_band_coverage(meta_list)
            if band_coverage_summary:
                meta["band_coverage_pct"] = band_coverage_summary

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
