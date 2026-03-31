# src/biodata/adapters/gee_raster.py
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Sequence

import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    ee = None

try:
    import geemap
except ImportError:
    geemap = None

try:
    from . import register as _register
except Exception:
    _register = None

logger = logging.getLogger(__name__)

_gee_initialized = False


# ---------------------------------------------------------------------------
# GEE initialisation
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
# UTM helpers
# ---------------------------------------------------------------------------


def _get_utm_crs(lon: float, lat: float) -> str:
    """Return the EPSG code for the UTM zone covering (lon, lat)."""
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"Invalid WGS84 coordinates: ({lon}, {lat})")
    zone_number = int((lon + 180) / 6) + 1
    base_epsg = 32600 if lat >= 0 else 32700
    return f"EPSG:{base_epsg + zone_number}"


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
# Image building (collection reduction, date filtering, cloud masking, derived bands)
# ---------------------------------------------------------------------------

_COLLECTION_REDUCERS = {
    "mean":   lambda col: col.mean(),
    "median": lambda col: col.median(),
    "mode":   lambda col: col.mode(),
    "mosaic": lambda col: col.mosaic(),
    "min":    lambda col: col.min(),
    "max":    lambda col: col.max(),
    "sum":    lambda col: col.sum(),
    "first":  lambda col: col.first(),
}


def _reduce_collection(col, reducer: str):
    """Reduce an ImageCollection to a single Image using *reducer* name."""
    fn = _COLLECTION_REDUCERS.get(reducer)
    if fn is None:
        logger.warning("Unknown collection reducer '%s', falling back to mean", reducer)
        return col.mean()
    return fn(col)


def _filter_collection_by_date(col, date, feature_spec: dict):
    """Filter a collection around a sample date using feature_spec config."""
    start_date = feature_spec.get("start_date")
    end_date = feature_spec.get("end_date")

    if start_date and end_date:
        return col.filterDate(start_date, end_date)

    days = feature_spec.get("temporal_window_days")
    if days and date is not None:
        half = (days - 1) / 2
        start = (date - pd.DateOffset(days=half)).strftime("%Y-%m-%d")
        end = (date + pd.DateOffset(days=half)).strftime("%Y-%m-%d")
        return col.filterDate(start, end)

    return col


def _mask_clouds_s2(image):
    """Mask clouds and cirrus for Sentinel-2 using QA60 band."""
    qa = image.select("QA60")
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
    mask = cloud_mask.And(cirrus_mask)
    return (
        image.updateMask(mask)
        .divide(10000)
        .select("B.*")
        .copyProperties(image, ["system:time_start"])
    )


_CLOUD_MASK_FNS = {
    "s2": _mask_clouds_s2,
}


def _apply_cloud_mask(col, mask_type: str):
    """Apply a cloud mask function to a collection, if mask_type is known."""
    fn = _CLOUD_MASK_FNS.get(mask_type)
    if fn is None:
        logger.warning("Unknown cloud_mask type '%s', skipping", mask_type)
        return col
    return col.map(fn)


def _apply_derived_band(img, derived: str):
    """Compute a derived band from an image (NDVI, EVI, slope, aspect)."""
    if derived == "NDVI":
        return img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    if derived == "EVI":
        return img.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
        ).rename("EVI")
    if derived == "slope":
        return ee.Terrain.slope(img)
    if derived == "aspect":
        return ee.Terrain.aspect(img)
    logger.warning("Unknown derived_band '%s', returning image unchanged", derived)
    return img


def _build_image(feature_spec: dict, date=None):
    """Build an ee.Image from a feature_spec config dict.

    Pipeline: load image/collection → date filter → cloud_pct filter →
              cloud mask → reduce → band select → derived band.
    """
    img = None

    if "image" in feature_spec:
        img = ee.Image(feature_spec["image"])

    elif "collection" in feature_spec:
        col = ee.ImageCollection(feature_spec["collection"])

        if date is not None:
            col = _filter_collection_by_date(col, date, feature_spec)
        elif feature_spec.get("start_date") and feature_spec.get("end_date"):
            col = col.filterDate(feature_spec["start_date"], feature_spec["end_date"])

        cloud_pct = feature_spec.get("cloud_pct_max")
        if cloud_pct is not None:
            col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))

        cloud_mask = feature_spec.get("cloud_mask")
        if cloud_mask:
            col = _apply_cloud_mask(col, cloud_mask)

        reducer = feature_spec.get("collection_reducer", "mean")
        img = _reduce_collection(col, reducer)

    else:
        raise ValueError("feature_spec must contain 'image' or 'collection'")

    band = feature_spec.get("band")
    if band is not None:
        img = img.select(band)

    derived = feature_spec.get("derived_band")
    if derived:
        img = _apply_derived_band(img, derived)

    return img


# ---------------------------------------------------------------------------
# EE Reducer helpers  (combined reducer pattern)
# ---------------------------------------------------------------------------

# Maps EDDP/user-facing reducer names to GEE reducer constructors
# and the suffix GEE appends to band names in reduceRegion output.
_GEE_REDUCER_MAP = {
    "mean":   ("mean",     "_mean"),
    "median": ("median",   "_median"),
    "mode":   ("mode",     "_mode"),
    "std":    ("stdDev",   "_stdDev"),
    "var":    ("variance", "_variance"),
    "min":    ("min",      "_min"),
    "max":    ("max",      "_max"),
    "count":  ("count",    "_count"),
    "sum":    ("sum",      "_sum"),
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

    return combined.unweighted(), suffixes


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


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class GeeRasterAdapter:
    """Adapter that samples data directly from Google Earth Engine.

    Three extraction modes:

    1. **fetch_batch / fetch_values** — raw pixel arrays via ``sampleRectangle``
       (feeds into Python-side reducers, same as LocalRasterAdapter).
    2. **fetch_stats_batch** — server-side statistics via ``reduceRegion``
       with combined reducers.  Much faster for the common stats use-case.
    3. **fetch_points_batch** — single pixel values via ``image.sample()``.
    4. **export_images** — download full GeoTIFF tiles via ``geemap``.

    Additional capabilities:
    - Dynamic UTM zone per point for accurate meter-based windows.
    - Per-point temporal filtering for ImageCollections.
    - Generic image building pipeline via module-level ``_build_image``.
    """

    spec: Dict[str, Any]
    _static_image: Any = field(default=None, init=False, repr=False)
    _needs_per_point_date: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if ee is None:
            raise ImportError(
                "earthengine-api is required for GEE adapter: "
                "pip install earthengine-api"
            )

        _ensure_gee_init()

        # feature_spec holds extra config (bands, date windows, derivatives, etc.)
        # If absent, auto-detect asset type from GEE using ee.data.getAsset().
        feature_spec = self.spec.get("feature_spec") or {}
        if not feature_spec and self.spec.get("path"):
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
                feature_spec = {"collection": asset_id}
            else:
                feature_spec = {"image": asset_id}
            if self.spec.get("band"):
                feature_spec["band"] = self.spec["band"]
            logger.debug("Auto-detected %s as %s", asset_id, asset_type)

        self.scale = self.spec.get("resolution_m")  # None → use native scale
        self.crs = self.spec.get("crs", "EPSG:4326")
        self.max_workers = self.spec.get("max_workers", 8)
        self._feature_spec = feature_spec

        # Determine if we can pre-build a static image
        is_collection = "collection" in feature_spec
        has_global_dates = bool(feature_spec.get("start_date") and feature_spec.get("end_date"))

        if not is_collection or has_global_dates:
            self._static_image = _build_image(feature_spec)
            self._needs_per_point_date = False
        else:
            self._static_image = None
            self._needs_per_point_date = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _src_label(self) -> str:
        cfg = self._feature_spec
        return f"gee://{cfg.get('image', cfg.get('collection', 'unknown'))}"

    def _get_image(self, date=None) -> ee.Image:
        """Return the ee.Image to sample, building per-point if needed."""
        if self._static_image is not None:
            return self._static_image
        dt = pd.to_datetime(date) if date is not None else None
        return _build_image(self._feature_spec, dt)

    def _get_band_name(self, img: ee.Image) -> str:
        """Get the first band name from the image (needed for result parsing)."""
        band = self._feature_spec.get("band")
        if band:
            return band
        derived = self._feature_spec.get("derived_band")
        if derived:
            return derived
        if self._feature_spec.get("derivative") == "slope":
            return "slope"
        if self._feature_spec.get("derivative") == "aspect":
            return "aspect"
        # Fallback: ask GEE (costs one getInfo call, cached after first use)
        if not hasattr(self, "_cached_band_name"):
            try:
                names = img.bandNames().getInfo()
                self._cached_band_name = names[0] if names else "value"
            except Exception:
                self._cached_band_name = "value"
        return self._cached_band_name

    def _get_scale(self, img):
        """Return user-specified scale or fall back to the image's native scale."""
        if self.scale is not None:
            return self.scale
        return img.projection().nominalScale()

    def _get_scale_value(self, img) -> float:
        """Return scale as a plain Python float (metres). Fetches from GEE once and caches."""
        if self.scale is not None:
            return float(self.scale)
        if not hasattr(self, "_cached_native_scale"):
            self._cached_native_scale = float(img.projection().nominalScale().getInfo())
        return self._cached_native_scale

    def _make_region(self, lat: float, lon: float, window_m: int) -> ee.Geometry:
        """Build a meter-accurate square region using the point's UTM zone."""
        point = ee.Geometry.Point([lon, lat])
        if window_m <= 0:
            return point
        utm = _get_utm_crs(lon, lat)
        return point.buffer(window_m / 2, proj=ee.Projection(utm)).bounds(maxError=1)

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

    def _empty_stats_result(self, window_m: int, reducer_names: Sequence[str]):
        stats = {r: None for r in reducer_names}
        _, meta = self._empty_result(window_m)
        return stats, meta

    # ------------------------------------------------------------------
    # Mode 1: Raw pixel arrays  (sampleRectangle)
    # ------------------------------------------------------------------

    def _fetch_single(self, lat: float, lon: float, window_m: int, date=None):
        """Core pixel-array fetch for one point."""
        img = self._get_image(date)
        region = self._make_region(lat, lon, window_m)

        if window_m <= 0:
            return self._sample_pixel(img, region, window_m)
        return self._sample_window(img, region, window_m)

    def _sample_pixel(self, img, region, window_m):
        proj = img.projection()
        result = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=region,
            scale=self._get_scale(img),
            crs=proj,
        ).getInfo()

        if not result:
            return self._empty_result(window_m)

        val = next((v for v in result.values() if v is not None), None)
        if val is None:
            return self._empty_result(window_m)

        vals = np.array([val], dtype=np.float64)
        meta = {
            "in_extent": True,
            "n_pixels": 1,
            "had_nodata": False,
            "coverage_pct": 100.0,
            "window_m": int(window_m),
            "raster_crs": self.crs,
            "transform": None,
            "dtype": "float64",
            "nodata": None,
            "src_path": self._src_label(),
            "window_arr": vals.reshape(1, 1),
        }
        return vals, meta

    def _sample_window(self, img, region, window_m):
        result = img.sampleRectangle(
            region=region,
            defaultValue=-9999,
        ).getInfo()

        if not result or "properties" not in result:
            return self._empty_result(window_m)

        props = result["properties"]
        band_data = next(iter(props.values()), None) if props else None
        if band_data is None:
            return self._empty_result(window_m)

        arr_2d = np.array(band_data, dtype=np.float64)
        arr_2d[arr_2d == -9999] = np.nan

        flat = arr_2d.ravel()
        valid_mask = np.isfinite(flat)
        vals = flat[valid_mask]
        total = flat.size
        valid_count = int(valid_mask.sum())

        meta = {
            "in_extent": True,
            "n_pixels": int(total),
            "had_nodata": valid_count < total,
            "coverage_pct": 100.0 * (valid_count / total) if total else 0.0,
            "window_m": int(window_m),
            "raster_crs": self.crs,
            "transform": None,
            "dtype": "float64",
            "nodata": None,
            "src_path": self._src_label(),
            "window_arr": arr_2d,
        }
        return vals, meta

    # ------------------------------------------------------------------
    # Mode 2: Server-side stats  (reduceRegion with combined reducers)
    # ------------------------------------------------------------------

    def _fetch_stats_single(
        self,
        lat: float,
        lon: float,
        window_m: int,
        combined_reducer: ee.Reducer,
        reducer_names: Sequence[str],
        suffixes: list[str],
        date=None,
    ):
        """Compute server-side stats for a single point via reduceRegion."""
        img = self._get_image(date)
        band_name = self._get_band_name(img)
        region = self._make_region(lat, lon, window_m)

        proj = img.projection()
        result = img.reduceRegion(
            reducer=combined_reducer,
            geometry=region,
            scale=proj.nominalScale(),
            crs=proj,
            bestEffort=True,
        ).getInfo()

        stats = _parse_reduce_result(result, band_name, reducer_names, suffixes)

        # Build QC meta from a count reducer if available, else approximate
        n_pixels = None
        if "count" in stats and stats["count"] is not None:
            n_pixels = int(stats["count"])

        has_values = any(v is not None for v in stats.values())
        meta = {
            "in_extent": has_values,
            "n_pixels": n_pixels or (1 if has_values else 0),
            "had_nodata": False,
            "coverage_pct": 100.0 if has_values else 0.0,
            "window_m": int(window_m),
            "raster_crs": self.crs,
            "transform": None,
            "dtype": "float64",
            "nodata": None,
            "src_path": self._src_label(),
            "window_arr": None,
        }
        return stats, meta

    # ------------------------------------------------------------------
    # Mode 3: Single pixel values  (image.sample)
    # ------------------------------------------------------------------

    def _fetch_point_single(self, lat: float, lon: float, date=None):
        """Sample a single pixel value at exact (lat, lon)."""
        img = self._get_image(date)
        point = ee.Geometry.Point([lon, lat])

        try:
            sample = img.sample(
                region=point,
                scale=self._get_scale(img),
                numPixels=1,
                dropNulls=False,
            ).first()

            props = sample.toDictionary().getInfo()
        except Exception:
            props = None

        if not props:
            return {}, {
                "in_extent": False, "n_pixels": 0,
                "had_nodata": False, "coverage_pct": 0.0,
                "src_path": self._src_label(),
            }

        return props, {
            "in_extent": True, "n_pixels": 1,
            "had_nodata": False, "coverage_pct": 100.0,
            "src_path": self._src_label(),
        }

    # ------------------------------------------------------------------
    # Mode 4: Image export  (geemap → GeoTIFF)
    # ------------------------------------------------------------------

    def _export_single(
        self,
        lat: float,
        lon: float,
        window_m: int,
        out_path: Path,
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
        if geemap is None:
            raise ImportError("geemap is required for image export: pip install geemap")

        img = self._get_image(date)
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

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        geemap.ee_export_image(
            img,
            filename=str(out_path),
            scale=scale_m,
            region=region,
            crs=utm,
        )
        return out_path

    # ==================================================================
    # Public interface
    # ==================================================================

    def fetch_values(
        self,
        lat: float,
        lon: float,
        window_m: int,
        *,
        return_meta: bool = False,
    ):
        """Sample raw pixel values (Mode 1).  Compatible with LocalRasterAdapter."""
        try:
            vals, meta = self._fetch_single(lat, lon, window_m)
        except Exception as e:
            logger.warning("GEE fetch failed for (%.4f, %.4f): %s", lat, lon, e)
            vals, meta = self._empty_result(window_m)
        return (vals, meta) if return_meta else vals

    def fetch_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        *,
        dates: Sequence | None = None,
        return_meta: bool = False,
    ) -> List:
        """Fetch raw pixel arrays for many points in parallel (Mode 1)."""
        n = len(lats)
        date_list = list(dates) if dates is not None else [None] * n
        results: List = [None] * n

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._fetch_single, lat, lon, window_m, date): i
                for i, (lat, lon, date) in enumerate(zip(lats, lons, date_list))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning("GEE batch fetch failed for point %d: %s", idx, e)
                    results[idx] = self._empty_result(window_m)

        if return_meta:
            return results
        return [r[0] for r in results]

    def fetch_stats_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        reducer_names: Sequence[str],
        *,
        dates: Sequence | None = None,
    ) -> List[tuple[dict[str, float | None], dict]]:
        """Compute server-side statistics for many points in parallel (Mode 2).

        Uses a single combined ``reduceRegion`` call per point with all
        requested reducers, avoiding the need to download raw pixel arrays.

        Returns a list of ``(stats_dict, meta_dict)`` tuples — one per point.
        ``stats_dict`` maps each reducer name to its computed value.
        """
        combined_reducer, suffixes = _build_combined_reducer(reducer_names)

        n = len(lats)
        date_list = list(dates) if dates is not None else [None] * n
        results: List = [None] * n

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._fetch_stats_single,
                    lat, lon, window_m,
                    combined_reducer, reducer_names, suffixes,
                    date,
                ): i
                for i, (lat, lon, date) in enumerate(zip(lats, lons, date_list))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning("GEE stats fetch failed for point %d: %s", idx, e)
                    results[idx] = self._empty_stats_result(window_m, reducer_names)

        return results

    def fetch_points_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        *,
        dates: Sequence | None = None,
    ) -> List[tuple[dict, dict]]:
        """Sample single pixel values for many points in parallel (Mode 3).

        Returns a list of ``(values_dict, meta_dict)`` tuples.
        ``values_dict`` maps band names to their sampled values.
        """
        n = len(lats)
        date_list = list(dates) if dates is not None else [None] * n
        results: List = [None] * n

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._fetch_point_single, lat, lon, date): i
                for i, (lat, lon, date) in enumerate(zip(lats, lons, date_list))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning("GEE point fetch failed for point %d: %s", idx, e)
                    results[idx] = ({}, {"in_extent": False, "src_path": self._src_label()})

        return results

    def export_images(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        out_dir: str | Path,
        *,
        ids: Sequence[str] | None = None,
        dates: Sequence | None = None,
        feature_name: str = "feature",
        resample_m: float | None = None,
    ) -> List[Path]:
        """Export GeoTIFF tiles for many points in parallel (Mode 4).

        If resample_m is set, all tiles are exported at that resolution so they
        are exactly round(window_m / resample_m) × round(window_m / resample_m) pixels.

        Returns list of output file paths.
        """
        out_dir = Path(out_dir) / feature_name
        out_dir.mkdir(parents=True, exist_ok=True)

        n = len(lats)
        date_list = list(dates) if dates is not None else [None] * n
        id_list = list(ids) if ids is not None else [str(i) for i in range(n)]
        results: List = [None] * n

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for i, (lat, lon, date, sample_id) in enumerate(
                zip(lats, lons, date_list, id_list)
            ):
                out_path = out_dir / f"{sample_id}-{feature_name}.tif"
                future = executor.submit(
                    self._export_single, lat, lon, window_m, out_path, date, resample_m
                )
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning("GEE export failed for point %d: %s", idx, e)
                    results[idx] = None

        return results


if _register is not None:
    _register("gee_raster", GeeRasterAdapter)
