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

from ..gee_features import build_image as _build_image

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
    - Feature-specific image builders via ``gee_features`` module.
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
            asset_info = ee.data.getAsset(asset_id)
            asset_type = asset_info.get("type")
            if asset_type == "IMAGE_COLLECTION":
                feature_spec = {"collection": asset_id}
            else:
                feature_spec = {"image": asset_id}
            if self.spec.get("band"):
                feature_spec["band"] = self.spec["band"]
            logger.debug("Auto-detected %s as %s", asset_id, asset_type)

        self.scale = self.spec.get("resolution_m", 250)
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

    def _make_region(self, lat: float, lon: float, window_m: int) -> ee.Geometry:
        """Build a meter-accurate square region using the point's UTM zone."""
        point = ee.Geometry.Point([lon, lat])
        if window_m <= 0:
            return point
        utm = _get_utm_crs(lon, lat)
        return point.buffer(window_m / 2, proj=ee.Projection(utm)).bounds()

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
        result = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=region,
            scale=self.scale,
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
        utm = _get_utm_crs(lon, lat)

        result = img.reduceRegion(
            reducer=combined_reducer,
            geometry=region,
            scale=self.scale,
            crs=utm,
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
                scale=self.scale,
                numPixels=1,
                dropNulls=False,
            ).first()

            props = sample.toDictionary().getInfo()
        except Exception:
            props = None

        if not props:
            return {}, {"in_extent": False, "src_path": self._src_label()}

        return props, {"in_extent": True, "src_path": self._src_label()}

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
    ):
        """Export a GeoTIFF tile for one point."""
        if geemap is None:
            raise ImportError("geemap is required for image export: pip install geemap")

        img = self._get_image(date)
        region = self._make_region(lat, lon, window_m)
        utm = _get_utm_crs(lon, lat)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        geemap.ee_export_image(
            img,
            filename=str(out_path),
            scale=self.scale,
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
    ) -> List[Path]:
        """Export GeoTIFF tiles for many points in parallel (Mode 4).

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
                    self._export_single, lat, lon, window_m, out_path, date
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
