# src/biodata/adapters/local_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import rasterio
from rasterio.features import geometry_window
from rasterio.warp import transform_geom
from rasterio.windows import transform as win_transform
from shapely.geometry import box, mapping
from pyproj import Transformer
from rasterio.errors import WindowError

from .base import BaseAdapter
from ..reducers import get_reducer

try:
    from . import register as _register
except Exception:
    _register = None


@dataclass
class LocalRasterAdapter(BaseAdapter):
    spec: Dict[str, Any]

    def __post_init__(self):
        self.path = Path(self.spec["path"])
        if not self.path.exists():
            raise FileNotFoundError(f"Raster not found: {self.path}")

        self.src = rasterio.open(self.path)
        self.raster_crs = self.src.crs

        # Band selection: 1-indexed, defaults to 1
        self.band = self.spec.get("bands", 1)

    @staticmethod
    def _get_utm_crs(lon: float, lat: float) -> str:
        """Return the EPSG code for the UTM zone covering (lon, lat)."""
        zone_number = int((lon + 180) / 6) + 1
        base_epsg = 32600 if lat >= 0 else 32700
        return f"EPSG:{base_epsg + zone_number}"

    def _project_meter_square_to_raster_geom(self, lat: float, lon: float, window_m: int):
        # Determine a metric CRS for building the square:
        # use the point's UTM zone for global flexibility
        metric_crs = self._get_utm_crs(lon, lat)
        to_metric = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
        cx, cy = to_metric.transform(lon, lat)
        # build a square in meters
        half = window_m / 2.0
        square_proj = box(cx - half, cy - half, cx + half, cy + half)
        # transform polygon to raster CRS
        square_raster_geojson = transform_geom(
            metric_crs, self.raster_crs, mapping(square_proj), precision=6
        )
        return square_raster_geojson

    def fetch_values(self, lat: float, lon: float, window_m: int, *, return_meta: bool = False):
        geom_raster = self._project_meter_square_to_raster_geom(lat, lon, window_m)
        try:
            win = geometry_window(
                self.src, [geom_raster], pad_x=0, pad_y=0, north_up=True, rotated=False
            )
        except (ValueError, WindowError):
            vals = np.array([])
            meta = {
                "in_extent": False,
                "n_pixels": 0,
                "had_nodata": False,
                "coverage_pct": 0.0,
                "window_m": int(window_m),
                "raster_crs": str(self.raster_crs),
                "region_crs": str(self.raster_crs),
                # JSON-safe placeholders for dump feature:
                "transform": None,
                "dtype": None,
                "nodata": None,
                "src_path": str(self.path),
                "window_arr": None,
            }
            return (vals, meta) if return_meta else vals

        arr = self.src.read(self.band, window=win, masked=True)
        # arr is 2D (H, W) for a single band int, 3D (n_bands, H, W) for a list

        if np.ma.isMaskedArray(arr):
            window_arr = arr.filled(self.src.nodata)
        else:
            window_arr = arr

        if arr.ndim == 3:
            # Combine masks: a pixel is valid only if unmasked AND finite in every band.
            # Per-band stats over one window must share a pixel set, else np.stack fails
            # when bands have different nodata patterns.
            if np.ma.isMaskedArray(arr):
                band_mask = np.ma.getmaskarray(arr)
                data = arr.filled(np.nan).astype(float, copy=False)
            else:
                band_mask = np.zeros(arr.shape, dtype=bool)
                data = np.asarray(arr, dtype=float)
            invalid = band_mask.any(axis=0) | (~np.isfinite(data)).any(axis=0)
            valid_mask_2d = ~invalid
            total = int(valid_mask_2d.size)
            valid = int(valid_mask_2d.sum())
            vals = np.stack([data[b][valid_mask_2d] for b in range(data.shape[0])], axis=0)
        else:
            total = arr.size
            valid = int(arr.count()) if np.ma.isMaskedArray(arr) else int(np.isfinite(arr).sum())
            vals = arr.compressed() if np.ma.isMaskedArray(arr) else arr.ravel()
            vals = vals[np.isfinite(vals)]

        had_nodata = bool(valid < total)
        coverage_pct = 100.0 * (valid / total) if total else 0.0

        # JSON-safe transform (list of 6 floats) to avoid breaking metadata JSON
        affine = win_transform(win, self.src.transform)
        transform_list = [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f]

        meta = {
            "in_extent": True,
            "n_pixels": int(total),
            "had_nodata": had_nodata,
            "coverage_pct": float(coverage_pct),
            "window_m": int(window_m),
            "raster_crs": str(self.raster_crs),
            "region_crs": str(self.raster_crs),
            # NEW: fields needed to write window tiles
            "transform": transform_list,  # JSON-safe
            "dtype": str(self.src.dtypes[0]),
            "nodata": self.src.nodata,
            "src_path": str(self.path),
            "window_arr": window_arr,  # for dump feature
        }
        return (np.asarray(vals), meta) if return_meta else np.asarray(vals)

    def export_windows(self, lats, lons, window_m: int, out_dir, *, ids=None, feature_name: str = "feature", resample_m: float | None = None):
        """Crop and save a GeoTIFF window centred on each point.

        If resample_m is set, the cropped window is resampled to
        round(window_m / resample_m) × round(window_m / resample_m) pixels
        so all tiles have identical dimensions regardless of native resolution.
        """
        from rasterio.transform import Affine
        from rasterio.warp import reproject, Resampling

        out_dir = Path(out_dir) / feature_name
        out_dir.mkdir(parents=True, exist_ok=True)

        id_list = list(ids) if ids is not None else [str(i) for i in range(len(list(lats)))]
        paths = []

        n_pixels = max(1, round(window_m / resample_m)) if resample_m is not None else None

        for lat, lon, sample_id in zip(lats, lons, id_list):
            out_path = out_dir / f"{sample_id}-{feature_name}.tif"
            _, meta = self.fetch_values(lat, lon, window_m, return_meta=True)

            arr2d = meta.get("window_arr")
            if arr2d is not None and arr2d.ndim == 3:
                arr2d = arr2d[0]  # raster export uses first band when multi-band
            transform_list = meta.get("transform")
            if arr2d is None or arr2d.size == 0 or transform_list is None:
                paths.append(None)
                continue

            src_transform = Affine(*transform_list)

            if n_pixels is not None:
                # Reproject/resample to a fixed n_pixels × n_pixels grid
                dst_arr = np.empty((n_pixels, n_pixels), dtype=np.float32)
                # Build a new transform with the same top-left corner but stretched pixels
                src_h, src_w = arr2d.shape
                dst_res_x = (src_transform.a * src_w) / n_pixels  # total width / n_pixels
                dst_res_y = (src_transform.e * src_h) / n_pixels  # total height / n_pixels (negative)
                dst_transform = Affine(dst_res_x, 0.0, src_transform.c,
                                       0.0, dst_res_y, src_transform.f)
                reproject(
                    source=arr2d.astype(np.float32),
                    destination=dst_arr,
                    src_transform=src_transform,
                    src_crs=self.raster_crs,
                    dst_transform=dst_transform,
                    dst_crs=self.raster_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=self.src.nodata,
                    dst_nodata=self.src.nodata,
                )
                out_arr = dst_arr
                out_transform = dst_transform
            else:
                out_arr = arr2d
                out_transform = src_transform

            profile = {
                "driver": "GTiff",
                "height": out_arr.shape[0],
                "width": out_arr.shape[1],
                "count": 1,
                "dtype": str(out_arr.dtype),
                "crs": self.raster_crs,
                "transform": out_transform,
                "nodata": self.src.nodata,
                "compress": "LZW",
            }
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_arr, 1)
            paths.append(out_path)

        return paths

    def fetch_stats_batch(self, lats, lons, window_m, reducer_names, *, dates=None):
        """Unified stats fetch: dispatches window reducers and 'point' to one call.

        Returns ``[(stats_dict, meta), ...]`` per input point. Keys in ``stats_dict``:
        single-band ``{rname}``, multi-band ``b{n}_{rname}``. Point values use the
        ``point`` suffix (``point`` or ``b{n}_point``).
        """
        reducer_names = list(reducer_names)
        want_point = "point" in reducer_names
        window_reducers = [r for r in reducer_names if r != "point"]

        if want_point and not hasattr(self, "fetch_points_batch"):
            raise ValueError(
                f"Adapter {type(self).__name__} does not support the 'point' reducer."
            )

        window_reducer_fns = [(r, get_reducer(r)) for r in window_reducers]

        lats = list(lats)
        lons = list(lons)
        n = len(lats)
        multiband = isinstance(self.band, list)

        window_results: list[tuple[dict, dict]] = []
        if window_reducers:
            for lat, lon in zip(lats, lons):
                vals, meta = self.fetch_values(lat, lon, window_m, return_meta=True)
                vals = np.asarray(vals)
                stats: dict[str, float | None] = {}
                if multiband and vals.ndim == 2:
                    for b_idx, band_num in enumerate(self.band):
                        for rname, fn in window_reducer_fns:
                            stats[f"b{band_num}_{rname}"] = (
                                fn(vals[b_idx]) if vals[b_idx].size else None
                            )
                else:
                    for rname, fn in window_reducer_fns:
                        stats[rname] = fn(vals) if vals.size else None
                window_results.append((stats, meta))

        point_results: list[tuple[dict, dict]] = []
        if want_point:
            pt_raw = self.fetch_points_batch(lats, lons, dates=dates)
            for pt_vals, pt_meta in pt_raw:
                stats = {}
                if multiband:
                    for band_num in self.band:
                        stats[f"b{band_num}_point"] = pt_vals.get(f"b{band_num}")
                else:
                    stats["point"] = pt_vals.get("point")
                point_results.append((stats, pt_meta))

        results: list[tuple[dict, dict]] = []
        for i in range(n):
            merged_stats: dict = {}
            if window_reducers:
                merged_stats.update(window_results[i][0])
            if want_point:
                merged_stats.update(point_results[i][0])

            if window_reducers:
                meta = window_results[i][1]
            else:
                pt_meta = point_results[i][1]
                meta = {
                    **pt_meta,
                    "window_m": int(window_m),
                    "raster_crs": str(self.raster_crs),
                    "region_crs": str(self.raster_crs),
                    "transform": None,
                    "dtype": None,
                    "nodata": None,
                    "window_arr": None,
                }
            results.append((merged_stats, meta))

        return results

    def fetch_points_batch(self, lats, lons, *, dates=None):
        """Sample the exact pixel value at each (lat, lon) coordinate."""
        transformer = Transformer.from_crs("EPSG:4326", self.raster_crs, always_xy=True)
        results = []
        multiband = isinstance(self.band, list)
        for lat, lon in zip(lats, lons):
            try:
                x, y = transformer.transform(lon, lat)
                raw = next(self.src.sample([(x, y)], indexes=self.band))
                nodata = self.src.nodata
                if multiband:
                    values = {}
                    for band_num, v in zip(self.band, raw):
                        fv = None if (nodata is not None and v == nodata) else float(v)
                        values[f"b{band_num}"] = fv
                    any_valid = any(v is not None for v in values.values())
                else:
                    v = raw[0]
                    fv = None if (nodata is not None and v == nodata) else float(v)
                    values = {"point": fv}
                    any_valid = fv is not None
                meta = {
                    "in_extent": any_valid,
                    "n_pixels": 1 if any_valid else 0,
                    "had_nodata": not any_valid,
                    "coverage_pct": 100.0 if any_valid else 0.0,
                    "src_path": str(self.path),
                }
            except Exception:
                values = {f"b{b}": None for b in self.band} if multiband else {"point": None}
                meta = {
                    "in_extent": False, "n_pixels": 0,
                    "had_nodata": False, "coverage_pct": 0.0,
                    "src_path": str(self.path),
                }
            results.append((values, meta))
        return results


if _register is not None:
    _register("local", LocalRasterAdapter)
