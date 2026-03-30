# src/biodata/adapters/local_raster.py
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
        self.band = self.spec.get("band", 1)

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
                # JSON-safe placeholders for dump feature:
                "transform": None,
                "dtype": None,
                "nodata": None,
                "src_path": str(self.path),
                "window_arr": None,
            }
            return (vals, meta) if return_meta else vals

        arr = self.src.read(self.band, window=win, masked=True)

        if np.ma.isMaskedArray(arr):
            window_arr = arr.filled(self.src.nodata)
        else:
            window_arr = arr

        total = arr.size
        valid = np.count_nonzero(~arr.mask) if np.ma.isMaskedArray(arr) else np.isfinite(arr).sum()
        had_nodata = bool(valid < total)
        vals = arr.compressed() if np.ma.isMaskedArray(arr) else arr.ravel()
        vals = vals[np.isfinite(vals)]
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
            # NEW: fields needed to write window tiles
            "transform": transform_list,  # JSON-safe
            "dtype": str(self.src.dtypes[0]),
            "nodata": self.src.nodata,
            "src_path": str(self.path),
            "window_arr": window_arr,  # for dump feature
        }
        return (np.asarray(vals), meta) if return_meta else np.asarray(vals)


if _register is not None:
    _register("local_raster", LocalRasterAdapter)
