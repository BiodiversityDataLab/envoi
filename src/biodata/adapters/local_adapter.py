# src/biodata/adapters/local_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Sequence

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping
from pyproj import Transformer
from rasterio.errors import WindowError
from tqdm.auto import tqdm

from .base import BaseAdapter
from ..reducers import get_reducer
from ..metadata import summarize_tile_export

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

        # Determine which bands to read. If the user specifies "bands" in the catalog
        # (a single int or list of ints), use that. Otherwise default to all bands in
        # the file so no data is silently dropped.
        bands_spec = self.spec.get("bands")
        if bands_spec is None:
            all_bands = list(range(1, self.src.count + 1))
            # A single-band raster is treated as a scalar (int) so downstream
            # code keeps the simpler single-band path and column naming.
            self.band = all_bands[0] if len(all_bands) == 1 else all_bands
        else:
            self.band = bands_spec

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
            # all_touched=False (default) uses center-in-polygon: only pixels
            # whose centres fall inside the polygon are included. This avoids
            # the bounding-box overshoot of geometry_window and matches GEE's
            # reduceRegion pixel-selection rule.
            arr, window_affine = rio_mask(
                self.src,
                [geom_raster],
                crop=True,
                all_touched=False,
                filled=False,
                indexes=self.band,
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

        # arr is 2D (H, W) for a single band int, 3D (n_bands, H, W) for a list

        if np.ma.isMaskedArray(arr):
            window_arr = arr.filled(self.src.nodata)
        else:
            window_arr = arr

        if arr.ndim == 3:
            # Multi-band case. We must reduce every band over the SAME set of
            # pixels, otherwise `np.stack` below would fail when bands have
            # different nodata footprints (e.g. one band has a cloud mask that
            # another doesn't). So: a pixel is considered valid only if it is
            # unmasked AND finite in EVERY band.
            if np.ma.isMaskedArray(arr):
                band_mask = np.ma.getmaskarray(arr)  # True where masked
                data = arr.filled(np.nan).astype(float, copy=False)
            else:
                band_mask = np.zeros(arr.shape, dtype=bool)
                data = np.asarray(arr, dtype=float)
            # invalid[row, col] is True if ANY band is masked or non-finite there.
            invalid = band_mask.any(axis=0) | (~np.isfinite(data)).any(axis=0)
            valid_mask_2d = ~invalid
            total = int(valid_mask_2d.size)
            valid = int(valid_mask_2d.sum())
            # Shape (n_bands, n_valid_pixels) — one column per band per reducer.
            vals = np.stack([data[b][valid_mask_2d] for b in range(data.shape[0])], axis=0)
        else:
            # Single-band: straightforward — drop masked and non-finite pixels.
            total = arr.size
            valid = int(arr.count()) if np.ma.isMaskedArray(arr) else int(np.isfinite(arr).sum())
            vals = arr.compressed() if np.ma.isMaskedArray(arr) else arr.ravel()
            vals = vals[np.isfinite(vals)]

        had_nodata = bool(valid < total)
        coverage_pct = 100.0 * (valid / total) if total else 0.0

        # JSON-safe transform (list of 6 floats) to avoid breaking metadata JSON
        # window_affine is returned directly by rio_mask, no win needed.
        transform_list = [
            window_affine.a,
            window_affine.b,
            window_affine.c,
            window_affine.d,
            window_affine.e,
            window_affine.f,
        ]

        meta = {
            "in_extent": True,
            # n_pixels reports valid (non-nodata, finite) cells only — total
            # cells can be derived from window_m / native pixel size.
            "n_pixels": int(valid),
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

    def fetch_stats_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        reducer_names: Sequence[str],
        *,
        dates: Sequence | None = None,
        progress_desc: str | None = None,
    ) -> List[tuple[dict, dict]]:
        """Unified stats fetch: dispatches window reducers and the "point" reducer.

        ``reducer_names`` may contain any mix of window reducers (e.g. "mean",
        "std") and the special ``"point"`` reducer, which samples the exact
        pixel value at the coordinate. Window reducers run over a square window
        of size ``window_m``; "point" is independent of ``window_m``.

        Returns ``[(stats_dict, meta), ...]`` — one tuple per input point.
        Keys in ``stats_dict``:
          * single-band window: ``{rname}``
          * multi-band window:  ``b{band_num}_{rname}``
          * single-band point:  ``point``
          * multi-band point:   ``b{band_num}_point``

        Mirrors the GEE adapter's design: a single per-point worker handles
        both the window and point branches, so callers see one consistent
        merged ``(stats, meta)`` tuple per input row.
        """
        # Separate "point" from ordinary window reducers — each has its own
        # internal branch inside _fetch_stats_single.
        reducer_names = list(reducer_names)
        want_point = "point" in reducer_names
        window_reducers = [r for r in reducer_names if r != "point"]

        # Pre-resolve window reducer callables once so we don't re-look them
        # up by name on every point.
        window_reducer_fns = [(r, get_reducer(r)) for r in window_reducers]

        # Single per-point loop — _fetch_stats_single returns the merged
        # (stats, meta) tuple, so no separate merge pass is needed.
        # The tqdm wrapper gives users per-point progress feedback, mirroring
        # the GEE adapter's progress UI so behaviour is consistent across sources.
        total_points = len(lats)
        results: List[tuple[dict, dict]] = []
        for lat, lon in tqdm(
            zip(lats, lons),
            total=total_points,
            desc=progress_desc or "Local stats",
            unit="pt",
        ):
            results.append(
                self._fetch_stats_single(
                    lat, lon, window_m, window_reducer_fns, want_point=want_point
                )
            )
        return results

    def _fetch_stats_single(
        self,
        lat: float,
        lon: float,
        window_m: int,
        window_reducer_fns: list[tuple[str, Any]],
        *,
        want_point: bool = False,
    ) -> tuple[dict, dict]:
        """Compute window stats and/or point sample at one (lat, lon).

        Two branches share the same rasterio dataset and run sequentially:

        - **window** — runs only when *window_reducer_fns* is non-empty.
          Reads the buffered window via :meth:`fetch_values` and applies
          each Python-side reducer to the valid pixels.
        - **point** — runs only when *want_point* is True. Samples the
          exact pixel containing (lat, lon) via ``self.src.sample()``.

        When both branches run, the window branch's meta is used (it carries
        transform/dtype/nodata fields needed by tile export). When only the
        point branch runs, the meta is synthesized with placeholder transform
        fields so callers downstream see a consistent shape.
        """
        is_multiband = isinstance(self.band, (list, tuple))
        band_nums = self.band if is_multiband else None

        stats: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}

        # ---- window branch ----
        if window_reducer_fns:
            vals, meta = self.fetch_values(lat, lon, window_m, return_meta=True)
            vals = np.asarray(vals)
            if is_multiband and vals.ndim == 2:
                # vals shape: (n_bands, n_valid_pixels)
                for b_idx, band_num in enumerate(band_nums):
                    v = vals[b_idx]
                    for rname, reducer in window_reducer_fns:
                        stats[f"b{band_num}_{rname}"] = reducer(v) if v.size else None
            else:
                for rname, reducer in window_reducer_fns:
                    stats[rname] = reducer(vals) if vals.size else None

        # ---- point branch ----
        if want_point:
            pt_values, pt_meta = self._sample_point_single(lat, lon)
            if is_multiband:
                # _sample_point_single returns "b{n}" keys — re-key with the
                # "_point" suffix so column names are unambiguous downstream.
                for band_num in band_nums:
                    stats[f"b{band_num}_point"] = pt_values.get(f"b{band_num}")
            else:
                stats["point"] = pt_values.get("point")

            # If only the point branch ran, synthesize the meta fields that
            # the window branch would normally have provided.
            if not window_reducer_fns:
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

        return stats, meta

    def _sample_point_single(self, lat: float, lon: float) -> tuple[dict, dict]:
        """Sample the exact pixel value at one (lat, lon) via rasterio.

        Returns ``(values, meta)`` where:
          * ``values`` is keyed ``"point"`` for single-band or ``"b{n}"``
            per band for multi-band — callers re-key with the ``_point``
            suffix as needed.
          * ``meta`` carries the per-point QC fields (in_extent, n_pixels,
            had_nodata, coverage_pct, src_path).
        """
        # Cache the WGS84 → raster-CRS transformer on first use; it's the
        # same for every point, so building it once amortises pyproj setup.
        if not hasattr(self, "_cached_raster_transformer"):
            self._cached_raster_transformer = Transformer.from_crs(
                "EPSG:4326", self.raster_crs, always_xy=True
            )
        transformer = self._cached_raster_transformer

        multiband = isinstance(self.band, list)
        try:
            x, y = transformer.transform(lon, lat)
            raw = next(self.src.sample([(x, y)], indexes=self.band))
            nodata = self.src.nodata
            if multiband:
                values: Dict[str, Any] = {}
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
            # Out-of-extent points or any rasterio sampling failure — return
            # None values so downstream code still sees a consistent schema.
            values = {f"b{b}": None for b in self.band} if multiband else {"point": None}
            meta = {
                "in_extent": False,
                "n_pixels": 0,
                "had_nodata": False,
                "coverage_pct": 0.0,
                "src_path": str(self.path),
            }
        return values, meta

    def export_tiles(
        self,
        lats,
        lons,
        window_m: int,
        out_dir,
        *,
        ids=None,
        dates=None,
        dataset_name: str = "dataset",
        resample_m: float | None = None,
        filename_suffix: str | None = None,
        progress_desc: str | None = None,
    ):
        """Crop and save a GeoTIFF window centred on each point.

        If resample_m is set, the cropped window is resampled to
        round(window_m / resample_m) × round(window_m / resample_m) pixels
        so all tiles have identical dimensions regardless of native resolution.

        ``filename_suffix`` is inserted before the .tif extension so multi-
        window runs can place every window's tiles in the same folder
        without overwriting one another. Pass ``None`` for the original
        ``"<id>-<dataset>.tif"`` naming.
        """
        from rasterio.transform import Affine
        from rasterio.warp import reproject, Resampling

        out_dir = Path(out_dir) / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        id_list = list(ids) if ids is not None else [str(i) for i in range(len(list(lats)))]
        paths = []

        n_pixels = max(1, round(window_m / resample_m)) if resample_m is not None else None

        # Suffix wrangling: when caller passes "200m", filenames become
        # "<id>-<dataset>-200m.tif". When suffix is None we keep the
        # historical "<id>-<dataset>.tif" naming so single-window callers
        # are completely unaffected.
        suffix_part = f"-{filename_suffix}" if filename_suffix else ""

        # Wrap the per-point tile loop with tqdm so users get visible progress
        # for what can be a long sequential operation on large input sets.
        for lat, lon, sample_id in tqdm(
            zip(lats, lons, id_list),
            total=len(id_list),
            desc=progress_desc or "Local tiles",
            unit="tile",
        ):
            out_path = out_dir / f"{sample_id}-{dataset_name}{suffix_part}.tif"
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
                dst_res_y = (
                    src_transform.e * src_h
                ) / n_pixels  # total height / n_pixels (negative)
                dst_transform = Affine(
                    dst_res_x, 0.0, src_transform.c, 0.0, dst_res_y, src_transform.f
                )
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

        return paths, [{}] * len(paths)

    def build_dataset_meta(
        self,
        spec: Dict[str, Any],
        meta_list: list | None = None,
        exported_paths: list | None = None,
        quality: dict | None = None,
        lats: Sequence[float] | None = None,
        lons: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        """Build per-dataset metadata using this adapter's local raster state.

        No date dimension and a single native CRS, so this is much simpler
        than the GEE implementation. Quality stats are added when present.
        """
        # Static dataset info from the catalog spec.
        meta: Dict[str, Any] = {
            "data_source": spec.get("data_source"),
            "path": spec.get("path"),
            "asset_type": "local_raster",
        }
        if spec.get("data_type"):
            meta["data_type"] = spec["data_type"]

        # Native CRS and resolution from the already-open rasterio dataset.
        meta["native_crs"] = str(self.raster_crs)
        if hasattr(self.src, "res"):
            meta["native_spatial_resolution_m"] = round(float(self.src.res[0]), 2)

        # For local rasters, tiles are exported in the file's native CRS
        # (unlike GEE, where each tile uses its own UTM zone).
        meta["tile_crs"] = str(self.raster_crs)

        # Pass-through catalog field for dataset description.
        dataset_info = spec.get("dataset_information")
        if dataset_info:
            meta["dataset_information"] = dataset_info

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


if _register is not None:
    _register("local", LocalRasterAdapter)
