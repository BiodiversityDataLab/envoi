# src/envoi/adapters/local_adapter.py
from __future__ import annotations
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Sequence

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping
from pyproj import Transformer
from rasterio.errors import RasterioIOError, WindowError
from tqdm.auto import tqdm

from .base import BaseAdapter
from ..reducers import get_reducer
from ..geo import get_utm_crs
from ..metadata import summarize_tile_export

try:
    from . import register as _register
except ImportError:
    # The adapter registry is optional at import time — when this module is
    # imported in isolation (e.g. some tests), the package-level register
    # may not be available yet. Any other exception is a real bug and should
    # surface, so we deliberately do not catch a broader class here.
    _register = None


def _is_nodata(value, nodata) -> bool:
    """Return True if ``value`` represents the raster's nodata sentinel.

    Handles the NaN-as-nodata case explicitly: ``nan == nan`` is False in
    IEEE-754, so a naive ``value == nodata`` comparison silently lets NaN
    nodata pixels through as "valid" data. When the raster declares a NaN
    nodata, we compare via ``np.isnan`` instead.
    """
    if nodata is None:
        return False
    # NaN nodata: we have to use isnan because nan == nan is False.
    if isinstance(nodata, float) and np.isnan(nodata):
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False
    return value == nodata


@dataclass
class LocalRasterAdapter(BaseAdapter):
    spec: Dict[str, Any]

    # ------------------------------------------------------------------
    # Setup / lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self):
        self.path = Path(self.spec["path"])
        if not self.path.exists():
            raise FileNotFoundError(f"Raster not found: {self.path}")

        self.src = rasterio.open(self.path)
        self.raster_crs = self.src.crs

        # Determine which bands to read. If the user specifies "bands" in the
        # catalog (a single int or list of ints), use that. Otherwise default
        # to all bands in the file so no data is silently dropped. The actual
        # branching lives in _normalize_bands_spec to keep this method short.
        self._normalize_bands_spec(self.spec.get("bands"))

    def _normalize_bands_spec(self, bands_spec) -> None:
        """Resolve ``self.band`` (list[int]) and ``self._is_multiband`` from spec.

        Normalises the bands list to ``self.band: list[int]`` so the rest of
        the class has a single shape to reason about. ``_is_multiband``
        captures the OUTPUT-SHAPE rule, which depends on the *form* the user
        supplied — not just on the length of the list:
          * scalar ``bands: 1``        → flat naming (``mean``, ``std``, ...)
          * list   ``bands: [1]``      → per-band naming (``b1_mean``, ...)
          * list   ``bands: [1, 2]``   → per-band naming
          * absent (defaults to all)   → per-band when the file has >1 band
        This matches how the GEE adapter / column-name logic distinguishes
        "I want one band" (scalar) from "I want this list of bands which
        happens to have one element" (still list-shaped output).
        """
        if bands_spec is None:
            self.band = list(range(1, self.src.count + 1))
            self._is_multiband = len(self.band) > 1
        elif isinstance(bands_spec, (list, tuple)):
            self.band = [int(b) for b in bands_spec]
            # Explicit list → always use per-band naming, even at length 1.
            self._is_multiband = True
        else:
            # Scalar (int / numpy scalar) — wrap in a list internally but keep
            # flat naming for backward compatibility with single-band specs.
            self.band = [int(bands_spec)]
            self._is_multiband = False

    def close(self) -> None:
        """Release the underlying rasterio dataset.

        Idempotent — safe to call from ``__exit__`` even if ``__post_init__``
        raised before the dataset was opened, or if ``close()`` is called more
        than once.
        """
        src = getattr(self, "src", None)
        if src is not None and not src.closed:
            src.close()

    # ------------------------------------------------------------------
    # Band metadata helpers
    # ------------------------------------------------------------------

    def _per_band_nodata(self) -> tuple:
        """Return a tuple of nodata values, one per band in ``self.band``.

        ``rasterio.DatasetReader.nodatavals`` is indexed from 0, so band N
        corresponds to ``nodatavals[N - 1]``. Each entry may be ``None`` if
        that band has no nodata declared.
        """
        all_nodatavals = self.src.nodatavals  # 0-indexed tuple, one per band
        return tuple(all_nodatavals[band_num - 1] for band_num in self.band)

    def _per_band_dtypes(self) -> tuple:
        """Return a tuple of numpy dtype strings, one per band in ``self.band``."""
        all_dtypes = self.src.dtypes  # 0-indexed tuple, one per band
        return tuple(all_dtypes[band_num - 1] for band_num in self.band)

    @staticmethod
    def _resolve_tile_dtype(meta_dtype, fallback_dtype) -> np.dtype:
        """Pick the single dtype to use for an exported tile profile.

        GeoTIFFs require one dtype across all bands. Most multi-band rasters
        already use a uniform dtype, so the common path is just
        ``np.dtype(meta_dtype[0])``. For the rare mixed-dtype case we promote
        with ``np.result_type`` to preserve every band's precision and warn
        the user that the on-disk size may have grown.

        Args:
            meta_dtype: as stored in fetch_values' meta — ``list[str]`` for
                multi-band, ``str`` for single-band, or ``None`` when no
                source dtype was recorded (e.g. the out-of-extent path).
            fallback_dtype: dtype used when ``meta_dtype`` is ``None``.
        """
        if isinstance(meta_dtype, list):
            unique_dtypes = {np.dtype(d) for d in meta_dtype}
            if len(unique_dtypes) == 1:
                # Common path: every band already shares a dtype.
                return next(iter(unique_dtypes))
            # Mixed dtypes: promote to a common one that won't truncate any
            # band's precision. np.result_type follows NumPy's standard
            # type-promotion rules (e.g. uint8 + float32 → float32), which
            # is what we want for preserving data fidelity in the export.
            promoted = np.result_type(*unique_dtypes)
            warnings.warn(
                "Source raster has heterogeneous band dtypes "
                f"({sorted(str(d) for d in unique_dtypes)}); exported tile "
                f"is promoted to {promoted} to preserve precision — on-disk "
                "size may increase.",
                stacklevel=2,
            )
            return promoted
        if meta_dtype is not None:
            return np.dtype(meta_dtype)
        # No dtype recorded in meta (e.g. no source dataset to query) —
        # fall back to whatever dtype the in-memory window already has.
        return np.dtype(fallback_dtype)

    @staticmethod
    def _synthetic_nodata_for_dtype(dtype) -> Any:
        """Choose a fabricated nodata sentinel for a band that lacks one.

        Picks a value compatible with the band's dtype so the output array
        stays in the source dtype (no float upcast for integer bands):
          * floating dtypes → ``np.nan``
          * integer dtypes  → ``np.iinfo(dtype).max`` (255 for uint8,
            32767 for int16, 2147483647 for int32, etc.). Using the dtype's
            max avoids overflow on small dtypes (a literal ``99999`` would
            wrap to ``159`` on uint8) and is far less likely to collide with
            real data than ``0``.
          * other dtypes (bool, complex, ...) → ``0`` cast to the dtype, as
            a defensive fallback. These are exotic for raster bands.
        """
        if np.issubdtype(dtype, np.floating):
            return np.nan
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).max
        return dtype.type(0)

    def _fill_masked_window(self, masked_array, per_band_nodata: tuple):
        """Convert a (possibly masked) rio_mask output to a plain ndarray.

        Returns ``(filled_array, resolved_nodata)`` where ``resolved_nodata``
        is a tuple the same length as ``per_band_nodata`` with every ``None``
        entry replaced by the synthetic sentinel actually written into the
        array. Callers store ``resolved_nodata`` in ``meta["nodata"]`` so
        downstream tile export can declare an honest GeoTIFF nodata value
        — without this, the file would silently contain fabricated fill
        pixels in its corners while declaring no nodata at all.

        For non-masked inputs nothing is fabricated, so the original
        ``per_band_nodata`` (which may legitimately contain ``None``) is
        returned unchanged.
        """
        if not np.ma.isMaskedArray(masked_array):
            return masked_array, per_band_nodata

        # Resolve None entries to a dtype-compatible synthetic sentinel.
        # This both fills the array and gets reported back so meta["nodata"]
        # describes what's actually in the array, not what was originally
        # declared on the source band.
        resolved_nodata = tuple(
            nodata if nodata is not None else self._synthetic_nodata_for_dtype(masked_array.dtype)
            for nodata in per_band_nodata
        )

        if masked_array.ndim == 2:
            # Single band (either a true single-band raster or one band
            # selected from a multi-band file): one resolved value covers
            # the whole window.
            return masked_array.filled(resolved_nodata[0]), resolved_nodata

        # Multi-band: fill each band independently with its own resolved value.
        filled_array = np.empty(masked_array.shape, dtype=masked_array.dtype)
        for band_index in range(masked_array.shape[0]):
            filled_array[band_index] = masked_array[band_index].filled(resolved_nodata[band_index])
        return filled_array, resolved_nodata

    # ------------------------------------------------------------------
    # Geometry / CRS
    # UTM helpers live in ``..metadata`` so both adapters share one impl.
    # ------------------------------------------------------------------

    def _project_meter_square_to_raster_geom(self, lat: float, lon: float, window_m: int):
        # Determine a metric CRS for building the square:
        # use the point's UTM zone for global flexibility
        metric_crs = get_utm_crs(lon, lat)
        wgs84_to_metric = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
        center_x, center_y = wgs84_to_metric.transform(lon, lat)
        # Build a square in metres around the centre.
        half_width_m = window_m / 2.0
        square_in_metric_crs = box(
            center_x - half_width_m,
            center_y - half_width_m,
            center_x + half_width_m,
            center_y + half_width_m,
        )
        # Transform the polygon into the raster's CRS so rio_mask can use it.
        square_raster_geojson = transform_geom(
            metric_crs, self.raster_crs, mapping(square_in_metric_crs), precision=6
        )
        return square_raster_geojson

    # ------------------------------------------------------------------
    # Core read
    # ------------------------------------------------------------------

    def fetch_values(self, lat: float, lon: float, window_m: int, *, return_meta: bool = False):
        geom_raster = self._project_meter_square_to_raster_geom(lat, lon, window_m)
        # Resolve the per-band nodata tuple once — rasterio exposes one entry
        # per band, which can differ across bands. We use this both to fill
        # the masked window for tile export and to record an accurate value
        # in the meta dict.
        per_band_nodata = self._per_band_nodata()
        per_band_dtypes = self._per_band_dtypes()
        # rio_mask returns a 2D array when `indexes` is an int and a 3D array
        # when it's a list. Pass an int for the single-band path so the rest
        # of fetch_values can keep the simpler 2D shape downstream.
        read_indexes = self.band if self._is_multiband else self.band[0]
        try:
            # all_touched=False (default) uses center-in-polygon: only pixels
            # whose centres fall inside the polygon are included. This avoids
            # the bounding-box overshoot of geometry_window and matches GEE's
            # reduceRegion pixel-selection rule.
            cropped_window, window_affine = rio_mask(
                self.src,
                [geom_raster],
                crop=True,
                all_touched=False,
                filled=False,
                indexes=read_indexes,
            )
        except (ValueError, WindowError):
            # Match the in-extent return shape so downstream code can rely on
            # ndim alone to distinguish multi-band from single-band output.
            # Without this, a multi-band point that falls outside the raster
            # would return a 1D empty array and _fetch_stats_single's
            # ``ndim == 2`` check would silently route to the single-band
            # branch — producing flat-named keys (``mean``) instead of the
            # per-band keys (``b1_mean``, ``b2_mean``) used for in-extent
            # points, leading to an inconsistent DataFrame schema.
            if self._is_multiband:
                valid_values = np.empty((len(self.band), 0))
            else:
                valid_values = np.array([])
            meta = {
                "in_extent": False,
                "n_pixels": 0,
                "had_nodata": False,
                "coverage_pct": 0.0,
                "window_m": int(window_m),
                "raster_crs": str(self.raster_crs),
                "region_crs": str(self.raster_crs),
                # JSON-safe placeholders so the meta schema stays consistent
                # with the in-extent path even when nothing was read.
                "transform": None,
                "dtype": None,
                "nodata": None,
                "src_path": str(self.path),
                "window_arr": None,
            }
            return (valid_values, meta) if return_meta else valid_values

        # cropped_window is 2D (H, W) for a single band int, 3D (n_bands, H, W) for a list.

        # Materialise the masked window into a plain ndarray for tile export.
        # For multi-band rasters each band can have its own nodata sentinel,
        # so we fill band-by-band with the matching value (falling back to a
        # generic numeric sentinel only when nodatavals[i] is None). The
        # resolved tuple replaces ``None`` entries with the synthetic value
        # actually written, so the meta dict (and the exported GeoTIFF's
        # nodata profile) accurately describes the array contents.
        window_array, resolved_nodata = self._fill_masked_window(cropped_window, per_band_nodata)

        if cropped_window.ndim == 3:
            # Multi-band case. We must reduce every band over the SAME set of
            # pixels, otherwise `np.stack` below would fail when bands have
            # different nodata footprints (e.g. one band has a cloud mask that
            # another doesn't). So: a pixel is considered valid only if it is
            # unmasked AND finite in EVERY band.
            if np.ma.isMaskedArray(cropped_window):
                band_mask = np.ma.getmaskarray(cropped_window)  # True where masked
                data_float = cropped_window.filled(np.nan).astype(float, copy=False)
            else:
                band_mask = np.zeros(cropped_window.shape, dtype=bool)
                data_float = np.asarray(cropped_window, dtype=float)
            # invalid_pixel_mask[row, col] is True if ANY band is masked or
            # non-finite at that location.
            invalid_pixel_mask = band_mask.any(axis=0) | (~np.isfinite(data_float)).any(axis=0)
            valid_mask_2d = ~invalid_pixel_mask
            total_pixels = int(valid_mask_2d.size)
            valid_pixel_count = int(valid_mask_2d.sum())
            # Shape (n_bands, n_valid_pixels) — one row per band, every column
            # is a pixel that was valid in every band.
            valid_values = np.stack(
                [
                    data_float[band_index][valid_mask_2d]
                    for band_index in range(data_float.shape[0])
                ],
                axis=0,
            )
        else:
            # Single-band: straightforward — drop masked and non-finite pixels.
            total_pixels = cropped_window.size
            valid_pixel_count = (
                int(cropped_window.count())
                if np.ma.isMaskedArray(cropped_window)
                else int(np.isfinite(cropped_window).sum())
            )
            valid_values = (
                cropped_window.compressed()
                if np.ma.isMaskedArray(cropped_window)
                else cropped_window.ravel()
            )
            valid_values = valid_values[np.isfinite(valid_values)]

        had_nodata = bool(valid_pixel_count < total_pixels)
        coverage_pct = 100.0 * (valid_pixel_count / total_pixels) if total_pixels else 0.0

        # JSON-safe transform: rasterio's Affine isn't JSON-serializable, so
        # flatten window_affine into a list of 6 floats for the meta dict.
        transform_list = [
            window_affine.a,
            window_affine.b,
            window_affine.c,
            window_affine.d,
            window_affine.e,
            window_affine.f,
        ]

        # Multi-band: surface the per-band nodata/dtype tuples so callers
        # (esp. tile export) can reproduce the source file's structure
        # faithfully. Single-band: keep the scalar form for backward compat.
        # We use ``resolved_nodata`` (not ``per_band_nodata``) so any synthetic
        # sentinel chosen by _fill_masked_window for a band lacking declared
        # nodata gets propagated — the exported GeoTIFF then declares the
        # same value it actually contains in the polygon-exterior corners.
        if self._is_multiband:
            meta_nodata: Any = list(resolved_nodata)
            meta_dtype: Any = [str(d) for d in per_band_dtypes]
        else:
            meta_nodata = resolved_nodata[0]
            meta_dtype = str(per_band_dtypes[0])

        meta = {
            "in_extent": True,
            # n_pixels reports valid (non-nodata, finite) cells only — total
            # cells can be derived from window_m / native pixel size.
            "n_pixels": int(valid_pixel_count),
            "had_nodata": had_nodata,
            "coverage_pct": float(coverage_pct),
            "window_m": int(window_m),
            "raster_crs": str(self.raster_crs),
            "region_crs": str(self.raster_crs),
            # Fields below are consumed by export_tiles to write the per-point
            # GeoTIFF. transform/dtype/nodata also surface in the sidecar JSON
            # for QC; window_arr is in-memory only and stripped before serialise.
            "transform": transform_list,  # JSON-safe (list, not Affine)
            "dtype": meta_dtype,
            "nodata": meta_nodata,
            "src_path": str(self.path),
            "window_arr": window_array,
        }
        return (np.asarray(valid_values), meta) if return_meta else np.asarray(valid_values)

    # ------------------------------------------------------------------
    # Mode 1: Tabular stats  (window reducers + optional point sample)
    # ------------------------------------------------------------------

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

        is_multiband = self._is_multiband
        per_band_nodata = self._per_band_nodata()

        # Helper: build the "missing/out-of-extent" return shape.
        def _missing() -> tuple[dict, dict]:
            values_missing = (
                {f"b{band_num}": None for band_num in self.band}
                if is_multiband
                else {"point": None}
            )
            meta_missing = {
                "in_extent": False,
                "n_pixels": 0,
                "had_nodata": False,
                "coverage_pct": 0.0,
                "src_path": str(self.path),
            }
            return values_missing, meta_missing

        raster_x, raster_y = transformer.transform(lon, lat)

        # Convert raster-CRS coordinates to pixel (col, row) via the inverse
        # affine, then bounds-check on the pixel grid. This works for any
        # affine — including rotated/skewed transforms where the raster's
        # axis-aligned bounding box would falsely include points that lie
        # outside the actual raster footprint. ``~transform * (x, y)`` is
        # the rasterio idiom for the forward Affine inverse.
        pixel_col, pixel_row = ~self.src.transform * (raster_x, raster_y)
        if not (0 <= pixel_row < self.src.height and 0 <= pixel_col < self.src.width):
            return _missing()

        try:
            raw_pixel_values = next(self.src.sample([(raster_x, raster_y)], indexes=self.band))
        except (RasterioIOError, IndexError, ValueError, StopIteration):
            # Genuine read failure (corrupt block, sample iterator empty,
            # band index out of range). Fall back to the missing shape so
            # downstream code still sees a consistent schema. Programming
            # errors (TypeError, AttributeError, ...) deliberately propagate.
            return _missing()

        if is_multiband:
            values: Dict[str, Any] = {}
            for band_num, pixel_value, band_nodata in zip(
                self.band, raw_pixel_values, per_band_nodata
            ):
                clean_value = None if _is_nodata(pixel_value, band_nodata) else float(pixel_value)
                values[f"b{band_num}"] = clean_value
            any_valid = any(value is not None for value in values.values())
        else:
            pixel_value = raw_pixel_values[0]
            clean_value = (
                None if _is_nodata(pixel_value, per_band_nodata[0]) else float(pixel_value)
            )
            values = {"point": clean_value}
            any_valid = clean_value is not None
        meta = {
            "in_extent": any_valid,
            "n_pixels": 1 if any_valid else 0,
            "had_nodata": not any_valid,
            "coverage_pct": 100.0 if any_valid else 0.0,
            "src_path": str(self.path),
        }
        return values, meta

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

        When both branches run, the window branch's meta is returned (it
        carries the full transform/dtype/nodata/window_arr fields from
        fetch_values). When only the point branch runs, those fields are
        synthesized as None so callers downstream see a consistent shape.
        """
        # ``self.band`` is always a list (normalised in __post_init__); the
        # ``_is_multiband`` flag captures the output-shape rule used below.
        is_multiband = self._is_multiband
        band_nums = self.band

        stats: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}

        # ---- window branch ----
        if window_reducer_fns:
            window_values, meta = self.fetch_values(lat, lon, window_m, return_meta=True)
            window_values = np.asarray(window_values)
            if is_multiband and window_values.ndim == 2:
                # window_values shape: (n_bands, n_valid_pixels)
                for band_index, band_num in enumerate(band_nums):
                    band_values = window_values[band_index]
                    for reducer_name, reducer in window_reducer_fns:
                        is_class_reducer = reducer_name in ("class_count", "class_fraction")
                        if not band_values.size:
                            # Empty window: scalar reducers get a None placeholder
                            # (preserves the missing-data signal). Class reducers
                            # get no key at all — _append_stat_columns zero-fills
                            # them against other rows' class observations.
                            if not is_class_reducer:
                                stats[f"b{band_num}_{reducer_name}"] = None
                            continue
                        result = reducer(band_values)
                        # Categorical reducers (class_count, class_fraction)
                        # return a dict {class_value: scalar}. Expand into
                        # per-class stat keys so the output column naming
                        # matches the single-stat-key convention everywhere
                        # else in the pipeline.
                        if isinstance(result, dict):
                            suffix = "count" if reducer_name == "class_count" else "fraction"
                            for class_value, scalar in result.items():
                                stats[f"b{band_num}_class_{class_value}_{suffix}"] = scalar
                        else:
                            stats[f"b{band_num}_{reducer_name}"] = result
            else:
                for reducer_name, reducer in window_reducer_fns:
                    is_class_reducer = reducer_name in ("class_count", "class_fraction")
                    if not window_values.size:
                        if not is_class_reducer:
                            stats[reducer_name] = None
                        continue
                    result = reducer(window_values)
                    if isinstance(result, dict):
                        suffix = "count" if reducer_name == "class_count" else "fraction"
                        for class_value, scalar in result.items():
                            stats[f"class_{class_value}_{suffix}"] = scalar
                    else:
                        stats[reducer_name] = result

        # ---- point branch ----
        if want_point:
            point_values, point_meta = self._sample_point_single(lat, lon)
            if is_multiband:
                # _sample_point_single returns "b{n}" keys — re-key with the
                # "_point" suffix so column names are unambiguous downstream.
                for band_num in band_nums:
                    stats[f"b{band_num}_point"] = point_values.get(f"b{band_num}")
            else:
                stats["point"] = point_values.get("point")

            # If only the point branch ran, synthesize the meta fields that
            # the window branch would normally have provided.
            if not window_reducer_fns:
                meta = {
                    **point_meta,
                    "window_m": int(window_m),
                    "raster_crs": str(self.raster_crs),
                    "region_crs": str(self.raster_crs),
                    "transform": None,
                    "dtype": None,
                    "nodata": None,
                    "window_arr": None,
                }

        return stats, meta

    def fetch_stats_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        reducer_names: Sequence[str],
        *,
        dates: (
            Sequence | None
        ) = None,  # noqa: ARG002 — accepted only for API parity with GeeRasterAdapter; local rasters have no time dimension.
        progress_desc: str | None = None,
    ) -> List[tuple[dict, dict]]:
        """Unified stats fetch: dispatches window reducers and the "point" reducer.

        ``reducer_names`` may contain any mix of window reducers (e.g. "mean",
        "std") and the special ``"point"`` reducer, which samples the exact
        pixel value at the coordinate. Window reducers run over a square window
        of size ``window_m``; "point" is independent of ``window_m``.

        Returns ``[(stats_dict, meta), ...]`` — one tuple per input point.
        Keys in ``stats_dict``:
          * single-band window: ``{reducer_name}``
          * multi-band window:  ``b{band_num}_{reducer_name}``
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

    # ------------------------------------------------------------------
    # Mode 2: Tile export  (per-point GeoTIFF crops)
    # ------------------------------------------------------------------

    def export_tiles(
        self,
        lats,
        lons,
        window_m: int,
        output_dir,
        *,
        ids=None,
        dates=None,
        dataset_name: str = "dataset",
        resample_m: float | None = None,
        filename_suffix: str | None = None,
        progress_desc: str | None = None,
    ):
        """Crop and save a GeoTIFF window centred on each point.

        If resample_m is set, the output tile is written at exactly resample_m
        metres/pixel in the point's UTM zone, with the pixel grid snapped to the
        nearest resample_m multiple — the same algorithm the GEE adapter uses.
        This means local and GEE tiles for the same point at the same resample_m
        are spatially aligned and directly comparable. Without resample_m, the
        tile is written in the source raster's native CRS at native resolution.

        ``filename_suffix`` is inserted before the .tif extension so multi-
        window runs can place every window's tiles in the same folder
        without overwriting one another. Pass ``None`` for the original
        ``"<id>-<dataset>.tif"`` naming.
        """
        from rasterio.transform import Affine
        from rasterio.warp import reproject, Resampling

        output_dir = Path(output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Materialise inputs to lists once so we can both call len() (for
        # tqdm) and iterate over them safely. Generators would otherwise be
        # exhausted by len(list(...)) and produce nothing in the loop.
        lats_list = list(lats)
        lons_list = list(lons)
        if ids is not None:
            id_list = list(ids)
        else:
            id_list = [str(i) for i in range(len(lats_list))]

        paths: List[Any] = []
        # One meta dict per tile, mirroring fetch_values' meta. Carries the
        # per-point QC info (in_extent, n_pixels, coverage_pct, …) so the
        # raster-mode QC summary can report which tiles failed and why.
        tile_metas: List[dict] = []

        n_pixels = max(1, round(window_m / resample_m)) if resample_m is not None else None

        # Suffix wrangling: when caller passes "200m", filenames become
        # "<id>-<dataset>-200m.tif". When suffix is None we keep the
        # historical "<id>-<dataset>.tif" naming so single-window callers
        # are completely unaffected.
        suffix_part = f"-{filename_suffix}" if filename_suffix else ""

        # Wrap the per-point tile loop with tqdm so users get visible progress
        # for what can be a long sequential operation on large input sets.
        for lat, lon, sample_id in tqdm(
            zip(lats_list, lons_list, id_list),
            total=len(id_list),
            desc=progress_desc or "Local tiles",
            unit="tile",
        ):
            output_path = output_dir / f"{sample_id}-{dataset_name}{suffix_part}.tif"
            _, meta = self.fetch_values(lat, lon, window_m, return_meta=True)

            window_array = meta.get("window_arr")
            transform_list = meta.get("transform")
            # Skip tiles for which the window couldn't be cropped at all
            # (e.g. the point fell outside the raster's extent). Preserve
            # the meta dict so callers can still see why each tile failed.
            if window_array is None or window_array.size == 0 or transform_list is None:
                paths.append(None)
                tile_metas.append(meta)
                continue

            # Multi-band rasters retain all bands in the exported tile so the
            # output GeoTIFF mirrors the source's band structure (matching
            # the GEE adapter's behaviour). Single-band rasters keep their
            # 2D shape with count == 1.
            if window_array.ndim == 3:
                n_bands = window_array.shape[0]
                # Multi-band: pass the (n_bands, h, w) array straight through —
                # both reproject() and rasterio.write() accept this shape.
                source_array = window_array
            else:
                n_bands = 1
                # rasterio.warp.reproject handles 2D arrays directly; only
                # the write step needs the band index, so we keep the 2D
                # shape rather than reshaping unnecessarily.
                source_array = window_array

            source_transform = Affine(*transform_list)
            # Per-band dtype/nodata are stored as a list when multi-band and
            # as a scalar when single-band. Normalise to the scalar form
            # below since GeoTIFFs use a single dtype/nodata for all bands.
            meta_dtype = meta.get("dtype")
            meta_nodata = meta.get("nodata")
            tile_dtype = self._resolve_tile_dtype(meta_dtype, source_array.dtype)
            if isinstance(meta_nodata, list):
                tile_nodata = meta_nodata[0]
            else:
                tile_nodata = meta_nodata

            if n_pixels is not None:
                # Build the output grid in the point's UTM zone, snapped to the
                # resample_m pixel boundary — matching GEE adapter behaviour exactly.
                # Two things must agree with GEE for tiles to be spatially comparable:
                #   1. Pixel size = resample_m (not native-span / n_pixels, which varies
                #      per point because rio_mask snaps to native pixel boundaries).
                #   2. Grid origin = center snapped to resample_m, then shifted by half
                #      the window extent — the same formula GEE's _snap_to_grid uses.
                utm_crs = get_utm_crs(lon, lat)
                utm_transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
                cx_utm, cy_utm = utm_transformer.transform(lon, lat)
                # Snap centre to nearest resample_m multiple (mirrors GEE's _snap_to_grid).
                cx_snapped = round(cx_utm / resample_m) * resample_m
                cy_snapped = round(cy_utm / resample_m) * resample_m
                half_m = (n_pixels // 2) * resample_m
                destination_transform = Affine(
                    resample_m,
                    0.0,
                    cx_snapped - half_m,
                    0.0,
                    -resample_m,
                    cy_snapped + half_m,
                )
                # Reproject directly from the open source dataset rather than from the
                # polygon-masked window_array. The polygon mask sets edge pixels to NaN
                # wherever the native pixel centre falls outside the 200 m polygon;
                # bilinear interpolation at output cells near the window boundary then
                # pulls from those NaN source pixels and propagates NaN into the output.
                # Going straight to the source file gives the interpolation valid data
                # everywhere, matching what GEE does when it exports a plain rectangle.
                if self._is_multiband:
                    destination_array = np.empty((n_bands, n_pixels, n_pixels), dtype=tile_dtype)
                    for band_idx, band_num in enumerate(self.band):
                        reproject(
                            source=rasterio.band(self.src, band_num),
                            destination=destination_array[band_idx],
                            dst_transform=destination_transform,
                            dst_crs=utm_crs,
                            resampling=Resampling.bilinear,
                            dst_nodata=tile_nodata,
                        )
                else:
                    destination_array = np.empty((n_pixels, n_pixels), dtype=tile_dtype)
                    reproject(
                        source=rasterio.band(self.src, self.band[0]),
                        destination=destination_array,
                        dst_transform=destination_transform,
                        dst_crs=utm_crs,
                        resampling=Resampling.bilinear,
                        dst_nodata=tile_nodata,
                    )
                output_array = destination_array
                output_transform = destination_transform
                output_crs = utm_crs
            else:
                output_array = source_array
                output_transform = source_transform
                output_crs = self.raster_crs

            # Determine the height/width index depending on whether we have a
            # 2D (single-band) or 3D (multi-band) array.
            if output_array.ndim == 3:
                profile_height, profile_width = output_array.shape[1], output_array.shape[2]
            else:
                profile_height, profile_width = output_array.shape

            profile = {
                "driver": "GTiff",
                "height": profile_height,
                "width": profile_width,
                "count": n_bands,
                "dtype": str(output_array.dtype),
                "crs": output_crs,
                "transform": output_transform,
                "nodata": tile_nodata,
                "compress": "LZW",
            }
            with rasterio.open(output_path, "w", **profile) as tile_writer:
                if output_array.ndim == 3:
                    # Write all bands at once — rasterio expects the data to
                    # be shape (count, h, w) and the indexes to be a list of
                    # 1-based band numbers.
                    tile_writer.write(output_array, indexes=list(range(1, n_bands + 1)))
                else:
                    tile_writer.write(output_array, 1)
            paths.append(output_path)
            tile_metas.append(meta)

        return paths, tile_metas

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

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
