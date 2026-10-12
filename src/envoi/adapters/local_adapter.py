# src/envoi/adapters/local_adapter.py
from __future__ import annotations

import contextlib
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.errors import CRSError, RasterioIOError, WindowError
from rasterio.features import bounds as geometry_bounds
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box, mapping
from tqdm.auto import tqdm

from .._filenames import build_tile_filenames
from ..geo import (
    build_tile_crs_zones,
    get_utm_crs,
    pixel_size_m,
    tile_pixel_count,
    tile_top_left,
)
from ..metadata import summarize_tile_export
from ..progress import ProgressStepCallback, emit_progress_step
from ..qc import format_point_error
from ..reducers import get_reducer
from .base import BaseAdapter

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


# ----------------------------------------------------------------------
# Raster settings: the CRS and nodata value that every read uses
# ----------------------------------------------------------------------


def _selected_band_numbers(bands_spec: Any, band_count: int) -> list[int]:
    """Return the 1-based band numbers that a ``bands`` setting selects.

    ``None`` selects every band of the file. A list or tuple selects its
    bands, and a single value selects one band.
    """
    if bands_spec is None:
        return list(range(1, band_count + 1))
    if isinstance(bands_spec, (list, tuple)):
        return [int(band) for band in bands_spec]
    return [int(bands_spec)]


def _nodata_values_equal(first_value: Any, second_value: Any) -> bool:
    """Return True if two nodata values are the same number.

    NaN equals NaN here, because ``nan == nan`` is False. An integer and a
    float with the same value (``-32768`` and ``-32768.0``) are equal.
    """
    first_is_nan = isinstance(first_value, float) and np.isnan(first_value)
    second_is_nan = isinstance(second_value, float) and np.isnan(second_value)
    if first_is_nan or second_is_nan:
        return first_is_nan and second_is_nan
    return first_value == second_value


def _nodata_fits_dtype(nodata: float, dtype_name: str) -> bool:
    """Return True if a band of type ``dtype_name`` can store ``nodata``.

    An integer band needs a whole number (an ``int``, or a ``float`` such as
    ``-9999.0``) within the range of its type. A float band can store NaN,
    infinity, and every number that its type rounds to a finite number, for
    example from about -3.4e38 to 3.4e38 for ``float32``.
    """
    dtype = np.dtype(dtype_name)
    if np.issubdtype(dtype, np.floating):
        if not np.isfinite(nodata):
            return True
        # A finite number fits when the band type rounds it to a finite
        # number. A comparison with the range would reject -3.4028235e38, the
        # printed form of the float32 minimum, which is a little below the
        # exact minimum but rounds to it. A larger number becomes infinity.
        with np.errstate(over="ignore"):
            return bool(np.isfinite(np.asarray(nodata, dtype=dtype)))
    if not np.issubdtype(dtype, np.integer):
        return True
    # NaN, infinity, and fractions such as 1.5 are not whole numbers.
    if isinstance(nodata, float) and not nodata.is_integer():
        return False
    dtype_range = np.iinfo(dtype)
    return dtype_range.min <= int(nodata) <= dtype_range.max


def _nodata_in_band_type(nodata: Any, dtype_name: Any) -> Any:
    """Return ``nodata`` as a band of type ``dtype_name`` stores it.

    An integer band gets an ``int``, and a float band the nearest number of
    its type as a Python ``float``: a ``float32`` band stores -3.4e38 as
    -3.3999999521443642e38. Comparisons of pixels with the converted value
    are then exact in every read path. Without it, the point sample would
    count such nodata pixels as data with NumPy 1.x, where a ``float32``
    pixel and a Python float compare as float64. None, and a value that the
    band cannot store, are returned unchanged.
    """
    if nodata is None or not _nodata_fits_dtype(nodata, dtype_name):
        return nodata
    dtype = np.dtype(dtype_name)
    if np.issubdtype(dtype, np.integer):
        return int(nodata)
    if np.issubdtype(dtype, np.floating):
        return float(np.asarray(nodata, dtype=dtype))
    return nodata


def _crs_label(crs: CRS) -> str:
    """Return ``EPSG:<code>`` for a CRS with an EPSG code, otherwise its text form."""
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else crs.to_string()


def _resolve_local_raster_settings(
    spec: dict[str, Any], *, context: str = "Local raster"
) -> tuple[CRS, int | float | None, str | None]:
    """Resolve the CRS and the nodata value that envoi uses for a local raster.

    Opens the file of ``spec["path"]`` (and closes it again) and combines the
    file's own CRS and nodata value with the optional catalog keys ``crs`` and
    ``nodata``. The catalog keys fill values that the file does not declare;
    they never replace a value of the file:

    - nodata: When the file declares a nodata value for the selected bands,
      envoi uses it. A catalog ``nodata`` that differs raises. When the file
      declares none, envoi uses the catalog ``nodata`` for every selected
      band, if the value fits the data type of each of these bands.
      The returned value, from the file or the catalog, is converted to the
      data type of the bands (``_nodata_in_band_type``): an integer band
      gets an ``int``, and a float band the nearest number that it can store
      (for ``float32``, -3.4e38 becomes -3.3999999521443642e38).
    - CRS: When the file has a CRS, envoi uses it. A catalog ``crs`` that
      differs raises. When the file has no CRS, envoi uses the catalog
      ``crs``, and without one it raises.

    The values that ``catalog._inspect_raster`` copies from the file into a
    catalog entry never raise: a GeoTIFF has one nodata value for all bands,
    and the CRS comparison uses the same ``to_epsg()`` rule as that function.

    ``extract()`` calls this for every local dataset of every run config
    before the first adapter is built, and ``LocalRasterAdapter`` calls it
    again to store the result.

    Args:
        spec: The merged catalog entry: the catalog entry plus the run
            config's band overrides. Uses ``path``, ``bands``, ``nodata``,
            and ``crs``.
        context: The start of each error message, for example
            ``"Output 'terrain': dataset 'my_dem'"``.

    Returns:
        ``(crs, nodata, nodata_source)``: the rasterio CRS that every read
        uses; the nodata value of the selected bands, or None when there is
        none; and where that value comes from: ``"file"``, ``"catalog"``, or
        None.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the catalog ``nodata`` is not one number, does not fit
            the data type of a selected band, or differs from the file's
            nodata value; if the catalog ``crs`` is not a valid CRS or differs
            from the file's CRS; if the file and the catalog entry have no
            CRS; or if a selected band is not in the file.
    """
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    # ---- catalog values ----
    catalog_nodata = spec.get("nodata")
    if catalog_nodata is not None:
        # A GeoTIFF has one nodata value for all its bands, so a list of
        # values (one per band) cannot be applied or compared.
        if isinstance(catalog_nodata, (list, tuple)):
            raise ValueError(
                f"{context}: the catalog entry sets 'nodata' to a list ({catalog_nodata!r}). "
                f"'nodata' must be one number, which envoi uses for every band, because a "
                f"GeoTIFF has one nodata value for all its bands."
            )
        if isinstance(catalog_nodata, (bool, np.bool_)) or not isinstance(
            catalog_nodata, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"{context}: the catalog entry sets 'nodata' to {catalog_nodata!r}, which is "
                f"not a number. Set it to one number, for example -9999 (for NaN, write .nan "
                f"in a YAML file or float('nan') in Python)."
            )
        # Plain Python numbers, so that the comparisons below and the JSON
        # sidecar do not depend on NumPy scalar types.
        if isinstance(catalog_nodata, (int, np.integer)):
            catalog_nodata = int(catalog_nodata)
        else:
            catalog_nodata = float(catalog_nodata)

    catalog_crs = None
    if spec.get("crs") is not None:
        try:
            catalog_crs = CRS.from_user_input(spec["crs"])
        except CRSError as error:
            raise ValueError(
                f"{context}: the catalog entry sets 'crs' to {spec['crs']!r}, which is not a "
                f"valid CRS. Use an EPSG code, for example 'EPSG:32633'."
            ) from error

    with rasterio.open(path) as src:
        band_numbers = _selected_band_numbers(spec.get("bands"), src.count)
        invalid_band_numbers = [band for band in band_numbers if not 1 <= band <= src.count]
        if invalid_band_numbers:
            raise ValueError(
                f"{context}: band(s) {invalid_band_numbers} are not in the file {path}, which "
                f"has {src.count} band(s), numbered from 1 to {src.count}."
            )
        declared_nodata_by_band = {
            band: src.nodatavals[band - 1]
            for band in band_numbers
            if src.nodatavals[band - 1] is not None
        }
        band_dtypes = {band: src.dtypes[band - 1] for band in band_numbers}
        # An empty CRS object counts as no CRS.
        file_crs = src.crs if src.crs else None

    # ---- nodata ----
    if declared_nodata_by_band:
        # The file's value wins. A different catalog value means that the
        # catalog entry describes another file, or has a typing error.
        for file_nodata in declared_nodata_by_band.values():
            if catalog_nodata is not None and not _nodata_values_equal(catalog_nodata, file_nodata):
                raise ValueError(
                    f"{context}: the catalog entry sets 'nodata' to {catalog_nodata!r}, but "
                    f"the file {path} declares the nodata value {file_nodata!r}. envoi uses "
                    f"the nodata value of the file. Remove 'nodata' from the catalog entry, "
                    f"or set it to {file_nodata!r}."
                )
        # Record the value as the bands store it, as _per_band_nodata gives
        # it to the read paths.
        first_band, first_nodata = next(iter(declared_nodata_by_band.items()))
        nodata = _nodata_in_band_type(first_nodata, band_dtypes[first_band])
        nodata_source = "file"
    elif catalog_nodata is not None:
        # The catalog value fills the missing nodata value of every selected
        # band, so it must be a value that each band can store.
        for band, dtype_name in band_dtypes.items():
            if not _nodata_fits_dtype(catalog_nodata, dtype_name):
                if np.issubdtype(np.dtype(dtype_name), np.integer):
                    dtype_range = np.iinfo(np.dtype(dtype_name))
                    allowed = f"a whole number from {dtype_range.min} to {dtype_range.max}"
                else:
                    dtype_range = np.finfo(np.dtype(dtype_name))
                    allowed = (
                        f"NaN or a number from {float(dtype_range.min)!r} "
                        f"to {float(dtype_range.max)!r}"
                    )
                raise ValueError(
                    f"{context}: the catalog entry sets 'nodata' to {catalog_nodata!r}, but "
                    f"band {band} of the file {path} has the data type {dtype_name}, which "
                    f"cannot store this value. Use {allowed}."
                )

        # Use the value as the bands store it, so that every read path
        # compares the pixels with the same number. A GeoTIFF has one data
        # type for all bands; np.result_type covers other formats.
        if band_dtypes:
            catalog_nodata = _nodata_in_band_type(
                catalog_nodata, np.result_type(*band_dtypes.values())
            )
        nodata, nodata_source = catalog_nodata, "catalog"
    else:
        nodata, nodata_source = None, None

    # ---- CRS ----
    if file_crs is None:
        if catalog_crs is None:
            raise ValueError(
                f"{context}: the file {path} has no CRS, and the catalog entry has no "
                f"'crs'. Add the CRS of the file to the catalog entry, for example "
                f"crs: 'EPSG:32633'."
            )
        return catalog_crs, nodata, nodata_source

    if catalog_crs is not None:
        # Compare EPSG codes first, as catalog._inspect_raster writes the code
        # of the file's CRS. A WKT text can differ in details from the file's
        # own CRS although both describe the same CRS.
        file_epsg = file_crs.to_epsg()
        catalog_epsg = catalog_crs.to_epsg()
        if file_epsg is not None and catalog_epsg is not None:
            crs_matches = file_epsg == catalog_epsg
        else:
            # CRS.equals() exists only from rasterio 1.4; envoi supports 1.3,
            # where == compares the two CRS definitions.
            crs_matches = file_crs == catalog_crs
        if not crs_matches:
            raise ValueError(
                f"{context}: the catalog entry sets 'crs' to {spec['crs']!r}, but the file "
                f"{path} has the CRS {_crs_label(file_crs)}. envoi uses the CRS of the file. "
                f"Remove 'crs' from the catalog entry, or correct it."
            )
    return file_crs, nodata, nodata_source


@dataclass
class LocalRasterAdapter(BaseAdapter):
    spec: dict[str, Any]

    # ------------------------------------------------------------------
    # Setup / lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self):
        self.path = Path(self.spec["path"])
        if not self.path.exists():
            raise FileNotFoundError(f"Raster not found: {self.path}")

        self.src = rasterio.open(self.path)
        # Guard the remaining setup so a failure (e.g. an invalid `bands` spec)
        # closes the rasterio handle we just opened. Without this, the caller
        # never sees a usable adapter — so the `with` block's __exit__ doesn't
        # run — and the file descriptor would leak for the lifetime of the
        # interpreter. Re-raise so the original error reaches the caller.
        try:
            # The CRS and the nodata value that every read uses: the file's
            # own values, or the catalog's `crs` and `nodata` where the file
            # declares none. Resolve them before anything that needs the CRS.
            # _nodata_source is "file", "catalog", or None (no nodata value).
            self.raster_crs, self._nodata, self._nodata_source = _resolve_local_raster_settings(
                self.spec
            )
            # A tile with resample_m must read the file through a WarpedVRT
            # that declares these catalog values (see export_tiles).
            self._uses_catalog_settings = self._nodata_source == "catalog" or not self.src.crs

            # Native pixel width in metres, or None for a CRS without a
            # length unit. Used for the sidecar and for sub-pixel windows.
            self._native_pixel_size_m = pixel_size_m(self.raster_crs, self.src.res[0])

            # resample_m of the last export_tiles call, or None. With it, tiles
            # are in each point's UTM zone, so build_dataset_meta must list
            # those zones as the tile CRS instead of the raster CRS.
            self._export_resample_m: float | None = None

            # Determine which bands to read. If the user specifies "bands" in the
            # catalog (a single int or list of ints), use that. Otherwise default
            # to all bands in the file so no data is silently dropped. The actual
            # branching lives in _normalize_bands_spec to keep this method short.
            self._normalize_bands_spec(self.spec.get("bands"))
        except BaseException:
            self.close()
            raise

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
        # The band list follows the same rule as the settings check in
        # _resolve_local_raster_settings, through _selected_band_numbers.
        self.band = _selected_band_numbers(bands_spec, self.src.count)
        if bands_spec is None:
            self._is_multiband = len(self.band) > 1
        elif isinstance(bands_spec, (list, tuple)):
            # Explicit list → always use per-band naming, even at length 1.
            self._is_multiband = True
        else:
            # Scalar (int / numpy scalar) — wrap in a list internally but keep
            # flat naming for backward compatibility with single-band specs.
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

        When the file declares no nodata value and the catalog sets one, every
        band gets the catalog value. Otherwise the values come from the file:
        ``rasterio.DatasetReader.nodatavals`` is indexed from 0, so band N
        corresponds to ``nodatavals[N - 1]``. Each entry may be ``None`` if
        that band has no nodata declared. Every value is converted to the data
        type of its band (``_nodata_in_band_type``).

        Uses attributes:
            _nodata: The catalog nodata value when ``_nodata_source`` is
                ``"catalog"``.
            _nodata_source: ``"file"``, ``"catalog"``, or None.
        """
        if self._nodata_source == "catalog":
            return (self._nodata,) * len(self.band)
        # Each file value as its band stores it (see _nodata_in_band_type),
        # so that the point sample compares in the band's data type.
        all_nodatavals = self.src.nodatavals  # 0-indexed tuple, one per band
        all_dtypes = self.src.dtypes
        return tuple(
            _nodata_in_band_type(all_nodatavals[band_num - 1], all_dtypes[band_num - 1])
            for band_num in self.band
        )

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

    def _count_window_pixels_beyond_raster(self, window_geometry: dict) -> int:
        """Return the number of window pixels whose centre is beyond the edge of the raster.

        These are the pixels of the file's pixel grid, continued beyond the
        raster, whose centre is inside ``window_geometry`` (GeoJSON in the
        raster CRS), with the pixel-centre rule of ``rio_mask``
        (``all_touched=False``). They have no data, so the coverage counts
        them as missing pixels, as Earth Engine counts the part of a window
        outside the image. Returns 0, without a rasterization, when the
        window is inside the raster. A MultiPolygon window is counted part
        by part.
        """
        # A window that crosses the antimeridian of a raster in degrees is a
        # MultiPolygon, with one part on each side of 180 degrees. Count each
        # part on its own: the bounding box of both parts spans about 360
        # degrees, and its mask could be too large for memory. The parts do
        # not overlap, so their counts add up.
        if window_geometry["type"] == "MultiPolygon":
            return sum(
                self._count_window_pixels_beyond_raster(
                    {"type": "Polygon", "coordinates": polygon_coordinates}
                )
                for polygon_coordinates in window_geometry["coordinates"]
            )

        # The rows and columns of the file grid that the window can reach,
        # without clipping to the raster: the pixel positions of the corners
        # of the window's bounding box. The inverse transform also works for
        # a rotated grid.
        left, bottom, right, top = geometry_bounds(window_geometry)
        inverse = ~self.src.transform
        corner_columns = []
        corner_rows = []
        for x, y in ((left, bottom), (left, top), (right, bottom), (right, top)):
            corner_columns.append(inverse.a * x + inverse.b * y + inverse.c)
            corner_rows.append(inverse.d * x + inverse.e * y + inverse.f)
        first_column = math.floor(min(corner_columns))
        first_row = math.floor(min(corner_rows))
        n_columns = math.ceil(max(corner_columns)) - first_column
        n_rows = math.ceil(max(corner_rows)) - first_row
        window_is_inside_raster = (
            first_column >= 0
            and first_row >= 0
            and first_column + n_columns <= self.src.width
            and first_row + n_rows <= self.src.height
        )
        if window_is_inside_raster:
            return 0

        # The pixels whose centre is inside the window, on the continued grid.
        grid_window = Window(first_column, first_row, n_columns, n_rows)
        inside_window_mask = geometry_mask(
            [window_geometry],
            out_shape=(n_rows, n_columns),
            transform=window_transform(grid_window, self.src.transform),
            all_touched=False,
            invert=True,
        )

        # Keep the pixels in a row or a column that the file does not have.
        row_numbers = np.arange(first_row, first_row + n_rows)
        column_numbers = np.arange(first_column, first_column + n_columns)
        row_is_in_raster = (row_numbers >= 0) & (row_numbers < self.src.height)
        column_is_in_raster = (column_numbers >= 0) & (column_numbers < self.src.width)
        pixel_is_in_raster = row_is_in_raster[:, np.newaxis] & column_is_in_raster[np.newaxis, :]
        return int((inside_window_mask & ~pixel_is_in_raster).sum())

    # ------------------------------------------------------------------
    # Core read
    # ------------------------------------------------------------------

    def _empty_window_result(self, window_m: int) -> tuple[np.ndarray, dict]:
        """Return the ``(valid_values, meta)`` of a window with no pixels read.

        Used for a point outside the raster's extent and for a failed point.
        The meta dict has the QC values of a point outside the extent.
        """
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
        return valid_values, meta

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
            # The window does not overlap the raster: a point outside the extent.
            valid_values, meta = self._empty_window_result(window_m)
            return (valid_values, meta) if return_meta else valid_values
        except RasterioIOError as error:
            # Reading the window failed, for example because a block of the
            # file is damaged. This is a failed point, not a point outside the
            # extent: the meta dict keeps the error, so extract() can report
            # the point, and the rest of the batch continues.
            valid_values, meta = self._empty_window_result(window_m)
            meta["error"] = format_point_error(error)
            meta["exception"] = error
            return (valid_values, meta) if return_meta else valid_values

        # cropped_window is 2D (H, W) for a single band int, 3D (n_bands, H, W) for a list.

        # The total of the coverage: the pixels of the file grid whose centre
        # is inside the window polygon (all_touched=False), also beyond the
        # edge of the raster. coverage_pct is the share of the window that
        # has valid data, and the part beyond the edge has none, as on Earth
        # Engine. The crop is the bounding box of the window, clipped to the
        # raster, so it also has pixels outside the window, which rio_mask
        # masks as it masks nodata pixels; they must not count. (With the
        # crop as total, a window without missing data had a coverage of
        # about 83 % for 100 m at 10 m pixels, and had_nodata True.) The
        # mask of the crop uses the same geometry, transform, and shape as
        # rio_mask, so every valid pixel is in it.
        inside_window_mask = geometry_mask(
            [geom_raster],
            out_shape=cropped_window.shape[-2:],
            transform=window_affine,
            all_touched=False,
            invert=True,
        )
        total_pixels = int(inside_window_mask.sum()) + self._count_window_pixels_beyond_raster(
            geom_raster
        )

        # rio_mask masks the pixels outside the window and the file's own
        # nodata pixels. When the file declares no nodata value and the
        # catalog sets one, it cannot know that value, so mask its pixels
        # here. Without this, they would count as data in every statistic.
        if self._nodata_source == "catalog":
            window_data = np.ma.getdata(cropped_window)
            # NaN nodata: we have to use isnan because nan == nan is False.
            if isinstance(self._nodata, float) and np.isnan(self._nodata):
                catalog_nodata_mask = np.isnan(window_data)
            else:
                catalog_nodata_mask = window_data == self._nodata
            cropped_window = np.ma.masked_array(
                window_data, mask=np.ma.getmaskarray(cropped_window) | catalog_nodata_mask
            )

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
                # Convert to float before filling: an integer array cannot hold
                # NaN, so filling first raises TypeError for integer rasters.
                data_float = cropped_window.astype(float).filled(np.nan)
            else:
                band_mask = np.zeros(cropped_window.shape, dtype=bool)
                data_float = np.asarray(cropped_window, dtype=float)
            # invalid_pixel_mask[row, col] is True if ANY band is masked or
            # non-finite at that location.
            invalid_pixel_mask = band_mask.any(axis=0) | (~np.isfinite(data_float)).any(axis=0)
            valid_mask_2d = ~invalid_pixel_mask
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
            # Count the pixels after both steps, as for several bands: a NaN
            # or infinite pixel that is not the nodata value is not masked,
            # but it is missing data, so it must lower the coverage.
            valid_values = (
                cropped_window.compressed()
                if np.ma.isMaskedArray(cropped_window)
                else cropped_window.ravel()
            )
            valid_values = valid_values[np.isfinite(valid_values)]
            valid_pixel_count = int(valid_values.size)

        # A window without a valid pixel had missing data, also a window that
        # contains no pixel centre (total 0): a small window of a raster whose
        # CRS has no length unit is not widened.
        had_nodata = bool(valid_pixel_count < total_pixels or valid_pixel_count == 0)
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
            had_nodata, coverage_pct, src_path). When the pixel read fails
            for a point inside the raster, it also has the error text
            (``error``) and the exception (``exception``).
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
        # outside the actual raster footprint. The coefficients are applied
        # by hand: affine 3 deprecates ``~transform * (x, y)``, and affine 2
        # has no ``@`` operator, so this form works with both.
        inverse = ~self.src.transform
        pixel_col = inverse.a * raster_x + inverse.b * raster_y + inverse.c
        pixel_row = inverse.d * raster_x + inverse.e * raster_y + inverse.f
        if not (0 <= pixel_row < self.src.height and 0 <= pixel_col < self.src.width):
            return _missing()

        try:
            raw_pixel_values = next(self.src.sample([(raster_x, raster_y)], indexes=self.band))
        except (RasterioIOError, IndexError, ValueError, StopIteration) as error:
            # The point is inside the raster, but reading its pixel failed
            # (corrupt block, sample iterator empty, band index out of range).
            # This is a failed point, not a point outside the extent: keep the
            # error in the meta dict, so extract() can report the point.
            # Programming errors (TypeError, AttributeError, ...) deliberately
            # propagate.
            values_missing, meta_missing = _missing()
            meta_missing["error"] = format_point_error(error)
            meta_missing["exception"] = error
            return values_missing, meta_missing

        # A pixel is missing when it is the band's nodata value, or when it is
        # NaN or infinite. Rasters often store missing data as NaN without
        # declaring a nodata value. Such a pixel is not a value: treat it as
        # nodata, as the window reducers do (they keep only finite values).
        def _clean_pixel_value(pixel_value, band_nodata) -> float | None:
            if _is_nodata(pixel_value, band_nodata) or not np.isfinite(pixel_value):
                return None
            return float(pixel_value)

        if is_multiband:
            values: dict[str, Any] = {}
            for band_num, pixel_value, band_nodata in zip(
                self.band, raw_pixel_values, per_band_nodata
            ):
                values[f"b{band_num}"] = _clean_pixel_value(pixel_value, band_nodata)
            any_valid = any(value is not None for value in values.values())
        else:
            clean_value = _clean_pixel_value(raw_pixel_values[0], per_band_nodata[0])
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
          each Python-side reducer to the valid pixels. A window smaller than
          one native pixel is read as a window of two native pixels, as on
          the GEE adapter. The meta dict keeps the requested ``window_m``,
          and its coverage describes the window that was read.
        - **point** — runs only when *want_point* is True. Samples the
          exact pixel containing (lat, lon) via ``self.src.sample()``.

        When both branches run, the window branch's meta is returned (it
        carries the full transform/dtype/nodata/window_arr fields from
        fetch_values). When only the point branch runs, those fields are
        synthesized as None so callers downstream see a consistent shape.

        A read error in either branch makes the point a failed point: the
        stats dict is empty, and the meta dict has the QC values of a point
        outside the extent plus the error text (``error``) and the exception
        (``exception``) of the first failed read.
        """
        # ``self.band`` is always a list (normalised in __post_init__); the
        # ``_is_multiband`` flag captures the output-shape rule used below.
        is_multiband = self._is_multiband
        band_nums = self.band

        stats: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        point_meta: dict[str, Any] = {}

        # ---- window branch ----
        if window_reducer_fns:
            # A window smaller than one native pixel can contain no pixel
            # centre, so it would give no pixels. Widen it to two native
            # pixels, as the Earth Engine adapter does. The coverage then
            # describes the widened window; meta keeps the requested window.
            # Without a pixel size in metres (no length unit), do not widen.
            read_window_m = window_m
            if self._native_pixel_size_m is not None and window_m < self._native_pixel_size_m:
                read_window_m = int(self._native_pixel_size_m * 2)
            window_values, meta = self.fetch_values(lat, lon, read_window_m, return_meta=True)
            meta["window_m"] = int(window_m)
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

        # ---- failed point ----
        # A read error in either branch makes this a failed point, as on the
        # Earth Engine adapter, where one request gives both branches. All its
        # statistics are null: it gets no statistic keys, and
        # _append_stat_columns gives it a null value in every column of the
        # batch. The error is kept, so extract() can report the point.
        failed_read_meta = next(
            (read_meta for read_meta in (meta, point_meta) if "error" in read_meta), None
        )
        if failed_read_meta is not None:
            _, failed_meta = self._empty_window_result(window_m)
            failed_meta["error"] = failed_read_meta["error"]
            failed_meta["exception"] = failed_read_meta["exception"]
            return {}, failed_meta

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
    ) -> list[tuple[dict, dict]]:
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
        merged ``(stats, meta)`` tuple per input row. A point whose read
        failed gets an empty stats dict, and its meta dict has the error text
        (``error``) and the exception (``exception``).
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
        results: list[tuple[dict, dict]] = []
        emit_progress_step(progress_callback, 0, total_points)
        for lat, lon in tqdm(
            zip(lats, lons),
            total=total_points,
            desc=progress_desc or "Local stats",
            unit="pt",
            disable=disable_progress,
        ):
            results.append(
                self._fetch_stats_single(
                    lat, lon, window_m, window_reducer_fns, want_point=want_point
                )
            )
            emit_progress_step(progress_callback, len(results), total_points)
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
        disable_progress: bool = False,
        progress_callback: ProgressStepCallback | None = None,
    ):
        """Crop and save a GeoTIFF window centred on each point.

        If resample_m is set, the output tile is written at exactly resample_m
        metres/pixel in the point's UTM zone, with n × n pixels, where n is
        ``window_m / resample_m`` rounded to the nearest whole number with ties
        rounded up (at least 1). The tile corners lie on multiples of
        resample_m, and for odd n the pixel that contains the point is the
        centre pixel (``geo.tile_pixel_count`` and ``geo.tile_top_left``, which
        the GEE adapter also uses). This means local and GEE tiles for the same
        point at the same resample_m are spatially aligned and directly
        comparable. Without resample_m, the tile is written in the source
        raster's native CRS at native resolution.

        ``filename_suffix`` is inserted before the .tif extension so multi-
        window runs can place every window's tiles in the same folder
        without overwriting one another. Pass ``None`` for the original
        ``"<id>-<dataset>.tif"`` naming.

        Returns ``(paths, meta_list)``, one entry per point. ``paths[i]`` is
        ``None`` when no tile was written: the point is outside the extent,
        its window has no pixels, or a read failed. For a failed read,
        ``meta_list[i]`` has the error text (``error``) and the exception
        (``exception``).
        """
        from rasterio.transform import Affine
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import Resampling, reproject

        # build_dataset_meta reads this to record the CRS of the tiles.
        self._export_resample_m = resample_m

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

        paths: list[Any] = []
        # One meta dict per tile, mirroring fetch_values' meta. Carries the
        # per-point QC info (in_extent, n_pixels, coverage_pct, …) so the
        # raster-mode QC summary can report which tiles failed and why.
        tile_metas: list[dict] = []

        n_pixels = tile_pixel_count(window_m, resample_m) if resample_m is not None else None

        # Build every tile filename up front, using the same shared helper as
        # the Earth Engine adapter so both sources name their tiles
        # identically. Sample IDs come from the user's input table and can
        # contain anything (URL-style occurrenceIDs with "/" in them,
        # "urn:catalog:..." IDs with ":"), so they are sanitized into a
        # character set that is valid on Windows, macOS and Linux alike. When
        # the caller passes "200m", filenames become "<id>-<dataset>-200m.tif";
        # with no suffix we keep the historical "<id>-<dataset>.tif" naming,
        # and IDs that were already safe are left untouched, unless two of them
        # differ only in letter case (then both get a hash suffix).
        tile_filenames = build_tile_filenames(id_list, dataset_name, filename_suffix)

        # Wrap the per-point tile loop with tqdm so users get visible progress
        # for what can be a long sequential operation on large input sets.
        emit_progress_step(progress_callback, 0, len(id_list))
        for lat, lon, tile_filename in tqdm(
            zip(lats_list, lons_list, tile_filenames),
            total=len(id_list),
            desc=progress_desc or "Local tiles",
            unit="tile",
            disable=disable_progress,
        ):
            output_path = output_dir / tile_filename
            _, meta = self.fetch_values(lat, lon, window_m, return_meta=True)

            window_array = meta.get("window_arr")
            transform_list = meta.get("transform")
            # Skip tiles for which the window couldn't be cropped at all
            # (e.g. the point fell outside the raster's extent, or the read
            # failed). Preserve the meta dict so callers can still see why
            # each tile failed; a failed read has "error" in it.
            if window_array is None or window_array.size == 0 or transform_list is None:
                paths.append(None)
                tile_metas.append(meta)
                emit_progress_step(progress_callback, len(paths), len(id_list))
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
                # Build the output grid in the point's UTM zone with the
                # shared tile helpers, as the GEE adapter does, so that local
                # and GEE tiles of the same point are spatially comparable:
                #   1. Pixel size = resample_m (not native-span / n_pixels, which varies
                #      per point because rio_mask snaps to native pixel boundaries).
                #   2. Corners on multiples of resample_m, with the point's pixel
                #      at the centre for odd n_pixels (geo.tile_top_left).
                utm_crs = get_utm_crs(lon, lat)
                utm_transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
                point_x_m, point_y_m = utm_transformer.transform(lon, lat)
                tile_left_m, tile_top_m = tile_top_left(point_x_m, point_y_m, n_pixels, resample_m)
                destination_transform = Affine(
                    resample_m,
                    0.0,
                    tile_left_m,
                    0.0,
                    -resample_m,
                    tile_top_m,
                )
                # Reproject directly from the open source dataset rather than from the
                # polygon-masked window_array. The polygon mask sets edge pixels to NaN
                # wherever the native pixel centre falls outside the 200 m polygon;
                # bilinear interpolation at output cells near the window boundary then
                # pulls from those NaN source pixels and propagates NaN into the output.
                # Going straight to the source file gives the interpolation valid data
                # everywhere, matching what GEE does when it exports a plain rectangle.
                #
                # A rasterio.band() source gives reproject() the CRS and nodata
                # value of the file, and reproject() ignores src_crs for such a
                # source. When envoi uses the catalog crs or nodata (the file
                # declares none), read the file through a WarpedVRT that
                # declares them. Without it, a file without a CRS cannot be
                # warped, and the bilinear interpolation would mix the catalog
                # nodata pixels into the tile as data.
                # The VRT keeps the grid of the file (its transform, width,
                # and height). Without them, WarpedVRT computes a north-up
                # grid, so a rotated or skewed raster would be resampled
                # twice: once into the VRT, and once into the tile.
                if self._uses_catalog_settings:
                    vrt_options: dict[str, Any] = {
                        "src_crs": self.raster_crs,
                        "transform": self.src.transform,
                        "width": self.src.width,
                        "height": self.src.height,
                    }
                    if self._nodata_source == "catalog":
                        vrt_options["src_nodata"] = self._nodata
                    resample_source = WarpedVRT(self.src, **vrt_options)
                else:
                    resample_source = contextlib.nullcontext(self.src)
                try:
                    with resample_source as source_dataset:
                        if self._is_multiband:
                            destination_array = np.empty(
                                (n_bands, n_pixels, n_pixels), dtype=tile_dtype
                            )
                            for band_idx, band_num in enumerate(self.band):
                                reproject(
                                    source=rasterio.band(source_dataset, band_num),
                                    destination=destination_array[band_idx],
                                    dst_transform=destination_transform,
                                    dst_crs=utm_crs,
                                    resampling=Resampling.bilinear,
                                    dst_nodata=tile_nodata,
                                )
                        else:
                            destination_array = np.empty((n_pixels, n_pixels), dtype=tile_dtype)
                            reproject(
                                source=rasterio.band(source_dataset, self.band[0]),
                                destination=destination_array,
                                dst_transform=destination_transform,
                                dst_crs=utm_crs,
                                resampling=Resampling.bilinear,
                                dst_nodata=tile_nodata,
                            )
                except RasterioIOError as error:
                    # Reading the source for this tile failed, for example
                    # because a block of the file is damaged. This is a failed
                    # tile: no file is written, and the meta dict keeps the
                    # error, so extract() can report the tile.
                    meta["error"] = format_point_error(error)
                    meta["exception"] = error
                    paths.append(None)
                    tile_metas.append(meta)
                    emit_progress_step(progress_callback, len(paths), len(id_list))
                    continue
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
            emit_progress_step(progress_callback, len(paths), len(id_list))

        return paths, tile_metas

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    def build_dataset_meta(
        self,
        spec: dict[str, Any],
        meta_list: list | None = None,
        exported_paths: list | None = None,
        quality: dict | None = None,
        lats: Sequence[float] | None = None,
        lons: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Build per-dataset metadata using this adapter's local raster state.

        No date dimension and a single native CRS, so this is much simpler
        than the GEE implementation. Quality stats are added when present.

        ``tile_crs`` is the raster CRS, except after an ``export_tiles`` call
        with ``resample_m``: then it is the sorted list of the UTM zones of
        the points in ``lats`` and ``lons`` (EPSG codes), as for Earth Engine.

        ``nodata`` is the nodata value that envoi used (a number, text for NaN
        or infinity, or None), and ``nodata_source`` tells where it came from
        (``"file"``, ``"catalog"``, or None).

        Uses attributes:
            _export_resample_m: ``resample_m`` of the last ``export_tiles``
                call, or None.
            _nodata, _nodata_source: The result of
                ``_resolve_local_raster_settings``.
        """
        # Static dataset info from the catalog spec.
        meta: dict[str, Any] = {
            "data_source": spec.get("data_source"),
            "path": spec.get("path"),
            "asset_type": "local_raster",
        }
        if spec.get("data_type"):
            meta["data_type"] = spec["data_type"]

        # Native CRS and pixel width in metres (converted from degrees or feet
        # when the CRS uses them; None for a CRS without a length unit).
        meta["native_crs"] = str(self.raster_crs)
        native_pixel_size_m = self._native_pixel_size_m
        meta["native_spatial_resolution_m"] = (
            round(native_pixel_size_m, 2) if native_pixel_size_m is not None else None
        )

        # Tiles with resample_m are written in the UTM zone of each point, as
        # on Earth Engine, so list every zone of the input points. Without
        # resample_m, tiles keep the file's native CRS.
        if self._export_resample_m is not None and lats is not None and lons is not None:
            meta["tile_crs"] = build_tile_crs_zones(lats, lons)
        else:
            meta["tile_crs"] = str(self.raster_crs)

        # The nodata value that envoi used for the selected bands, and where it
        # came from: "file", "catalog" (the file declares none), or None (no
        # nodata value). JSON has no NaN or infinity, so a non-finite value is
        # recorded as text ("nan", "inf", or "-inf").
        if isinstance(self._nodata, float) and not np.isfinite(self._nodata):
            meta["nodata"] = str(self._nodata)
        else:
            meta["nodata"] = self._nodata
        meta["nodata_source"] = self._nodata_source

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
