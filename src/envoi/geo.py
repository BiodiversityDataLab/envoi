# src/envoi/geo.py
"""General-purpose CRS / geometry helpers used across adapters and metadata.

These functions resolve a WGS84 (lon, lat) into the appropriate UTM zone for
meter-accurate window construction. They are used by both the Earth Engine
adapter (per-point region geometry) and the local raster adapter (UTM
reprojection for tile export).

The tile helpers (``tile_pixel_count``, ``tile_top_left``) define the pixel
grid of a raster tile, so that both adapters export tiles of the same size at
the same place. ``pixel_size_m`` converts a raster's pixel width from CRS
units to metres. No function here makes a network request.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rasterio.errors import CRSError

if TYPE_CHECKING:
    from rasterio.crs import CRS

# Length of one degree at the equator, in metres (2π × 6 378 137 m / 360).
# Earth Engine uses this value for the nominal scale of assets in degrees, so a
# pixel size in degrees gives the same number of metres on both data sources.
_METRES_PER_DEGREE_AT_EQUATOR = 111_319.49


def get_utm_crs(lon: float, lat: float) -> str:
    """Return the EPSG code for the UTM zone covering (lon, lat).

    Raises :class:`ValueError` when ``(lon, lat)`` is outside the WGS84
    range. The naive zone formula yields 61 at ``lon == 180``; UTM only
    defines zones 1-60, so the result is clamped at that boundary.
    """
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"Invalid WGS84 coordinates: ({lon}, {lat})")
    zone_number = int((lon + 180) / 6) + 1
    # Clamp the lon == 180 edge case so the EPSG code stays valid.
    zone_number = min(zone_number, 60)
    base_epsg = 32600 if lat >= 0 else 32700
    return f"EPSG:{base_epsg + zone_number}"


def get_utm_zone_label(lon: float, lat: float) -> str:
    """Return the UTM zone label like ``"33N"`` or ``"34S"`` for a lon/lat point.

    Raises :class:`ValueError` when ``(lon, lat)`` is outside the WGS84 range.
    """
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"Invalid WGS84 coordinates: ({lon}, {lat})")
    # Plain 6° zones on purpose, without the UTM exceptions for south-west
    # Norway (zone 32V) and Svalbard (zones 31X, 33X, 35X, and 37X). The GEE
    # `use_utm_zone` filter compares this label with the `UTM_ZONE` property
    # of the satellite-embedding tiles, which uses the plain zones: a live
    # check on 2026-10-12 found Bergen tiles labelled "31N" and Ny-Ålesund
    # tiles labelled "32N" and "33N". With the exceptions, the label at Bergen
    # would be "32N", and the filter would select no tile there.
    zone_number = int((lon + 180) / 6) + 1
    zone_number = min(zone_number, 60)
    hemisphere = "N" if lat >= 0 else "S"
    return f"{zone_number}{hemisphere}"


def build_tile_crs_zones(lats: Sequence[float], lons: Sequence[float]) -> list[str]:
    """Return the sorted unique UTM EPSG codes for a set of sample points.

    GEE raster tiles are exported in the UTM zone of each individual point,
    so the list of zones reflects exactly which CRSs appear in the output.
    """
    zones = {get_utm_crs(lon, lat) for lat, lon in zip(lats, lons)}
    return sorted(zones)


# ---------------------------------------------------------------------------
# Raster tile grid  (shared by the Earth Engine and local tile exports)
# ---------------------------------------------------------------------------


def tile_pixel_count(window_m: float, pixel_m: float) -> int:
    """Return the number of pixels per side of a square raster tile.

    The count is ``window_m / pixel_m`` rounded to the nearest whole number,
    with ties rounded up, and at least 1. For example, a 200 m window gives
    7 pixels at 30 m and 20 pixels at 10 m, and a 45 m window at 30 m gives 2.

    Args:
        window_m: Requested tile side length in metres (``window_size_m``).
        pixel_m: Pixel size of the tile in metres (``resample_m``, or the
            native scale of an Earth Engine image).

    Returns:
        The number of pixels per side, at least 1.
    """
    # floor(x + 0.5) rounds ties up. Python's round() rounds ties to the even
    # number (round(2.5) == 2), which would make the size jump between odd
    # and even values.
    return max(1, math.floor(window_m / pixel_m + 0.5))


def tile_top_left(x_m: float, y_m: float, n_pixels: int, pixel_m: float) -> tuple[float, float]:
    """Return the top-left corner of an ``n_pixels`` × ``n_pixels`` tile around a point.

    The coordinates are projected metres in the point's UTM zone, easting
    (x) first. The tile edges lie on multiples of ``pixel_m``, so tiles of
    different points and of different data sources share one pixel grid.

    - When ``n_pixels`` is odd, the pixel that contains the point is the
      centre pixel of the tile.
    - When ``n_pixels`` is even, the centre of the tile is the grid line
      nearest to the point.

    Args:
        x_m: Easting of the point, in metres.
        y_m: Northing of the point, in metres.
        n_pixels: Pixels per side, from :func:`tile_pixel_count`.
        pixel_m: Pixel size in metres.

    Returns:
        ``(x_left, y_top)``: the west and north edges of the tile, in metres.
        The east edge is ``x_left + n_pixels * pixel_m``, and the south edge
        is ``y_top - n_pixels * pixel_m``.
    """
    tile_side_m = n_pixels * pixel_m

    # Round the west and south edges of a tile centred on the point to the
    # nearest grid line, ties up. With x = k·p + r (0 <= r < p), this gives
    # x_left = (k - (n - 1) / 2)·p for odd n: the point's pixel [k·p, (k+1)·p)
    # is then column (n - 1) / 2, the centre column. For even n, it gives a
    # tile centre of x rounded to the nearest grid line. The same holds for y.
    x_left = math.floor((x_m - tile_side_m / 2) / pixel_m + 0.5) * pixel_m
    y_bottom = math.floor((y_m - tile_side_m / 2) / pixel_m + 0.5) * pixel_m
    return x_left, y_bottom + tile_side_m


# ---------------------------------------------------------------------------
# Pixel size in metres
# ---------------------------------------------------------------------------


def pixel_size_m(crs: CRS | None, pixel_width: float) -> float | None:
    """Return a pixel width in metres, from a width in the units of ``crs``.

    - Geographic CRS: the width in the CRS's angular unit, converted to
      degrees, times 111 319.49 m per degree (the length of one degree at the
      equator, as Earth Engine uses for assets in degrees). Away from the
      equator, this is larger than the east-west width of the pixel on the
      ground, by the factor 1 / cos(latitude).
    - Projected CRS: the width times the CRS's linear unit factor
      (for example 0.3048 for feet).

    Args:
        crs: The raster's rasterio CRS, or None.
        pixel_width: Pixel width in CRS units (for example ``src.res[0]``).
            The sign is ignored.

    Returns:
        The pixel width in metres. None when ``crs`` is None, is neither
        geographic nor projected (for example an engineering CRS), or has
        units that GDAL cannot read.
    """
    if crs is None:
        return None
    pixel_width = abs(float(pixel_width))
    try:
        if crs.is_geographic:
            # units_factor gives radians per angular unit for a geographic CRS.
            _, radians_per_unit = crs.units_factor
            return pixel_width * math.degrees(radians_per_unit) * _METRES_PER_DEGREE_AT_EQUATOR
        if crs.is_projected:
            _, metres_per_unit = crs.linear_units_factor
            return pixel_width * metres_per_unit
    except CRSError:
        return None
    return None
