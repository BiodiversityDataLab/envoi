# src/envoi/geo.py
"""General-purpose CRS / geometry helpers used across adapters and metadata.

These functions resolve a WGS84 (lon, lat) into the appropriate UTM zone for
meter-accurate window construction. They are used by both the Earth Engine
adapter (per-point region geometry) and the local raster adapter (UTM
reprojection for tile export).
"""

from __future__ import annotations
from typing import Sequence


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
