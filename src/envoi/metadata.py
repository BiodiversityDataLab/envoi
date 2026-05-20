# src/envoi/metadata.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Sequence


def summarize_tile_export(exported_paths: list, n_points: int) -> dict:
    """Summarise the outcome of a raster tile export for inclusion in run metadata."""
    n_exported = sum(1 for ep in (exported_paths or []) if ep is not None)
    return {
        "n_exported": n_exported,
        "n_failed": n_points - n_exported,
        "total": n_points,
    }


def summarize_date_info(meta_list: list[dict]) -> dict | None:
    """Summarise per-point date decisions for inclusion in group metadata.

    Returns None when the meta dicts contain no date info (e.g. local rasters
    or IMAGE assets where date selection does not apply).
    """
    if not meta_list or "image_time_start" not in meta_list[0]:
        return None
    sources = [m.get("date_source", "") for m in meta_list]

    # Collect unique [start, end] pairs actually used, sorted by start date.
    # When image_time_end is absent (e.g. no system:time_end on the asset),
    # record the range as [start, start] so the field stays a consistent list
    # of two-element lists.
    seen: dict[tuple, None] = {}
    for m in meta_list:
        start = m.get("image_time_start")
        if start is None:
            continue
        end = m.get("image_time_end", start)
        seen[(start, end)] = None
    ranges_used = [[start, end] for start, end in sorted(seen)]

    return {
        "n_nearest_to_sample": sum(1 for s in sources if s == "nearest_to_sample"),
        "n_clamped_to_nearest": sum(1 for s in sources if s == "clamped_to_nearest"),
        "n_most_recent_no_date": sum(1 for s in sources if s == "most_recent_no_date"),
        "image_date_ranges_used": ranges_used,
    }


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


def write_metadata_sidecar(
    out_dir: str | Path,
    group_name: str,
    *,
    output_type: str,
    n_points: int,
    datasets: dict,
    config: dict,
    warnings: dict | None = None,
) -> Path:
    """Write a sidecar metadata JSON for a group output.

    Structure:
      run       — when and how (auto-generated)
      config    — what the user requested
      datasets  — per-dataset source details, including nested quality and
                  date_info where applicable (built by each adapter)
      warnings  — per-dataset warnings raised during the run (e.g. wrong reducer for data type)
    """
    from . import __version__

    # Use the system's local time with its UTC offset attached
    # (e.g. "2026-04-28T14:30:00+02:00"). astimezone() with no argument
    # tags the local datetime with the OS-configured timezone, so the
    # timestamp is unambiguous and correct for whoever runs the package.
    local_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    meta = {
        "run": {
            "timestamp": local_timestamp,
            "package_version": __version__,
            "n_points": n_points,
        },
        "config": {
            "batch_id": group_name,
            **config,
        },
        "datasets": datasets,
    }

    if warnings:
        meta["warnings"] = warnings

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{group_name}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path
