# src/biodata/metadata.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
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
    if not meta_list or "image_date_used" not in meta_list[0]:
        return None
    sources = [m.get("date_source", "") for m in meta_list]
    dates_used = sorted({m["image_date_used"] for m in meta_list if m.get("image_date_used")})
    return {
        "n_nearest_to_sample": sum(1 for s in sources if s == "nearest_to_sample"),
        "n_clamped_to_nearest": sum(1 for s in sources if s == "clamped_to_nearest"),
        "n_most_recent_no_date": sum(1 for s in sources if s == "most_recent_no_date"),
        "image_dates_used": dates_used,
    }


def build_tile_crs_zones(lats: Sequence[float], lons: Sequence[float]) -> list[str]:
    """Return the sorted unique UTM EPSG codes for a set of sample points.

    GEE raster tiles are exported in the UTM zone of each individual point,
    so the list of zones reflects exactly which CRSs appear in the output.
    """
    zones: set[str] = set()
    for lat, lon in zip(lats, lons):
        zone_number = int((lon + 180) / 6) + 1
        base_epsg = 32600 if lat >= 0 else 32700
        zones.add(f"EPSG:{base_epsg + zone_number}")
    return sorted(zones)


def write_metadata(
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

    meta = {
        "run": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
