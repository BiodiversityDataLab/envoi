# src/biodata/metadata.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Sequence


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


def build_feature_meta(
    spec: dict,
    adapter,
    tile_crs_zones: list[str] | None = None,
) -> dict:
    """Build per-feature metadata from catalog spec and adapter state."""
    meta = {
        "source": spec.get("source"),
        "path": spec.get("path"),
    }

    # Asset type
    if spec.get("source") == "earth_engine":
        feature_spec = getattr(adapter, "_feature_spec", {})
        if "collection" in feature_spec:
            meta["asset_type"] = "IMAGE_COLLECTION"
        else:
            meta["asset_type"] = "IMAGE"
    else:
        meta["asset_type"] = "local_raster"

    # Native CRS
    if hasattr(adapter, "raster_crs"):
        meta["native_crs"] = str(adapter.raster_crs)
    elif hasattr(adapter, "crs"):
        meta["native_crs"] = str(adapter.crs)

    # Native spatial resolution
    # Priority: local file → user-specified scale → GEE cached proj (one .getInfo() call)
    res = None
    if hasattr(adapter, "src") and hasattr(adapter.src, "res"):
        res = float(adapter.src.res[0])
    elif hasattr(adapter, "scale") and adapter.scale is not None:
        res = float(adapter.scale)
    elif hasattr(adapter, "_native_proj") and adapter._native_proj is not None:
        try:
            res = float(adapter._native_proj.nominalScale().getInfo())
        except Exception:
            pass
    if res is not None:
        meta["native_spatial_resolution_m"] = round(res, 2)

    # Band names — available once _get_band_name has been called (after first fetch)
    band_names = getattr(adapter, "_cached_band_names", None)
    if band_names:
        meta["band_names"] = band_names

    # Tile CRS — how raster tiles are projected when exported
    if spec.get("source") == "earth_engine":
        if tile_crs_zones:
            meta["tile_crs"] = tile_crs_zones
    elif hasattr(adapter, "raster_crs"):
        meta["tile_crs"] = str(adapter.raster_crs)

    # Collection date range and date selection info
    timestamps = getattr(adapter, "_collection_timestamps", None)
    if timestamps is not None and len(timestamps) > 0:
        meta["collection_date_range"] = [
            timestamps.min().strftime("%Y-%m-%d"),
            timestamps.max().strftime("%Y-%m-%d"),
        ]
    date_source = getattr(adapter, "_date_source", None)
    if date_source:
        meta["date_source"] = date_source
    image_date_used = getattr(adapter, "_image_date_used", None)
    if image_date_used is not None:
        meta["image_date_used"] = image_date_used.strftime("%Y-%m-%d")

    if spec.get("license"):
        meta["license"] = spec["license"]

    return meta


def write_metadata(
    out_dir: str | Path,
    group_name: str,
    *,
    kind: str,
    n_points: int,
    features: dict,
    config: dict,
    quality: dict | None = None,
    date_info: dict | None = None,
) -> Path:
    """Write a sidecar metadata JSON for a group output.

    Structure:
      run       — when and how (auto-generated)
      config    — what the user requested
      features  — per-feature source details
      quality   — per-feature coverage summary (tabular only)
      date_info — per-feature date selection summary (GEE collections only)
    """
    from . import __version__

    meta = {
        "run": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "package_version": __version__,
            "n_points": n_points,
        },
        "config": {
            "name": group_name,
            **config,
        },
        "features": features,
    }

    if quality:
        meta["quality"] = quality

    if date_info:
        meta["date_info"] = date_info

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{group_name}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path
