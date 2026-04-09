# src/biodata/metadata.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Sequence


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
) -> Path:
    """Write a sidecar metadata JSON for a group output.

    Structure:
      run      — when and how (auto-generated)
      config   — what the user requested
      features — per-feature source details
      quality  — per-feature coverage summary (tabular only)
    """
    from . import __version__

    meta = {
        "run": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "package_version": __version__,
            "n_points": n_points,
        },
        "config": {
            "group": group_name,
            **config,
        },
        "features": features,
    }

    if quality:
        meta["quality"] = quality

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{group_name}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path
