# src/biodata/metadata.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone


def build_feature_meta(spec: dict, adapter) -> dict:
    """Build per-feature metadata from catalog spec and adapter state."""
    meta = {
        "source": spec.get("source"),
        "path": spec.get("path"),
    }

    # Native CRS and scale from local adapter
    if hasattr(adapter, "raster_crs"):
        meta["native_crs"] = str(adapter.raster_crs)
    if hasattr(adapter, "src") and hasattr(adapter.src, "res"):
        meta["native_scale_m"] = float(adapter.src.res[0])

    # Native CRS and scale from GEE adapter
    if hasattr(adapter, "crs"):
        meta["native_crs"] = str(adapter.crs)
    if hasattr(adapter, "_cached_native_scale"):
        meta["native_scale_m"] = float(adapter._cached_native_scale)
    elif hasattr(adapter, "scale") and adapter.scale is not None:
        meta["native_scale_m"] = float(adapter.scale)

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
