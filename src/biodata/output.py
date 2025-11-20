# src/biodata/output.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from datetime import datetime


class OutputManager:
    def __init__(self, out_dir: str | Path = "out"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_tabular(self, df: pd.DataFrame, name: str) -> Path:
        path = self.out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path


def write_group_parquet(df: pd.DataFrame, group_name: str, provenance: dict, config: dict) -> Path:
    om = OutputManager(config.get("out_dir", "out"))
    path = om.write_tabular(df, group_name)
    meta_path = path.with_name(f"{group_name}_metadata.json")
    meta = {
        "group": group_name,
        "provenance": provenance,
        "project_crs": config.get("project_crs", "EPSG:3006"),
        "min_coverage_pct": config.get("min_coverage_pct", 80),
        "reducers": config.get("reducers"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return path


def write_merged_parquet(outputs: dict[str, Path]) -> Path:
    frames = []
    for gname, path in outputs.items():
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            df["group"] = gname
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    merged = OutputManager().write_tabular(out, "merged")
    return merged


# --- History manifest writer ---
def write_run_manifest(manifest: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # default timestamp if not provided
    ts = manifest.get("timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    manifest["timestamp"] = ts

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    p_ts = runs_dir / f"run_{ts}.json"
    p_last = out_dir / "last_run.json"

    p_ts.write_text(json.dumps(manifest, indent=2))
    p_last.write_text(json.dumps(manifest, indent=2))
    return p_last


# --- Raster window TIFF writer ---
# Writes a single-band GeoTIFF file from a 2D array and georeference info
# Used for dumping sampled raster windows
# Returns the path to the written file
def write_window_tiff(arr, transform, crs, dtype, nodata, path):
    import numpy as np
    import rasterio
    from pathlib import Path

    # handle masked arrays
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(nodata)

    assert getattr(arr, "ndim", 0) == 2, "Expected 2-D window array"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": str(dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "compress": "LZW",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path
