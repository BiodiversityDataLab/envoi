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

    # NEW: very simple “demo” raster writer used by groups kind="raster"
    def write_raster_demo(
        self,
        values,
        lats,
        lons,
        group_name: str,
        feature_name: str,
    ) -> Path:
        """
        Write a tiny 1-row GeoTIFF that encodes the per-point values.

        This is mainly for visualization / debugging (viz_tiles groups), not
        a production interpolation method.
        """
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        vals = np.array(list(values), dtype="float32")
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)  # (rows=1, cols=N)

        nrows, ncols = vals.shape

        # crude georeferencing: span the observed lon/lat range
        lons = list(lons)
        lats = list(lats)
        if not lons or not lats:
            raise ValueError("write_raster_demo: empty lat/lon sequence")

        west = float(min(lons))
        east = float(max(lons))
        north = float(max(lats))
        south = float(min(lats))

        # avoid zero pixel size
        pixel_width = max((east - west) / max(ncols, 1), 1e-6)
        pixel_height = max((north - south) / max(nrows, 1), 1e-6)

        transform = from_origin(west, north, pixel_width, pixel_height)

        out_path = self.out_dir / f"{group_name}_{feature_name}.tif"

        profile = {
            "driver": "GTiff",
            "height": int(nrows),
            "width": int(ncols),
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",  # lat/lon
            "transform": transform,
            "nodata": np.nan,
            "tiled": True,
            "compress": "LZW",
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(vals, 1)

        return out_path


def write_group_parquet(
    df: pd.DataFrame,
    group_name: str,
    meta_info: dict,
    config: dict,
) -> Path:
    """
    Write a Parquet file for a group and a sidecar JSON with metadata.

    `meta_info` is a free-form dict, typically containing:
      - "provenance": {feature -> ...}
      - "coverage_backlog": {feature -> {buffer -> counts}}

    `config` provides run-level context (CRS, thresholds, etc.).
    """
    om = OutputManager(config.get("out_dir", "out"))
    path = om.write_tabular(df, group_name)

    meta_path = path.with_name(f"{group_name}_metadata.json")

    meta = {
        "group": group_name,
        "project_crs": config.get("project_crs", "EPSG:3006"),
        "min_coverage_pct": config.get("min_coverage_pct", 80),
        "summary_statistics": config.get("summary_statistics"),
        "buffer_sizes": config.get("buffer_sizes"),
    }

    if isinstance(meta_info, dict):
        # merge provenance, coverage_backlog, etc.
        meta.update(meta_info)

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
