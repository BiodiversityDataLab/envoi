# src/biodata/output.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


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
