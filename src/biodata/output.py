# src/biodata/output.py
from __future__ import annotations
from pathlib import Path

import pandas as pd


class OutputManager:
    def __init__(self, out_dir: str | Path = "out", fmt: str = "parquet"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if fmt not in ("parquet", "csv"):
            raise ValueError(f"Unsupported output format: '{fmt}'. Use 'parquet' or 'csv'.")
        self.fmt = fmt

    def write_tabular(self, df: pd.DataFrame, name: str) -> Path:
        if self.fmt == "csv":
            path = self.out_dir / f"{name}.csv"
            df.to_csv(path, index=False)
        else:
            path = self.out_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
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
