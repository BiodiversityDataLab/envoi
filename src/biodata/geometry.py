# src/biodata/geometry.py
from __future__ import annotations
import pandas as pd
from pyproj import Transformer


def df_to_project_crs(df: pd.DataFrame, target_epsg: int = 3006) -> pd.DataFrame:
    """Return a copy of df with x,y columns in project CRS (meters)."""
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
    x, y = tr.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    out = df.copy()
    out["x"], out["y"] = x, y
    return out
