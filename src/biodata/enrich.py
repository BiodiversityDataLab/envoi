from typing import List, Optional
import pandas as pd

def enrich(
    df: pd.DataFrame,
    predictors: List[str],
    catalog: str = "configs/catalog.yml",
    window_m: int = 500,
    temporal: str = "nearest_month",
    out_path: Optional[str] = None,
    cache_dir: str = "~/.biodata_cache",
) -> pd.DataFrame:
    """MVP stub: validates input, adds a placeholder column, writes Parquet if requested.
    TODO: wire adapters, reducers, provenance, QC.
    """
    required = {"id", "lat", "lon"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    out["dummy_predictor"] = 0  # placeholder so CLI works end-to-end
    if out_path:
        if out_path.endswith(".parquet"):
            out.to_parquet(out_path, index=False)
        else:
            out.to_csv(out_path, index=False)
    return out
