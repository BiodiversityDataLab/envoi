# src/biodata/qc.py
from __future__ import annotations
import logging
import pandas as pd

_DATE_META_KEYS = ("image_date_used", "date_clamped", "date_source")


def compute_qc_flags(meta_list, min_coverage_pct: int = 80) -> pd.DataFrame:
    """meta_list: list of dicts returned by adapters for each row"""
    df = pd.DataFrame(meta_list)
    low = df["coverage_pct"] < float(min_coverage_pct)
    if low.any():
        logging.warning("Low coverage for %d sample(s) (<%s%%).", int(low.sum()), min_coverage_pct)
    return df[["in_extent", "n_pixels", "had_nodata", "coverage_pct"]]


def extract_crs_column(meta_list: list[dict]) -> pd.DataFrame:
    """Extract the per-point region CRS from adapter meta dicts into a DataFrame.

    Returns a single-column DataFrame with column 'region_crs', or empty if
    no CRS info is present. No GEE calls — reads from already-collected meta dicts.
    """
    if not meta_list or "region_crs" not in meta_list[0]:
        return pd.DataFrame()
    return pd.DataFrame({"region_crs": [m.get("region_crs") for m in meta_list]})


def extract_date_columns(meta_list: list[dict]) -> pd.DataFrame:
    """Extract per-point date info from adapter meta dicts into a DataFrame.

    Returns columns: image_date_used, date_clamped, date_source.
    Returns an empty DataFrame if no date info is present in the meta dicts.
    No GEE calls — reads only from the already-collected meta dicts.
    """
    if not meta_list or _DATE_META_KEYS[0] not in meta_list[0]:
        return pd.DataFrame()
    rows = [{k: m.get(k) for k in _DATE_META_KEYS} for m in meta_list]
    return pd.DataFrame(rows)
