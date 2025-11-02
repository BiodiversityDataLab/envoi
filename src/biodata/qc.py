# src/biodata/qc.py
from __future__ import annotations
import logging
import pandas as pd


def compute_qc_flags(meta_list, min_coverage_pct: int = 80) -> pd.DataFrame:
    """meta_list: list of dicts returned by adapters for each row"""
    df = pd.DataFrame(meta_list)
    low = df["coverage_pct"] < float(min_coverage_pct)
    if low.any():
        logging.warning("Low coverage for %d sample(s) (<%s%%).", int(low.sum()), min_coverage_pct)
    return df[["in_extent", "n_pixels", "had_nodata", "coverage_pct"]]
