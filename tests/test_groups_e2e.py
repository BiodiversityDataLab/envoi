from pathlib import Path
import json

import pandas as pd

from biodata.enrich import enrich


def test_groups_e2e(tmp_path: Path) -> None:
    """End-to-end smoke test for groups mode with a single DEM feature.

    Verifies:
    - Stats Parquet file is written for the group.
    - QC Parquet file is written for the group.
    - Reducer columns live only in the stats file.
    - QA columns live only in the QC file.
    - Row count is preserved.
    - Metadata JSON exists and contains provenance.
    """
    df = pd.read_csv(Path("data/points_sample.csv"))

    cfg = {
        "groups": [
            {
                "name": "dem_100m",
                "predictors": ["dem_mini"],
                "output": {
                    "kind": "tabular",
                    "reducers": ["mean", "std"],
                    "window_m": 100,
                },
            }
        ],
        "min_coverage_pct": 0,
    }

    outputs = enrich(df, groups=cfg, out_dir=tmp_path)

    stats_path = outputs["dem_100m"]
    qc_path = outputs["dem_100m_qc"]

    # --- stats parquet ---
    assert stats_path.exists()
    stats_df = pd.read_parquet(stats_path)
    assert len(stats_df) == len(df)

    # --- qc parquet ---
    assert qc_path.exists()
    qc_df = pd.read_parquet(qc_path)
    assert len(qc_df) == len(df)

    # Shared id column, row order preserved
    assert list(stats_df["id"]) == list(qc_df["id"]) == list(df["id"])

    # Expected reducer & QA columns (with buffer-suffix)
    reducer_cols = {"dem_mini_mean_b100", "dem_mini_std_b100"}
    qa_cols = {
        "dem_mini_in_extent_b100",
        "dem_mini_n_pixels_b100",
        "dem_mini_had_nodata_b100",
        "dem_mini_coverage_pct_b100",
    }

    stats_cols = set(stats_df.columns)
    qc_cols = set(qc_df.columns)

    # Reducers only in stats
    assert reducer_cols.issubset(stats_cols)
    assert reducer_cols.isdisjoint(qc_cols)

    # QA only in QC
    assert qa_cols.issubset(qc_cols)
    assert qa_cols.isdisjoint(stats_cols)

    # No legacy unsuffixed names anywhere
    legacy_cols = {
        "dem_mini_mean",
        "dem_mini_std",
        "dem_mini_in_extent",
        "dem_mini_coverage_pct",
    }
    assert legacy_cols.isdisjoint(stats_cols)
    assert legacy_cols.isdisjoint(qc_cols)

    # Metadata JSON (same name as stats group)
    meta_path = stats_path.with_name("dem_100m_metadata.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert "provenance" in meta
