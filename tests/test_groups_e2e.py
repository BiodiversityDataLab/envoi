import json
from pathlib import Path

import pandas as pd

from biodata.enrich import enrich


def test_groups_e2e(tmp_path: Path) -> None:
    """End-to-end smoke test for groups mode with a single DEM feature.

    Verifies:
    - Parquet file is written for the group.
    - Buffer-suffixed reducer + QA columns are present.
    - No unsuffixed legacy columns are created.
    - Row count is preserved.
    - Metadata JSON exists and contains provenance + coverage_backlog.
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
    parquet_path = outputs["dem_100m"]

    # Parquet exists and has expected shape
    assert parquet_path.exists()
    got = pd.read_parquet(parquet_path)
    assert len(got) == len(df)

    # Expected buffer-suffixed columns
    expected_cols = {
        "dem_mini_mean_b100",
        "dem_mini_std_b100",
        "dem_mini_in_extent_b100",
        "dem_mini_n_pixels_b100",
        "dem_mini_had_nodata_b100",
        "dem_mini_coverage_pct_b100",
    }
    assert expected_cols.issubset(set(got.columns))

    # Ensure legacy unsuffixed names are NOT present anymore
    legacy_cols = {
        "dem_mini_mean",
        "dem_mini_std",
        "dem_mini_in_extent",
        "dem_mini_coverage_pct",
    }
    assert legacy_cols.isdisjoint(set(got.columns))

    # Metadata JSON exists and has the main keys we expect
    meta_path = parquet_path.with_name("dem_100m_metadata.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert "provenance" in meta
    prov = meta["provenance"]

    # coverage_backlog is stored inside provenance
    assert "coverage_backlog" in prov
    coverage_backlog = prov["coverage_backlog"]

    # basic sanity: we should have an entry for our feature
    assert "dem_mini" in coverage_backlog
