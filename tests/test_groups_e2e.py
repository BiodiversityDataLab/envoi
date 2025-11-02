import pandas as pd
from pathlib import Path
from biodata.enrich import enrich


def test_groups_e2e(tmp_path):
    df = pd.read_csv(Path("data/points_sample.csv"))
    cfg = {
        "groups": [
            {
                "name": "dem_100m",
                "predictors": ["dem_mini"],
                "output": {"kind": "tabular", "reducers": ["mean", "std"], "window_m": 100},
            }
        ],
        "min_coverage_pct": 0,
    }
    out = enrich(df, groups=cfg, out_dir=tmp_path)
    p = out["dem_100m"]
    assert p.exists()
    got = pd.read_parquet(p)
    assert {"dem_mini_mean", "dem_mini_std", "dem_mini_in_extent", "dem_mini_coverage_pct"} <= set(
        got.columns
    )
    meta = p.with_name("dem_100m_metadata.json")
    assert meta.exists()
