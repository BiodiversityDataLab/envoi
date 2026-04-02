import pandas as pd
from pathlib import Path
from biodata.enrich import enrich

CATALOG = {
    "datasets": {
        "dem_local": {
            "source": "local_raster",
            "path": "data/for_testing/dem/TG4NHB-dem.tif",
        }
    }
}


def test_enrich_roundtrip_from_sample(tmp_path):
    sample_csv = Path("data/for_testing/adrian_example.csv")
    df = pd.read_csv(sample_csv)

    cfg = {
        "name": "dem_test",
        "predictors": ["dem_local"],
        "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 100},
    }

    outputs = enrich(df, cfg, catalog=CATALOG, out_dir=tmp_path)

    stats_path = outputs["dem_test"]
    result = pd.read_parquet(stats_path)

    assert {"id", "lat", "lon"}.issubset(result.columns)
    assert "dem_local_mean_b100" in result.columns
    assert len(result) == len(df)
