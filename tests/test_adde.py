from pathlib import Path
import json

import pandas as pd

from biodata.enrich import enrich

tmp_path = Path("tests/output")
df = pd.read_csv(Path("/home/adrba603/repos/EDDP/data/for_testing/adrian_example.csv"))

cfg = {
    "groups": [
        {
            "name": "gee_testing_resample_10m",
            "predictors": ["dem_aster"],
            "output": {
                "kind": "raster",
                "window_m": 200,
                "resample_m": 10,
            },
        }
    ],
}

outputs = enrich(df, groups=cfg, out_dir=tmp_path)

stats_path = outputs["dem_100m_gee_new"]

qc_path = outputs["dem_100m_gee_new_qc"]
print(pd.read_parquet(stats_path).to_string())
print(pd.read_parquet("tests/output/dem_100m.parquet").to_string())
print(pd.read_parquet(qc_path).to_string())