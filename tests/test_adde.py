from pathlib import Path

import pandas as pd

from biodata.enrich import enrich

tmp_path = Path("tests/output")
df = pd.read_csv(Path("/home/adrba603/repos/EDDP/data/for_testing/adrian_example.csv"))

cfg = {
    "name": "gee_testing",
    "predictors": ["dem_aster", "bioclim"],
    "output": {
        "kind": "raster",
        "window_m": 200,
    },
}

outputs = enrich(df, cfg, out_dir=tmp_path)
print(outputs)
