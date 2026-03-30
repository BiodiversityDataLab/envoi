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

    out = enrich(df, predictors=["dem_local"], catalog=CATALOG, out_path=None)

    assert {"id", "lat", "lon"}.issubset(out.columns)
    assert "dem_local" in out.columns
    assert len(out) == len(df)
