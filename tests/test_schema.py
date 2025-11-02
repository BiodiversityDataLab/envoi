import pandas as pd
from pathlib import Path
from biodata.enrich import enrich


def test_enrich_roundtrip_from_sample(tmp_path):
    sample_csv = Path("data/points_sample.csv")
    df = pd.read_csv(sample_csv)

    out = enrich(df, predictors=["dem_mini"], out_path=None)

    assert {"id", "lat", "lon"}.issubset(out.columns)
    assert "dem_mini" in out.columns
    assert len(out) == len(df)
