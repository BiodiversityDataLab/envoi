import pandas as pd


def test_enrich_stub_roundtrip(tmp_path):
    # Minimal smoke test for CLI stub behavior
    df = pd.DataFrame({"id": [1, 2], "lat": [59.86, 59.33], "lon": [17.64, 18.06]})
    from biodata.enrich import enrich

    out = enrich(df, predictors=["dem_elev"], out_path=None)
    assert "dummy_predictor" in out.columns
    assert len(out) == 2
