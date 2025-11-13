# src/biodata/history.py

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .enrich import enrich

DEFAULT_MANIFEST = Path("out/last_run.json")


def replay_last_run(manifest_path: str | Path = DEFAULT_MANIFEST):
    """Re-run the last enrichment using a saved manifest (out/last_run.json by default)."""
    m = json.loads(Path(manifest_path).read_text())
    df = pd.read_csv(m["input_csv_path"])

    if m.get("mode") == "groups":
        return enrich(
            df,
            groups=m["groups_config"],
            catalog=m["catalog_path"],
            out_dir=m["out_dir"],
            window_m=m.get("window_m", 500),
            temporal=m.get("temporal", "nearest_month"),
        )

    # flat
    out_dir = Path(m.get("out_dir", "out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = m.get("out_path") or str(out_dir / "flat_rerun.parquet")
    return enrich(
        df,
        predictors=m["predictors"],
        catalog=m["catalog_path"],
        out_path=out_path,
        window_m=m.get("window_m", 500),
        temporal=m.get("temporal", "nearest_month"),
    )
