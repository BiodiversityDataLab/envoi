# flow/prefect_biodiversity_pipeline.py
from __future__ import annotations
from pathlib import Path
import sys
from typing import Dict, Optional, Any
from prefect import flow, task, get_run_logger


# -----------------------------------------------------------------------------
# Paths and Imports
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # repository root
SRC = ROOT / "src"                           # your package root (src/biodata)
CONFIGS = ROOT / "configs"                   # configs/catalog.yml, groups.yml
DATA_DIR = ROOT / "data"                     # input CSVs
OUT = ROOT / "out"                           # outputs live here
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))  # Ensure `from biodata.* import ...` works


from biodata.enrich import enrich  # core enrichment function [groups mode supported]


def load_groups_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Prefect task wrappers
# -----------------------------------------------------------------------------


@task(name="Load Input CSV", retries=1, retry_delay_seconds=5)
def t_load_input(input_csv: str):
    import pandas as pd
    log = get_run_logger()
    log.info(f"Loading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    required = {"id", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    log.info(f"✓ Loaded {len(df)} rows")
    return df


@task(name="Load Groups", retries=1, retry_delay_seconds=5)
def t_load_groups(groups_yaml: str) -> Dict[str, Any]:
    log = get_run_logger()
    log.info(f"Loading groups: {groups_yaml}")
    groups = load_groups_yaml(groups_yaml)
    try:
        group_names = list(groups.keys())[:5]
        log.info(f"✓ Groups loaded: {group_names}")
    except Exception:
        log.info("✓ Groups loaded")
    return groups


@task(name="Enrich (groups mode)", retries=1, retry_delay_seconds=5)
def t_enrich_groups(
    df,
    catalog_yaml: str,
    groups_cfg: Dict[str, Any],
    window_m: int,
    out_dir: str,
) -> Dict[str, str]:
    print("DEBUG t_enrich_groups catalog_yaml type:", type(catalog_yaml), catalog_yaml)  # <--- DEBUG PRINT
    log = get_run_logger()
    log.info(f"Enriching {len(df)} points | window={window_m}m | out={out_dir}")
    outputs = enrich(
        df=df,
        groups=groups_cfg,
        catalog=catalog_yaml,  # Pass YAML path, not dict!
        window_m=window_m,
        out_dir=out_dir,
    )
    if isinstance(outputs, dict):
        for k, v in outputs.items():
            log.info(f"✓ Wrote {k}: {v}")
        return outputs
    else:
        out_path = str(Path(out_dir) / "enriched.parquet")
        try:
            outputs.to_parquet(out_path)  # type: ignore[attr-defined]
            log.info(f"✓ Wrote {out_path}")
            return {"output": out_path}
        except Exception:
            log.info("Note: enrich returned non-dict and may have written files itself")
            return {"output": str(out_dir)}


# -----------------------------------------------------------------------------
# Prefect flow (UPDATED FOR THE TEST)
# -----------------------------------------------------------------------------
@flow(name="Biodata Enrichment (Groups)")
def biodata_enrichment_flow(
    input_csv: str = str(DATA_DIR / "points.csv"),
    groups_yaml: str = str(CONFIGS / "groups.yml"),
    catalog_yaml: str = str(CONFIGS / "catalog.yml"),
    window_m: int = 500,
    out_dir: str = str(OUT),
) -> Dict[str, str]:
    """
    Orchestrates the biodata enrich pipeline in 'groups' mode using your src/biodata package.
    """
    log = get_run_logger()
    log.info("🚀 Starting Biodata Enrichment (Groups mode)")

    # --- THIS IS THE TEST ---
    print("--- EXECUTING THE NEWEST VERSION OF THE FLOW ---")
    print(f"--- Passing catalog_yaml: {catalog_yaml} (type: {type(catalog_yaml)}) ---")
    # --- END OF TEST ---

    df = t_load_input(input_csv)
    groups = t_load_groups(groups_yaml)

    # This is the corrected call, passing the file path string
    outputs = t_enrich_groups(df, catalog_yaml, groups, window_m, out_dir)

    log.info("✅ Flow complete")
    return outputs


# -----------------------------------------------------------------------------
# CLI entrypoints (can be called from biodata/cli.py or directly)
# -----------------------------------------------------------------------------
def run_from_cli(args: Optional[Dict[str, Any]] = None):
    params = args or {}
    return biodata_enrichment_flow(
        input_csv=str(params.get("input", DATA_DIR / "points.csv")),
        groups_yaml=str(params.get("groups", CONFIGS / "groups.yml")),
        catalog_yaml=str(params.get("catalog", CONFIGS / "catalog.yml")),
        window_m=int(params.get("window", 500)),
        out_dir=str(params.get("out", OUT)),
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run Prefect-wrapped Biodata enrichment (groups mode)")
    p.add_argument("--input", default=str(DATA_DIR / "points.csv"))
    p.add_argument("--groups", default=str(CONFIGS / "groups.yml"))
    p.add_argument("--catalog", default=str(CONFIGS / "catalog.yml"))
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--serve", action="store_true", help="Serve as a scheduled deployment")
    p.add_argument("--cron", default=None, help="Cron for serve mode (e.g. '0 0 * * 0')")
    a = p.parse_args()


    if a.serve:
        biodata_enrichment_flow.serve(
            name="biodata_enrichment_groups",
            cron=a.cron or "0 0 * * 0",  # default weekly on Sunday at 00:00
        )
    else:
        biodata_enrichment_flow(
            input_csv=a.input,
            groups_yaml=a.groups,
            catalog_yaml=a.catalog,
            window_m=a.window,
            out_dir=a.out,
        )
