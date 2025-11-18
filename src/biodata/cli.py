import argparse
from pathlib import Path
import sys
import pandas as pd

# Existing imports
from .enrich import enrich
import json
from datetime import datetime
import yaml
from .output import write_run_manifest


def main():
    ap = argparse.ArgumentParser("biodata")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enrich", help="Enrich points with environmental predictors")
    e.add_argument("--in", dest="inp", required=True, help="Input CSV file with id,lat,lon[,date]")
    e.add_argument("--out", dest="out", required=True, help="Output directory (groups) or file (flat)")
    e.add_argument("--catalog", default="configs/catalog.yml")
    e.add_argument("--predictors", help="Comma-separated predictor names (flat mode)")
    e.add_argument("--groups", help="YAML file defining groups (alternative to --predictors)")
    e.add_argument("--window_m", type=int, default=500)
    e.add_argument("--temporal", default="nearest_month")

    args = ap.parse_args()
    df = pd.read_csv(args.inp)

    if args.groups:
        # Groups mode → out is a directory
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = enrich(
            df,
            groups=args.groups,
            catalog=args.catalog,
            window_m=args.window_m,
            temporal=args.temporal,
            out_dir=out_dir,
        )
        for k, p in outputs.items():
            print(f"[groups] wrote {k}: {p}")
    elif args.predictors:
        # Flat mode → out is a single file
        predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        enrich(
            df,
            predictors=predictors,
            catalog=args.catalog,
            window_m=args.window_m,
            temporal=args.temporal,
            out_path=out_path,
        )
        print(f"[flat] wrote: {out_path}")
    else:
        raise ValueError("You must provide either --predictors or --groups")

    elif args.cmd == "prefect":
        # Ensure repo root and src/ are importable so we can import the flow
        ROOT = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(ROOT))          # allow `from flow...` import
        sys.path.insert(0, str(ROOT / "src"))  # allow `from biodata...` import

        # Import the Prefect wrapper (filename: prefect_biodiversity_pipeline.py)
        try:
            from flow.prefect_biodiversity_pipeline import run_from_cli, biodata_enrichment_flow
        except Exception as e:
            raise ImportError(
                "Could not import flow.prefect_biodiversity_pipeline. Make sure the file exists at "
                "flow/prefect_biodiversity_pipeline.py and that you run this command from the repo root."
            ) from e

        if args.serve:
            # Optional: serve the flow on a schedule (requires running agent or Prefect server)
            biodata_enrichment_flow.serve(
                name="biodata_enrichment_groups",
                cron=args.cron or "0 0 * * 0",  # weekly Sunday midnight
            )
        else:
            run_from_cli({
                "input": args.input,
                "groups": args.groups,
                "catalog": args.catalog,
                "window": args.window,
                "out": args.out,
            })

    else:
        ap.print_help()

if __name__ == "__main__":
    main()
