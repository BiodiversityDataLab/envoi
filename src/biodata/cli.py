import argparse
from pathlib import Path
import pandas as pd
from .enrich import enrich


def main():
    ap = argparse.ArgumentParser("biodata")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enrich", help="Enrich points with environmental predictors")
    e.add_argument("--in", dest="inp", required=True, help="Input CSV file with id,lat,lon[,date]")
    e.add_argument(
        "--out", dest="out", required=True, help="Output directory (groups) or file (flat)"
    )
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


if __name__ == "__main__":
    main()
