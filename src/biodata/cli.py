import argparse
import pandas as pd
from .enrich import enrich


def main():
    ap = argparse.ArgumentParser("biodata")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enrich", help="Enrich points with environmental predictors (MVP stub)")
    e.add_argument("--in", dest="inp", required=True, help="Input CSV file with id,lat,lon[,date]")
    e.add_argument("--out", dest="out", required=True, help="Output Parquet/CSV path")
    e.add_argument("--catalog", default="configs/catalog.yml")
    e.add_argument("--predictors", required=True, help="Comma-separated predictor names")
    e.add_argument("--window_m", type=int, default=500)
    e.add_argument("--temporal", default="nearest_month")

    args = ap.parse_args()
    df = pd.read_csv(args.inp)
    predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
    enrich(
        df,
        predictors=predictors,
        catalog=args.catalog,
        window_m=args.window_m,
        temporal=args.temporal,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
