import argparse
import pandas as pd
from biodata.enrich import enrich

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/points_sample.csv", help="Input CSV with id,lat,lon[,date]")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--catalog", default="configs/catalog.yml", help="Catalog file")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)


    features = ["dem_mini"]  # change to ["dem_elev","slope"] if present

    cfg = {
        "groups": [{
            "name": "quick_demo",
            "features": features,                       # formerly "predictors"
            "summary_statistics": ["mean","std","q10","q90"],  # formerly "reducers"
            "buffer_sizes": [100, 500, 1000]                 # meters; add more if you want
        }],
        "project_crs": "EPSG:3006", # Change
    }

    outs = enrich(df, groups=cfg, catalog=args.catalog, out_dir=args.out)
    print("Wrote:", outs)

    g = pd.read_parquet(f"{args.out}/quick_demo.parquet")
    print(g.head().to_string(index=False))

if __name__ == "__main__":
    main()


# compute meta data in its own file
# QA in one file
# Join later on