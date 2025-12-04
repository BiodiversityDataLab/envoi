import argparse
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import yaml

from .enrich import enrich
from .output import write_run_manifest


def main():
    ap = argparse.ArgumentParser("biodata")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- enrich ---
    e = sub.add_parser("enrich", help="Enrich points with environmental predictors")
    e.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="Input CSV file with id,lat,lon[,date]",
    )
    e.add_argument(
        "--out",
        dest="out",
        required=True,
        help="Output directory (groups) or file (flat)",
    )
    e.add_argument("--catalog", default="configs/catalog.yml")
    e.add_argument(
        "--predictors",
        help="Comma-separated predictor names (flat mode)",
    )
    e.add_argument(
        "--groups",
        help="YAML file defining groups (alternative to --predictors)",
    )
    e.add_argument("--window_m", type=int, default=500)
    e.add_argument("--temporal", default="nearest_month")

    # --- rerun ---
    r = sub.add_parser(
        "rerun",
        help="Re-run a previous enrichment from a saved manifest",
    )
    r.add_argument(
        "--from",
        dest="manifest",
        default="out/last_run.json",
        help="Path to manifest JSON (default: out/last_run.json)",
    )

    # --- gee-download (optional, requires earthengine-api) ---
    g = sub.add_parser(
        "gee-download",
        help="Download GEE rasters into local GeoTIFFs based on catalog.yml",
    )
    g.add_argument(
        "--catalog",
        default="configs/catalog.yml",
        help="Catalog file with gee_raster datasets",
    )

    args = ap.parse_args()

    # ---------------------------
    # Command: gee-download
    # ---------------------------
    if args.cmd == "gee-download":
        try:
            from .gee_download import run_gee_download  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SystemExit(
                "GEE support is not installed.\n\n"
                "Install the extra dependency with:\n"
                "  pip install 'biodata-enricher[gee]'\n"
                "or add 'earthengine-api' to your environment."
            ) from exc

        run_gee_download(args.catalog)
        return

    # ---------------------------
    # Command: rerun
    # ---------------------------
    if args.cmd == "rerun":
        mpath = Path(args.manifest)
        if not mpath.exists():
            raise FileNotFoundError(f"Manifest not found: {mpath}")
        with mpath.open() as f:
            m = json.load(f)

        df = pd.read_csv(m["input_csv_path"])
        if m.get("mode") == "groups":
            outputs = enrich(
                df,
                groups=m["groups_config"],
                catalog=m["catalog_path"],
                out_dir=m["out_dir"],
                window_m=m.get("window_m", 500),
                temporal=m.get("temporal", "nearest_month"),
            )
            for k, p in outputs.items():
                print(f"[rerun:groups] wrote {k}: {p}")
        else:
            out_dir = Path(m.get("out_dir", "out"))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = m.get("out_path") or str(out_dir / "flat_rerun.parquet")
            enrich(
                df,
                predictors=m["predictors"],
                catalog=m["catalog_path"],
                out_path=out_path,
                window_m=m.get("window_m", 500),
                temporal=m.get("temporal", "nearest_month"),
            )
            print(f"[rerun:flat] wrote: {out_path}")
        return

    # ---------------------------
    # Command: enrich
    # ---------------------------
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

        # Save manifest for replay
        manifest = {
            "timestamp": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "mode": "groups",
            "input_csv_path": args.inp,
            "catalog_path": args.catalog,
            "out_dir": str(out_dir),
            "groups_config": yaml.safe_load(open(args.groups, encoding="utf-8")),
            "window_m": args.window_m,
            "temporal": args.temporal,
        }
        write_run_manifest(manifest, out_dir)

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

        # Save manifest for replay (store out_dir as the parent folder)
        manifest = {
            "timestamp": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "mode": "flat",
            "input_csv_path": args.inp,
            "catalog_path": args.catalog,
            "out_dir": str(out_path.parent),
            "predictors": predictors,
            "window_m": args.window_m,
            "temporal": args.temporal,
            "out_path": str(out_path),
        }
        write_run_manifest(manifest, out_path.parent)

    else:
        raise ValueError("You must provide either --predictors or --groups")


if __name__ == "__main__":
    main()
