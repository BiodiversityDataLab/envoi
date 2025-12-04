# src/biodata/gee_download.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

import yaml
import ee
import geemap

from .auth import init_gee  # you implement this to use your service account JSON


# ----------------------------------------------------------------------
# Catalog helpers
# ----------------------------------------------------------------------


def _load_catalog(path: str | Path) -> Dict[str, Any]:
    """Load catalog.yml as a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_gee_datasets(cat: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield (name, spec) for datasets that should be downloaded from GEE.

    We select datasets with:
      source: gee_raster
      gee: { ... }

    Skips any malformed entries and prints a hint.
    """
    datasets = cat.get("datasets", {})
    for name, spec in datasets.items():
        if not isinstance(spec, dict):
            print(f"[gee-download] skipping dataset '{name}': spec is {spec!r}")
            continue
        if spec.get("source") == "gee_raster" and "gee" in spec:
            yield name, spec


# ----------------------------------------------------------------------
# Geometry / image builders
# ----------------------------------------------------------------------


def _build_region_from_bbox(bbox: list[float]) -> ee.Geometry:
    """
    Build an Earth Engine rectangle from [xmin, ymin, xmax, ymax] in EPSG:4326.
    """
    return ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)


def _build_image(gee_cfg: Dict[str, Any]) -> ee.Image:
    """
    Build a GEE image from the 'gee' config block.

    Supports:
      - gee.image: single ee.Image asset (e.g. NASA/NASADEM_HGT/001)
      - gee.collection: ee.ImageCollection with optional date filters and reducer

    Optional keys:
      - band: band name to select
      - reducer: "mean" | "median" | "none" | "slope"
        * "slope" applies ee.Terrain.slope to the image (after band selection).
    """
    img: ee.Image | None = None

    # Case 1: single image asset
    if "image" in gee_cfg:
        img = ee.Image(gee_cfg["image"])

    # Case 2: collection asset
    elif "collection" in gee_cfg:
        col = ee.ImageCollection(gee_cfg["collection"])

        start = gee_cfg.get("start_date")
        end = gee_cfg.get("end_date")
        if start and end:
            col = col.filterDate(start, end)

        reducer = gee_cfg.get("reducer", "none")
        if reducer == "mean":
            img = col.mean()
        elif reducer == "median":
            img = col.median()
        else:
            # fall back to first image in collection
            img = col.first()
    else:
        raise ValueError("gee config must contain either 'image' or 'collection'")

    # Select band if specified
    band = gee_cfg.get("band")
    if band:
        img = img.select(band)

    # Optional post-processing based on reducer keyword
    if gee_cfg.get("reducer") == "slope":
        # derive slope (degrees) from elevation
        img = ee.Terrain.slope(img)

    return img


def _export_image(
    img: ee.Image,
    out_path: str | Path,
    crs: str,
    scale: float,
    region: ee.Geometry,
) -> None:
    """
    Export an image to a GeoTIFF using geemap.ee_export_image.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [export] scale={scale}, crs={crs}, path={out_path}")
    geemap.ee_export_image(
        img,
        filename=str(out_path),
        scale=scale,
        crs=crs,
        region=region,
    )


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def run_gee_download(catalog_path: str | Path) -> None:
    """
    Programmatic entry point used by cli.py.

    - Initializes GEE using your service account (via auth.init_gee()).
    - Reads catalog.yml.
    - For each dataset with source: gee_raster and a gee: block:
        * builds a GEE image
        * exports it as a GeoTIFF to the dataset's 'path'.
    """
    # 1) Initialize Earth Engine
    print("[gee-download] Initializing Earth Engine...")
    init_gee()  # your auth.py sets service account + key JSON [web:73]
    print("[gee-download] Earth Engine initialized.")

    # 2) Load catalog and defaults
    cat = _load_catalog(catalog_path)
    gee_defaults = cat.get("gee_defaults", {})  # optional global defaults

    default_crs = gee_defaults.get("crs", "EPSG:4326")
    default_scale = gee_defaults.get("scale", 1000)  # meters

    if "bbox" in gee_defaults:
        default_region = _build_region_from_bbox(gee_defaults["bbox"])
    else:
        # backup extent; adjust to your study area if you like
        default_region = ee.Geometry.Rectangle([-180.0, -60.0, 180.0, 80.0], proj="EPSG:4326")

    # 3) Process all gee_raster datasets
    any_found = False
    for name, spec in _iter_gee_datasets(cat):
        any_found = True
        gee_cfg = spec["gee"]

        ds_crs = spec.get("crs", default_crs)
        # prefer per-dataset resolution_m, else global default scale
        ds_scale = spec.get("resolution_m", default_scale)

        if "bbox" in gee_cfg:
            ds_region = _build_region_from_bbox(gee_cfg["bbox"])
        else:
            ds_region = default_region

        out_path = spec["path"]

        print(f"[gee-download] {name} -> {out_path}")
        try:
            img = _build_image(gee_cfg)
            _export_image(img, out_path, crs=ds_crs, scale=ds_scale, region=ds_region)
        except Exception as e:
            print(f"  [gee-download] ERROR for {name}: {e}")

    if not any_found:
        print("[gee-download] No gee_raster datasets found in catalog.")
    else:
        print("[gee-download] Completed.")
