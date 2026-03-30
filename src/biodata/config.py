# src/biodata/config.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Mapping
import logging
import yaml

REQUIRED_DATASET_KEYS = {"source", "path"}

logger = logging.getLogger(__name__)


class CatalogError(ValueError):
    """Invalid or incomplete catalog.yaml."""


def _require_keys(d: Dict[str, Any], required: set, ctx: str) -> None:
    missing = required - set(d.keys())
    if missing:
        raise CatalogError(f"{ctx}: missing required key(s): {sorted(missing)}")


def _inspect_raster(name: str, spec: Dict[str, Any]) -> None:
    """Read CRS, resolution, type, and nodata from a local raster file
    and fill in any missing spec fields automatically."""
    if spec.get("source") != "local_raster":
        return

    p = Path(spec["path"])
    if not p.exists():
        return

    try:
        import rasterio

        with rasterio.open(p) as src:
            if "crs" not in spec:
                epsg = src.crs.to_epsg()
                if epsg:
                    spec["crs"] = f"EPSG:{epsg}"
                else:
                    spec["crs"] = str(src.crs)
                logger.debug("datasets.%s: auto-detected crs=%s", name, spec["crs"])

            if "resolution_m" not in spec:
                spec["resolution_m"] = abs(src.res[0])
                logger.debug("datasets.%s: auto-detected resolution_m=%s", name, spec["resolution_m"])

            if "type" not in spec:
                spec["type"] = "raster"

            if "nodata" not in spec and src.nodata is not None:
                spec["nodata"] = src.nodata

            if "bands" not in spec:
                spec["bands"] = src.count

    except Exception as e:
        logger.warning("datasets.%s: could not read raster metadata from %s: %s", name, p, e)


def load_catalog(path: str | Path) -> Dict[str, Any]:
    """
    Load and validate the predictor catalog YAML.
    Required structure:
      datasets:
        <predictor_name>:
          source: <str>
          path: <str>
          # optional: type, crs, resolution_m, default_reducer, band, ...
          # For local_raster sources, crs and resolution_m are auto-detected from the file.
    """
    p = Path(path)
    if not p.exists():
        raise CatalogError(f"Catalog file not found: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise CatalogError(f"Catalog YAML parse error in {p}: {e}") from e

    if not isinstance(data, dict):
        raise CatalogError(f"Top-level YAML must be a mapping (dict), got {type(data)}")

    if "datasets" not in data or not isinstance(data["datasets"], dict) or not data["datasets"]:
        raise CatalogError("Top-level key 'datasets' must be a non-empty mapping")

    # Per-dataset validation and auto-inspection
    for name, spec in data["datasets"].items():
        if not isinstance(spec, dict):
            raise CatalogError(f"datasets.{name}: must be a mapping")
        _require_keys(spec, REQUIRED_DATASET_KEYS, f"datasets.{name}")

        if not isinstance(spec["source"], str) or not spec["source"]:
            raise CatalogError(f"datasets.{name}.source must be a non-empty string")
        if not isinstance(spec["path"], str) or not spec["path"]:
            raise CatalogError(f"datasets.{name}.path must be a non-empty string")

        _inspect_raster(name, spec)

    return data


def _load_catalog_any(src: Any) -> Dict[str, Any]:
    """
    Internal helper: accept a path or a dict-like catalog and return
    a normalized {'datasets': {...}} structure.
    """
    if src is None:
        return {"datasets": {}}

    # If it's already a mapping, assume it's a parsed catalog dict.
    if isinstance(src, Mapping):
        d = dict(src)
        if "datasets" not in d:
            d["datasets"] = {}
        return d

    # Otherwise, treat it as a path and reuse the validated loader.
    return load_catalog(src)


def load_catalogs(*sources: Any) -> Dict[str, Any]:
    """
    Merge one or more catalogs (paths or dicts) into a single catalog.

    Later sources override earlier ones on a per-predictor basis.
    Always returns: {'datasets': {...}}.
    """
    merged: Dict[str, Any] = {"datasets": {}}

    for src in sources:
        cat = _load_catalog_any(src)
        for name, spec in cat.get("datasets", {}).items():
            # override or add
            merged["datasets"][name] = spec

    return merged
