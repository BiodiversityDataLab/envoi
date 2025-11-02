# src/biodata/config.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml

REQUIRED_DATASET_KEYS = {"source", "path", "type", "crs"}


class CatalogError(ValueError):
    """Invalid or incomplete catalog.yaml."""


def _require_keys(d: Dict[str, Any], required: set, ctx: str) -> None:
    missing = required - set(d.keys())
    if missing:
        raise CatalogError(f"{ctx}: missing required key(s): {sorted(missing)}")


def load_catalog(path: str | Path) -> Dict[str, Any]:
    """
    Load and validate the predictor catalog YAML.
    Required structure:
      datasets:
        <predictor_name>:
          source: <str>
          path: <str>
          type: <str>        # e.g., 'raster' | 'numeric' | ...
          crs: <str>         # e.g., 'EPSG:4326'
          # optional: default_reducer, resolution_m, ...
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

    # Per-dataset validation
    for name, spec in data["datasets"].items():
        if not isinstance(spec, dict):
            raise CatalogError(f"datasets.{name}: must be a mapping")
        _require_keys(spec, REQUIRED_DATASET_KEYS, f"datasets.{name}")

        # minimal sanity checks
        if not isinstance(spec["source"], str) or not spec["source"]:
            raise CatalogError(f"datasets.{name}.source must be a non-empty string")
        if not isinstance(spec["path"], str) or not spec["path"]:
            raise CatalogError(f"datasets.{name}.path must be a non-empty string")
        if not isinstance(spec["type"], str) or not spec["type"]:
            raise CatalogError(f"datasets.{name}.type must be a non-empty string")
        if not isinstance(spec["crs"], str) or not spec["crs"].upper().startswith("EPSG:"):
            raise CatalogError(f"datasets.{name}.crs must look like 'EPSG:XXXX'")

    return data
