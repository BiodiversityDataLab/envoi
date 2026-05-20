# src/envoi/config.py
from __future__ import annotations
import importlib.resources as importlib_resources
from pathlib import Path
from typing import Dict, Any, Mapping
import logging
import yaml

REQUIRED_DATASET_KEYS = {"data_source", "path"}
# Earth Engine datasets need a `data_type` so the right reducer set (continuous
# vs categorical) gets picked. Local rasters can omit it — type is inferred at
# read time from the raster file when needed.
EE_REQUIRED_DATASET_KEYS = {"data_type"}

# Module-level cache so each bundled YAML is only read once per process.
_defaults_cache: Dict[str, Any] | None = None

# Datasets registered by the user via update_catalog(). Persists for the
# duration of the Python session. Applied as the final merge layer in
# load_catalogs() so user entries always override built-in ones.
_user_catalog_datasets: Dict[str, Any] = {}

logger = logging.getLogger(__name__)


def _read_builtin_yaml(filename: str) -> Dict[str, Any]:
    """Read a YAML file bundled inside the envoi package under configs/.

    Uses importlib.resources so the file is found correctly whether the
    package is installed via pip or run directly from source.
    """
    ref = importlib_resources.files("envoi").joinpath("configs").joinpath(filename)
    text = ref.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def load_defaults() -> Dict[str, Any]:
    """Load project-wide defaults from the bundled configs/defaults.yml.

    Returns a dict of default values (e.g. window_size_m, output_file_format).
    The result is cached after the first read so the file is only opened once.
    """
    global _defaults_cache
    if _defaults_cache is not None:
        return _defaults_cache

    _defaults_cache = _read_builtin_yaml("defaults.yml")
    return _defaults_cache


class CatalogError(ValueError):
    """Invalid or incomplete catalog.yaml."""


def _require_keys(d: Dict[str, Any], required: set, ctx: str) -> None:
    missing = required - set(d.keys())
    if missing:
        raise CatalogError(f"{ctx}: missing required key(s): {sorted(missing)}")


def _inspect_raster(name: str, spec: Dict[str, Any]) -> None:
    """Read CRS, resolution, type, and nodata from a local raster file
    and fill in any missing spec fields automatically."""
    if spec.get("data_source") != "local":
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
                logger.debug(
                    "datasets.%s: auto-detected resolution_m=%s", name, spec["resolution_m"]
                )

            if "type" not in spec:
                spec["type"] = "raster"

            if "nodata" not in spec and src.nodata is not None:
                spec["nodata"] = src.nodata

            if "band_count" not in spec:
                # Store the number of bands as informational metadata only.
                # Which bands to actually read is decided by the adapter at runtime.
                spec["band_count"] = src.count

    except Exception as e:
        logger.warning("datasets.%s: could not read raster metadata from %s: %s", name, p, e)


def _validate_catalog(data: Dict[str, Any], source_label: str) -> Dict[str, Any]:
    """Validate a parsed catalog dict and run auto-inspection on local rasters.

    Raises CatalogError if the structure is invalid.
    Returns the (possibly mutated) data dict.
    """
    if not isinstance(data, dict):
        raise CatalogError(f"Top-level YAML must be a mapping (dict), got {type(data)}")

    if "datasets" not in data or not isinstance(data["datasets"], dict) or not data["datasets"]:
        raise CatalogError("Top-level key 'datasets' must be a non-empty mapping")

    for name, spec in data["datasets"].items():
        if not isinstance(spec, dict):
            raise CatalogError(f"datasets.{name}: must be a mapping")
        _require_keys(spec, REQUIRED_DATASET_KEYS, f"datasets.{name}")

        if not isinstance(spec["data_source"], str) or not spec["data_source"]:
            raise CatalogError(f"datasets.{name}.data_source must be a non-empty string")
        if not isinstance(spec["path"], str) or not spec["path"]:
            raise CatalogError(f"datasets.{name}.path must be a non-empty string")

        # Earth Engine entries must declare `data_type` up front. Local rasters
        # are allowed to omit it because their type can be inferred from the
        # raster file when needed.
        if spec["data_source"] == "earth_engine":
            _require_keys(spec, EE_REQUIRED_DATASET_KEYS, f"datasets.{name}")

        _inspect_raster(name, spec)

    return data


def load_catalog(path: str | Path) -> Dict[str, Any]:
    """Load and validate a dataset catalog YAML from a file path.

    Required structure:
      datasets:
        <dataset_name>:
          data_source: <str>
          path: <str>
          # optional: type, crs, resolution_m, default_reducer, band, ...
          # For local sources, crs and resolution_m are auto-detected from the file.
    """
    p = Path(path)
    if not p.exists():
        raise CatalogError(f"Catalog file not found: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise CatalogError(f"Catalog YAML parse error in {p}: {e}") from e

    return _validate_catalog(data, str(p))


# Sentinel object used as a default value in extract() to mean "load the
# bundled built-in EE catalog". Using an object instead of a string avoids
# any accidental collision with a user-supplied file path.
BUILTIN_EE_CATALOG = object()


def _load_catalog_any(src: Any) -> Dict[str, Any]:
    """
    Internal helper: accept a path, a dict-like catalog, or the BUILTIN_EE_CATALOG
    sentinel and return a normalized {'datasets': {...}} structure.
    """
    if src is None:
        return {"datasets": {}}

    # Load the built-in GEE catalog bundled with the package.
    if src is BUILTIN_EE_CATALOG:
        return _validate_catalog(_read_builtin_yaml("ee_catalog.yml"), "builtin:ee_catalog.yml")

    # If it's already a mapping, assume it's a parsed catalog dict.
    if isinstance(src, Mapping):
        d = dict(src)
        if "datasets" not in d:
            d["datasets"] = {}
        for name, spec in d["datasets"].items():
            if not isinstance(spec, dict):
                raise CatalogError(f"datasets.{name}: must be a mapping")
            if "data_source" not in spec or not spec.get("data_source"):
                raise CatalogError(
                    f"datasets.{name}: missing required key 'data_source'.\n"
                    f"Valid values are: earth_engine, local."
                )
            if "path" not in spec or not spec.get("path"):
                raise CatalogError(
                    f"datasets.{name}: missing required key 'path'.\n"
                    f"For GEE assets, this is the asset ID (e.g. 'COPERNICUS/S2_SR_HARMONIZED').\n"
                    f"For local rasters, this is the file path (e.g. 'data/dem.tif')."
                )
            # Earth Engine entries must declare `data_type` so we know which
            # reducer set (continuous / categorical) to apply.
            if spec.get("data_source") == "earth_engine" and not spec.get("data_type"):
                raise CatalogError(
                    f"datasets.{name}: missing required key 'data_type' for earth_engine dataset.\n"
                    f"Valid values are: continuous, categorical, mixed."
                )
        return d

    # Otherwise, treat it as a path and reuse the validated loader.
    return load_catalog(src)


def load_catalogs(*sources: Any) -> Dict[str, Any]:
    """
    Merge one or more catalogs (paths, dicts, or lists of paths/dicts) into a
    single catalog. Later sources override earlier ones on a per-dataset basis.
    Always returns: {'datasets': {...}}.
    """
    merged: Dict[str, Any] = {"datasets": {}}

    flat = []
    for src in sources:
        if isinstance(src, (list, tuple)):
            flat.extend(src)
        else:
            flat.append(src)

    for src in flat:
        cat = _load_catalog_any(src)
        for name, spec in cat.get("datasets", {}).items():
            merged["datasets"][name] = spec

    # Apply user-registered datasets as the final layer so they override
    # both built-in and caller-supplied catalogs.
    if _user_catalog_datasets:
        merged["datasets"].update(_user_catalog_datasets)

    return merged


def update_catalog(source: str | Path | dict) -> None:
    """Register additional datasets into the session-wide user catalog.

    Datasets added here are automatically available in every subsequent
    extract() call without needing to pass extra_catalog=. Multiple calls
    are cumulative — each call adds to (or updates) the existing user catalog.

    Args:
        source: A path to a catalog YAML file, or a dict in the format:
                  {"datasets": {"my_dataset": {"data_source": ..., "path": ...}}}

    Raises:
        CatalogError: if the source is invalid or missing required keys.

    Example:
        update_catalog("my_catalog.yml")
        update_catalog({"datasets": {"ndvi_local": {"data_source": "local", "path": "data/ndvi.tif"}}})
    """
    global _user_catalog_datasets
    cat = _load_catalog_any(source)
    _user_catalog_datasets.update(cat.get("datasets", {}))


def reset_catalog() -> None:
    """Clear all datasets previously registered with update_catalog().

    After calling this, extract() will only see the built-in catalogs
    (plus any catalog= or extra_catalog= arguments passed directly).
    """
    global _user_catalog_datasets
    _user_catalog_datasets = {}
