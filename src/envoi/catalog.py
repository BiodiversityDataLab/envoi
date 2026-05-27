# src/envoi/catalog.py
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

# Cached set of built-in dataset names, used by update_catalog() to detect
# when a user-registered entry shadows a built-in. Lazy-loaded on first use.
_builtin_dataset_names_cache: frozenset[str] | None = None

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


def _get_builtin_dataset_names() -> frozenset[str]:
    """Return the set of names defined in the bundled built-in EE catalog.

    Cached on first call so the YAML is only parsed once per process.
    """
    global _builtin_dataset_names_cache
    if _builtin_dataset_names_cache is not None:
        return _builtin_dataset_names_cache

    builtin = _read_builtin_yaml("ee_catalog.yml")
    _builtin_dataset_names_cache = frozenset(builtin.get("datasets", {}).keys())
    return _builtin_dataset_names_cache


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
                    f"Valid values are: continuous, categorical."
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
    new_datasets = cat.get("datasets", {})

    # Log when a registered name shadows a built-in catalog entry. Done at
    # info level (not warn) because shadowing is documented and often
    # intentional — users sometimes register their own version of a
    # built-in dataset (e.g. a local DEM that replaces the GEE one). But
    # it's also a common typo source, so surfacing it once per call helps
    # the user spot accidental collisions.
    builtin_names = _get_builtin_dataset_names()
    for name in new_datasets:
        if name in builtin_names:
            logger.info(
                "update_catalog: '%s' shadows the built-in catalog entry of the same name",
                name,
            )

    _user_catalog_datasets.update(new_datasets)


def reset_catalog() -> None:
    """Clear all datasets previously registered with update_catalog().

    After calling this, extract() will only see the built-in catalogs
    (plus any catalog= or extra_catalog= arguments passed directly).
    """
    global _user_catalog_datasets
    _user_catalog_datasets = {}


# Verbosity levels accepted by list_datasets(). Kept as a module-level
# constant so callers (and tests) can introspect what's valid without
# duplicating the literal set.
LIST_DATASETS_VERBOSITY = ("names", "info", "full")


def list_datasets(verbosity: str = "names") -> list:
    """List every dataset currently registered with envoi.

    Combines the built-in EE catalog with any datasets the user has added
    via ``update_catalog()`` — i.e. exactly what ``extract()`` would see.
    The result is both printed to stdout (for interactive / notebook use)
    and returned (so callers can keep processing it programmatically).

    Args:
        verbosity: How much detail to include for each dataset.
            - ``"names"`` (default): one dataset key per line.
              Returns ``list[str]`` of sorted dataset names.
            - ``"info"``: name plus the ``dataset_information`` fields
              (description, citation, ee_source_url, source_url) and the
              top-level ``data_source`` / ``data_type``.
              Returns ``list[dict]`` with those fields.
            - ``"full"``: the complete catalog entry — every key present
              in the YAML (bands, dataset_spec, paths, etc.).
              Returns ``list[dict]`` where each dict starts with ``name``
              and then mirrors the catalog entry exactly.

    Raises:
        ValueError: if ``verbosity`` is not one of the supported levels.

    Examples:
        >>> list_datasets()              # just the names
        >>> list_datasets("info")        # name + descriptions / citations
        >>> list_datasets("full")        # everything
    """
    # Validate the verbosity argument up front so a typo fails loudly
    # instead of silently falling through to the default branch.
    if verbosity not in LIST_DATASETS_VERBOSITY:
        raise ValueError(
            f"verbosity must be one of {list(LIST_DATASETS_VERBOSITY)}, got {verbosity!r}"
        )

    # Load the merged catalog (built-in EE + any user-registered datasets).
    # This is the same view extract() uses, so what we print matches what
    # the user can actually pass to extract().
    catalog = load_catalogs(BUILTIN_EE_CATALOG)
    datasets = catalog.get("datasets", {})
    sorted_names = sorted(datasets.keys())

    # ----- "names" -------------------------------------------------------
    # Just dump the keys, one per line. Cheap, scannable, and easy to grep.
    if verbosity == "names":
        for name in sorted_names:
            print(name)
        return sorted_names

    # ----- "info" --------------------------------------------------------
    # Pull just the human-readable metadata that lives under
    # `dataset_information`, plus the two top-level fields users most often
    # want to see at a glance (data_source, data_type).
    if verbosity == "info":
        info_records: list[dict] = []
        for name in sorted_names:
            entry = datasets[name]
            dataset_information = entry.get("dataset_information", {}) or {}
            info_records.append(
                {
                    "name": name,
                    "data_source": entry.get("data_source"),
                    "data_type": entry.get("data_type"),
                    "description": dataset_information.get("description"),
                    "citation": dataset_information.get("citation"),
                    "ee_source_url": dataset_information.get("ee_source_url"),
                    "source_url": dataset_information.get("source_url"),
                }
            )

        # Pretty print each record as a small block. We intentionally skip
        # missing fields so the output stays compact for sparse entries.
        for record in info_records:
            header_type = record["data_type"] or "unspecified type"
            print(f"{record['name']} ({record['data_source']}, {header_type})")
            if record["description"]:
                print(f"  description: {record['description']}")
            if record["ee_source_url"]:
                print(f"  ee_source_url: {record['ee_source_url']}")
            if record["source_url"]:
                print(f"  source_url: {record['source_url']}")
            if record["citation"]:
                print(f"  citation: {record['citation']}")
            # Trailing blank line separates entries so the block is readable.
            print()
        return info_records

    # ----- "full" --------------------------------------------------------
    # Return the entire catalog entry. We use yaml.dump for printing because
    # entries can be deeply nested (bands, dataset_spec, dataset_information)
    # and Python's default repr is unreadable for nested dicts.
    full_records: list[dict] = []
    for name in sorted_names:
        entry = datasets[name]
        full_records.append({"name": name, **entry})

    for record in full_records:
        # Print as a single YAML block per dataset so nested fields render
        # cleanly instead of as one-line Python repr.
        yaml_text = yaml.dump(
            {record["name"]: {k: v for k, v in record.items() if k != "name"}},
            sort_keys=False,
            default_flow_style=False,
        )
        print(yaml_text)
    return full_records
