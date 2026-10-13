# src/envoi/catalog.py
from __future__ import annotations

import copy
import importlib.resources as importlib_resources
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ._filenames import describe_unsafe_path_component, is_safe_path_component

# The top-level keys that every catalog entry can have, whatever its
# data_source. Each adapter accepts more keys through its `catalog_keys`
# declaration. Any other key is rejected: an ignored key, for example a typing
# error in `data_type`, would give a silently different run.
_GENERIC_CATALOG_KEYS = frozenset(
    {
        "data_source",
        "path",
        "data_type",
        "display_name",
        "category",
        "bands",
        "derived_bands",
        "supported_derived_bands",
        "dataset_information",
    }
)

# The values of `data_type`. It selects the continuous or the categorical
# reducer list of the `statistics` setting. A null or absent data_type uses
# the continuous list. Another value is rejected, because it would also use
# the continuous list, without a message. The same values are the keys of a
# typed `statistics` dict in a run config (`_config_parsing.py`).
_DATA_TYPES = frozenset({"continuous", "categorical"})

# Module-level cache so each bundled YAML is only read once per process.
_defaults_cache: dict[str, Any] | None = None

# Cached set of built-in dataset names, used by update_catalog() to detect
# when a user-registered entry shadows a built-in. Lazy-loaded on first use.
_builtin_dataset_names_cache: frozenset[str] | None = None

# Datasets registered by the user via update_catalog(). Persists for the
# duration of the Python session. Applied as the final merge layer in
# load_catalogs() so user entries always override built-in ones.
_user_catalog_datasets: dict[str, Any] = {}

logger = logging.getLogger(__name__)


def _read_builtin_yaml(filename: str) -> dict[str, Any]:
    """Read a YAML file bundled inside the envoi package under configs/.

    Uses importlib.resources so the file is found correctly whether the
    package is installed via pip or run directly from source.
    """
    ref = importlib_resources.files("envoi").joinpath("configs").joinpath(filename)
    text = ref.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def load_defaults() -> dict[str, Any]:
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


def _get_adapter_class(name: str, entry: Mapping) -> type:
    """Return the adapter class that is registered for the ``data_source`` of ``entry``.

    Checking when a catalog is loaded or registered stops a typo before
    ``extract()`` builds an adapter for any dataset, or uses Earth Engine quota.

    Raises:
        CatalogError: If ``entry`` has no ``data_source``, or no adapter is
            registered for it. The message lists the registered data sources.
    """
    # Imported here, not at module level: envoi.adapters imports this module
    # (the GEE adapter reads load_defaults()), so a module-level import would
    # be an import cycle. Validation runs only after both modules are loaded.
    from .adapters import _REG

    if "data_source" not in entry:
        raise CatalogError(
            f"datasets.{name}: missing required key 'data_source'. "
            f"Valid values are: {', '.join(sorted(_REG))}."
        )
    data_source = entry["data_source"]
    if not isinstance(data_source, str) or data_source not in _REG:
        raise CatalogError(
            f"datasets.{name}: unknown data_source {data_source!r}. "
            f"Valid values are: {', '.join(sorted(_REG))}."
        )
    return _REG[data_source]


def _derived_band_names(name: str, key: str, value: Any) -> list:
    """Return the band names of a ``derived_bands`` or ``supported_derived_bands`` value.

    The value is one band name, a list of band names, or null (no bands).

    Raises:
        CatalogError: If the value has another shape.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise CatalogError(
        f"datasets.{name}: {key} must be a band name or a list of band names, got {value!r}."
    )


def _validate_catalog_entry(name: str, entry: Any, *, inspect_entry: bool) -> dict[str, Any]:
    """Check one catalog entry and return the copy of it that envoi stores.

    The YAML path (``_validate_catalog``) and the dict path
    (``_load_catalog_any``) both call this. It checks, in this order:

    1. The generic rules: the entry is a mapping; ``data_source`` names a
       registered adapter; every key is in ``_GENERIC_CATALOG_KEYS`` or in
       the adapter's ``catalog_keys``; ``path`` is a non-empty ``str`` or an
       ``os.PathLike``; ``data_type`` is absent, null, or in ``_DATA_TYPES``;
       and ``derived_bands`` and ``supported_derived_bands`` name only the
       adapter's ``computable_derived_bands``.
    2. The adapter's own rules (``validate_catalog_entry``).
    3. With ``inspect_entry`` (catalog files only), the adapter's
       ``inspect_catalog_entry``, which can add values that it reads from the
       data source, for example the CRS of a local raster.

    Args:
        name: The name of the dataset (its key under ``datasets``).
        entry: The catalog entry, as the user wrote it. Not changed.
        inspect_entry: Whether to run the adapter's ``inspect_catalog_entry``.

    Returns:
        A deep copy of ``entry``, with ``path`` as a ``str`` and the values
        of the inspection. Later changes to ``entry`` do not reach it.

    Raises:
        CatalogError: If a rule is broken. The message names the dataset, the
            key, and the allowed values or keys. A ``ValueError`` of an
            adapter hook becomes a ``CatalogError`` with the same message
            after ``datasets.<name>: ``.
    """
    if not isinstance(entry, Mapping):
        raise CatalogError(f"datasets.{name}: must be a mapping")

    # ---- data_source: the rules below use the declarations of its adapter ----
    adapter_class = _get_adapter_class(name, entry)
    data_source = entry["data_source"]

    # ---- keys ----
    allowed_keys = _GENERIC_CATALOG_KEYS | adapter_class.catalog_keys
    unknown_keys = sorted((key for key in entry if key not in allowed_keys), key=str)
    if unknown_keys:
        raise CatalogError(
            f"datasets.{name}: unknown key(s) {unknown_keys} for data_source "
            f"{data_source!r}. Allowed keys: {sorted(allowed_keys)}. Put free text, for "
            f"example notes about the dataset, under 'dataset_information'."
        )

    # ---- path ----
    if "path" not in entry or entry["path"] is None:
        raise CatalogError(
            f"datasets.{name}: missing required key 'path'. Set it to the location of the "
            f"data, for example an Earth Engine asset ID ('COPERNICUS/S2_SR_HARMONIZED') or "
            f"the path of a raster file ('data/dem.tif')."
        )
    path = entry["path"]
    # A pathlib.Path is accepted, and stored as text, so that the metadata
    # sidecar (JSON) can record it.
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str) or not path:
        raise CatalogError(
            f"datasets.{name}: path must be a non-empty text or a pathlib.Path, "
            f"got {entry['path']!r}."
        )

    # ---- data_type: null counts as absent ----
    data_type = entry.get("data_type")
    if data_type is not None and (not isinstance(data_type, str) or data_type not in _DATA_TYPES):
        raise CatalogError(
            f"datasets.{name}: unknown data_type {data_type!r}. "
            f"Valid values are: {', '.join(sorted(_DATA_TYPES))}."
        )

    # ---- derived bands: only the ones that the adapter can compute ----
    computable_derived_bands = adapter_class.computable_derived_bands
    for key in ("derived_bands", "supported_derived_bands"):
        band_names = _derived_band_names(name, key, entry.get(key))
        uncomputable_bands = [
            band
            for band in band_names
            if not isinstance(band, str) or band not in computable_derived_bands
        ]
        if uncomputable_bands:
            computable_label = (
                str(sorted(computable_derived_bands)) if computable_derived_bands else "none"
            )
            raise CatalogError(
                f"datasets.{name}: {key} has {uncomputable_bands}, which the adapter of "
                f"data_source {data_source!r} cannot compute. Derived bands that this "
                f"adapter computes: {computable_label}."
            )

    # The stored copy. A deep copy, so that a later change to the user's dict
    # (or to its nested lists and dicts) does not change the stored entry.
    stored_entry = copy.deepcopy(dict(entry))
    stored_entry["path"] = path

    # ---- the adapter's own rules, then the inspection (catalog files only) ----
    try:
        adapter_class.validate_catalog_entry(name, stored_entry)
        if inspect_entry:
            stored_entry = adapter_class.inspect_catalog_entry(name, stored_entry)
    except CatalogError:
        raise
    except ValueError as error:
        raise CatalogError(f"datasets.{name}: {error}") from error

    return stored_entry


def _validate_catalog(data: dict[str, Any], source_label: str) -> dict[str, Any]:
    """Validate a parsed catalog file, and inspect each entry through its adapter.

    Raises CatalogError if the structure or an entry is invalid
    (``_validate_catalog_entry``).
    Returns a copy of ``data`` whose ``datasets`` are the checked and
    inspected copies of the entries.
    """
    if not isinstance(data, dict):
        raise CatalogError(f"Top-level YAML must be a mapping (dict), got {type(data)}")

    if "datasets" not in data or not isinstance(data["datasets"], dict) or not data["datasets"]:
        raise CatalogError("Top-level key 'datasets' must be a non-empty mapping")

    validated_datasets = {
        name: _validate_catalog_entry(name, entry, inspect_entry=True)
        for name, entry in data["datasets"].items()
    }
    return {**data, "datasets": validated_datasets}


def load_catalog(path: str | Path) -> dict[str, Any]:
    """Load and validate a dataset catalog YAML from a file path.

    Required structure:
      datasets:
        <dataset_name>:
          data_source: <str>   # earth_engine or local
          path: <str>          # GEE asset ID, or local raster file path
          # earth_engine entries also need data_type (continuous or categorical).
          # For the optional keys, see the example entry at the top of
          # src/envoi/configs/ee_catalog.yml.

    For each local entry whose file exists, the raster's ``crs``,
    ``resolution_m`` (in metres), ``nodata``, and ``band_count`` are added to
    the entry when the entry does not set them. A raster that cannot be read
    gives a logged warning, not an error. ``resolution_m`` and ``band_count``
    are for information only. An extraction uses the file's own CRS and
    nodata value; the entry's ``crs`` and ``nodata`` only fill a value that
    the file does not declare, and ``extract()`` raises ``ValueError`` when
    they differ from the file's values.

    Raises:
        CatalogError: if the file does not exist, the YAML cannot be parsed, or
            an entry is invalid, for example without a required key, or with an
            unknown ``data_source`` or ``dataset_spec.collection_date_policy``.
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


def _load_catalog_any(src: Any) -> dict[str, Any]:
    """
    Internal helper: accept a path, a dict-like catalog, or the BUILTIN_EE_CATALOG
    sentinel and return a normalized {'datasets': {...}} structure.

    The entries in the result are checked copies (``_validate_catalog_entry``),
    so a change to them does not change ``src``. A dict without a ``datasets``
    mapping raises ``CatalogError``; ``{"datasets": {}}`` is accepted.
    """
    if src is None:
        return {"datasets": {}}

    # Load the built-in GEE catalog bundled with the package.
    if src is BUILTIN_EE_CATALOG:
        return _validate_catalog(_read_builtin_yaml("ee_catalog.yml"), "builtin:ee_catalog.yml")

    # If it's already a mapping, assume it's a parsed catalog dict. Its
    # entries are checked as the entries of a catalog file, but not inspected.
    if isinstance(src, Mapping):
        datasets = src.get("datasets")
        if not isinstance(datasets, Mapping):
            raise CatalogError(
                f"A catalog dict needs the top-level key 'datasets', a mapping of dataset "
                f"names to catalog entries, got {datasets!r}. Example: "
                f"{{'datasets': {{'my_dem': {{'data_source': ..., 'path': 'data/dem.tif'}}}}}}."
            )
        validated_datasets = {
            name: _validate_catalog_entry(name, entry, inspect_entry=False)
            for name, entry in datasets.items()
        }
        return {"datasets": validated_datasets}

    # Otherwise, treat it as a path and reuse the validated loader.
    return load_catalog(src)


def load_catalogs(*sources: Any) -> dict[str, Any]:
    """
    Merge one or more catalogs (paths, dicts, or lists of paths/dicts) into a
    single catalog. Later sources override earlier ones on a per-dataset basis.
    Always returns: {'datasets': {...}}. The entries are copies: a change to
    them does not change a source or the datasets of ``update_catalog()``.
    """
    merged: dict[str, Any] = {"datasets": {}}

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
    # both built-in and caller-supplied catalogs. Deep copies, so that the
    # caller can change the returned entries without an effect on the stored ones.
    if _user_catalog_datasets:
        merged["datasets"].update(copy.deepcopy(_user_catalog_datasets))

    return merged


def update_catalog(source: str | Path | dict) -> None:
    """Register additional datasets into the session-wide user catalog.

    Datasets added here are available in every later extract() and
    list_datasets() call of the same Python session, and override built-in
    entries with the same name. Multiple calls are cumulative — each call adds
    to (or updates) the existing user catalog.

    A catalog file goes through ``load_catalog()``, which also adds the
    metadata of local rasters to their entries. Entries of a dict get the same
    checks, but are stored without reading the raster files. envoi stores
    copies of the entries, so a later change to ``source`` has no effect.
    In both cases, the local adapter reads the CRS, pixel size, nodata,
    and band count from the file when an extraction opens it. The entry's
    ``crs`` and ``nodata`` only fill a CRS or nodata value that the file does
    not declare; ``extract()`` raises ``ValueError`` when they differ from the
    file's values.

    Args:
        source: A path to a catalog YAML file, or a dict in the format:
                  {"datasets": {"my_dataset": {"data_source": ..., "path": ...}}}

    Raises:
        CatalogError: if the source is invalid (also a dict without a
            ``datasets`` mapping), an entry is missing a required key, has an
            unknown key, or has an unknown value of ``data_source``,
            ``data_type``, or ``dataset_spec.collection_date_policy``, or a
            dataset name cannot be used as a folder name.

    Example:
        update_catalog("my_catalog.yml")
        update_catalog({"datasets": {"ndvi_local": {"data_source": "local", "path": "data/ndvi.tif"}}})
    """
    # No `global` declaration needed: the module-level dict is mutated in
    # place (see the .update() below), never rebound. The entries of `cat` are
    # copies (_validate_catalog_entry), so the user's dict is not stored.
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
        # A dataset key becomes a folder name in raster mode
        # (output_dir/<batch_id>/<dataset>/) and a column prefix in tabular
        # mode, so it must be a valid path component on every OS. Catalog keys
        # are hand-authored, so an unsafe one is reported rather than silently
        # rewritten.
        if not is_safe_path_component(name):
            raise CatalogError(
                f"Invalid dataset name {name!r}: {describe_unsafe_path_component(name)}. "
                f"Dataset names are used as folder names and column prefixes, so they may "
                f"contain only letters, digits, '.', '_' and '-'."
            )

        if name in builtin_names:
            logger.info(
                "update_catalog: '%s' shadows the built-in catalog entry of the same name",
                name,
            )

    _user_catalog_datasets.update(new_datasets)


def reset_catalog() -> None:
    """Clear all datasets previously registered with update_catalog().

    After calling this, extract() and list_datasets() see only the built-in
    catalog.
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
    Returns the requested dataset information.

    Args:
        verbosity: How much detail to include for each dataset.
            - ``"names"`` (default): one dataset key per line.
              Returns ``list[str]`` of sorted dataset names.
            - ``"info"``: name plus the ``dataset_information`` fields
              (description, citation, ee_source_url, source_url) and the
              top-level ``display_name`` / ``data_source`` / ``data_type`` /
              ``category``. ``display_name`` falls back to the dataset name
              when the catalog entry does not define one.
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
    # This is the same view extract() uses, so the listing is honest.
    catalog = load_catalogs(BUILTIN_EE_CATALOG)
    datasets = catalog.get("datasets", {})
    sorted_names = sorted(datasets.keys())

    # ----- "names" -------------------------------------------------------
    if verbosity == "names":
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
                    # Human-readable label for menus and documentation. Falls
                    # back to the catalog key so callers can always display
                    # something, even for entries that omit the field.
                    "display_name": entry.get("display_name") or name,
                    "data_source": entry.get("data_source"),
                    "data_type": entry.get("data_type"),
                    "category": entry.get("category"),
                    "description": dataset_information.get("description"),
                    "citation": dataset_information.get("citation"),
                    "ee_source_url": dataset_information.get("ee_source_url"),
                    "source_url": dataset_information.get("source_url"),
                }
            )

        return info_records

    # ----- "full" --------------------------------------------------------
    full_records: list[dict] = []
    for name in sorted_names:
        entry = datasets[name]
        full_records.append({"name": name, **entry})
    return full_records
