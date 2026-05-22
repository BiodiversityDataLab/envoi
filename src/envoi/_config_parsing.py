# src/envoi/_config_parsing.py
"""Validation and normalization of the run config that extract() receives.

Lifted out of ``extract.py`` so the public entry point stays focused on
orchestration. Everything here is pure-Python config-validation logic —
no GEE calls, no file I/O beyond reading a YAML config file.

Public surface re-imported by ``extract.py``:

* :class:`RunSettings` — immutable container of one parsed output config.
* :func:`_parse_run_config` — turn one raw config dict into RunSettings,
  raising ValueError for any malformed setting.
* :func:`_resolve_stats_for_dataset` — pick the right reducer list for a
  dataset based on its catalog data_type.
* :func:`_as_config_list` — load YAML (if given a path) and normalize the
  result into a list of output config dicts.

Most other helpers (``_normalize_dataset_entry``, ``_validate_*``,
``_parse_statistics``, ``_load_yaml``) are private to this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml

from .adapters.earth_engine import KNOWN_DERIVED_BANDS

# Keys allowed inside the full-form dict value
# (e.g. {"sen2": {"bands": [...]}}). Adding more per-call band_overrides in the
# future means extending this set and validating each new key inside the
# entry-normalization helper below.
_ALLOWED_OVERRIDE_KEYS = frozenset({"bands"})

# Recognised data_type values for the statistics dispatch. A dataset whose
# data_type doesn't match one of these falls back to "continuous" — the
# common case for ecological rasters that don't bother declaring a type.
_VALID_STAT_TYPES = frozenset({"continuous", "categorical"})

# Every reducer name extract() will accept. The set is closed because each
# name maps to a specific server-side reducer (in the GEE adapter) or a
# Python-side reducer (in the local adapter); an unknown name would silently
# produce no output, so we surface it early as a validation error instead.
_ALL_KNOWN_REDUCERS = frozenset(
    {
        "mean",
        "median",
        "min",
        "max",
        "sum",
        "std",
        "var",
        "count",
        "mode",
        "point",
        "class_count",
        "class_fraction",
        "q05",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "q95",
    }
)


@dataclass
class RunSettings:
    """Validated and parsed settings for a single output run."""

    batch_id: str
    # Each entry is a (dataset_name, band_overrides) pair. `band_overrides` is an empty
    # dict for plain-string entries in the user's `datasets` list, and contains
    # `bands` and/or `derived_bands` keys when the user supplied a per-call
    # override (e.g. {"sen2": ["B4", "B8"]} or {"sen2": {"bands": ["B4"]}}).
    datasets: List[Tuple[str, Dict[str, Any]]]
    output_type: str  # "tabular" or "raster"
    output_file_format: str  # "csv", "parquet", or "dataframe"
    window_sizes: List[int]  # one or more square-sampling-window sizes in metres
    min_coverage: float  # 0–100 — threshold for low-coverage QC flag
    # Normalized stats dict: {"continuous": [...], "categorical": [...]}.
    # A flat list from the user is normalized to identical lists on both keys.
    # Downstream code calls _resolve_stats_for_dataset() to pick the right list
    # per dataset rather than reading this dict directly.
    stats: Dict[str, List[str]]
    # Original user-supplied form (flat list or typed dict), stored verbatim
    # so the metadata sidecar round-trips it without normalizing it away.
    user_stats: list | Dict[str, List[str]]
    resample_m: float | None  # target pixel size in metres (raster mode only)
    user_window_size: int | List[int]  # original input form, preserved for metadata


def _normalize_dataset_entry(
    entry: Any,
    batch_id: str,
    catalog_datasets: dict,
) -> Tuple[str, Dict[str, Any]]:
    """Normalize one item from the user's `datasets` list into (name, band_overrides).

    Three accepted shapes:
      * A plain string  -> ("name", {})
      * Shorthand dict  -> {"name": [bands...]}
                          where the list is a unified band list (source +
                          derived bands mixed). Names recognised as derived
                          (KNOWN_DERIVED_BANDS) are split into the
                          `derived_bands` override; the rest go to `bands`.
      * Full-form dict  -> {"name": {"bands": [...]}}
                          where the inner dict accepts the keys listed in
                          `_ALLOWED_OVERRIDE_KEYS`. Reserved for future
                          per-call band_overrides; today only `bands` is allowed.

    Raises ValueError for any malformed entry and KeyError for an unknown
    dataset name (matching the existing error message used elsewhere in the
    pipeline so the user sees a consistent failure mode).
    """
    # ---- shape 1: plain string ----
    if isinstance(entry, str):
        if not entry:
            raise ValueError(f"Output '{batch_id}': dataset name cannot be empty.")
        if entry not in catalog_datasets:
            raise KeyError(f"Output '{batch_id}': dataset(s) ['{entry}'] not found in catalog.")
        # Even with no per-call override, the catalog default `derived_bands`
        # still becomes the effective list. Validate it against the dataset's
        # applicability whitelist so a misconfigured catalog entry surfaces
        # up front instead of silently producing nonsense bands.
        _validate_effective_derived_bands(
            override_derived=[],
            dataset_name=entry,
            dataset_config=catalog_datasets[entry],
            batch_id=batch_id,
        )
        return entry, {}

    # ---- shapes 2 & 3: single-key dict ----
    if not isinstance(entry, dict):
        raise ValueError(
            f"Output '{batch_id}': each dataset entry must be a string or a single-key "
            f"dict, got {type(entry).__name__}: {entry!r}."
        )
    if len(entry) != 1:
        raise ValueError(
            f"Output '{batch_id}': each dataset dict must have exactly one key "
            f"(the dataset name), got {len(entry)} keys: {sorted(entry.keys())}."
        )

    # Pull out the single (name, value) pair. The value is either a list
    # (shorthand) or a dict (full form); anything else is rejected below.
    name, value = next(iter(entry.items()))

    if not isinstance(name, str) or not name:
        raise ValueError(
            f"Output '{batch_id}': dataset key must be a non-empty string, got {name!r}."
        )
    if name not in catalog_datasets:
        raise KeyError(f"Output '{batch_id}': dataset(s) ['{name}'] not found in catalog.")

    # Normalize the value into a unified bands list. Both the shorthand and
    # the full form ultimately produce the same list, which we then split
    # into source bands and derived bands below.
    if isinstance(value, list):
        # Shorthand: the list IS the unified band list.
        unified_bands = value
    elif isinstance(value, dict):
        # Full form: validate keys, then read `bands` out of the inner dict.
        unknown_keys = set(value.keys()) - _ALLOWED_OVERRIDE_KEYS
        if unknown_keys:
            raise ValueError(
                f"Output '{batch_id}': unknown override key(s) {sorted(unknown_keys)} for "
                f"dataset '{name}'. Allowed: {sorted(_ALLOWED_OVERRIDE_KEYS)}."
            )
        unified_bands = value.get("bands")
        if unified_bands is None:
            # Empty full-form dict — treat as no band_overrides at all.
            return name, {}
        if not isinstance(unified_bands, list):
            raise ValueError(
                f"Output '{batch_id}': 'bands' for dataset '{name}' must be a list, "
                f"got {type(unified_bands).__name__}."
            )
    else:
        raise ValueError(
            f"Output '{batch_id}': override for dataset '{name}' must be a list (shorthand) "
            f"or a dict (full form), got {type(value).__name__}: {value!r}."
        )

    if not unified_bands:
        raise ValueError(
            f"Output '{batch_id}': band list for dataset '{name}' must contain at least one band."
        )

    # Split the unified list into source bands and derived bands. Order is
    # preserved within each side so the resulting output band order matches
    # what the user wrote.
    derived_bands = [b for b in unified_bands if b in KNOWN_DERIVED_BANDS]
    source_bands = [b for b in unified_bands if b not in KNOWN_DERIVED_BANDS]

    # Local rasters cannot have derived bands (no slope/aspect computation
    # path exists in LocalRasterAdapter). Surface this clearly so the user
    # knows the catalog is the right place for that.
    dataset_config = catalog_datasets[name]
    data_source = dataset_config.get("data_source")
    if derived_bands and data_source != "earth_engine":
        raise ValueError(
            f"Output '{batch_id}': dataset '{name}' is a {data_source!r} raster — "
            f"derived bands {sorted(set(derived_bands))} are only supported for "
            f"earth_engine datasets."
        )

    # Enforce the catalog's per-dataset applicability whitelist. A derived
    # band (e.g. slope) only makes physical sense for some datasets (e.g.
    # DEMs), so the catalog declares `supported_derived_bands` to mark which
    # derived bands are valid here. The check runs against the EFFECTIVE
    # derived bands — the call-site override if present, otherwise the
    # catalog's `derived_bands` default — so a misconfigured catalog entry
    # surfaces even when the user supplies no override.
    _validate_effective_derived_bands(
        override_derived=derived_bands,
        dataset_name=name,
        dataset_config=dataset_config,
        batch_id=batch_id,
    )

    band_overrides: Dict[str, Any] = {}
    if source_bands:
        band_overrides["bands"] = source_bands
    if derived_bands:
        band_overrides["derived_bands"] = derived_bands
    return name, band_overrides


def _validate_effective_derived_bands(
    *,
    override_derived: list,
    dataset_name: str,
    dataset_config: dict,
    batch_id: str,
) -> None:
    """Raise ValueError if the effective derived bands violate the whitelist.

    The effective list is the call-site override when provided; otherwise it
    falls back to the catalog's own `derived_bands` default. The list is then
    validated against the dataset's `supported_derived_bands` whitelist — a
    missing or empty whitelist means the dataset declines derived bands
    entirely.
    """
    # When the user passed any derived bands at the call site, those win and
    # we ignore the catalog default. Otherwise the catalog default IS the
    # effective list and must also satisfy the whitelist.
    if override_derived:
        effective = list(override_derived)
    else:
        catalog_default = dataset_config.get("derived_bands")
        if isinstance(catalog_default, str):
            effective = [catalog_default]
        elif catalog_default:
            effective = list(catalog_default)
        else:
            effective = []

    if not effective:
        return

    supported = dataset_config.get("supported_derived_bands") or []
    unsupported = [b for b in effective if b not in supported]
    if not unsupported:
        return

    if supported:
        supported_msg = f"Supported for this dataset: {sorted(supported)}."
    else:
        supported_msg = (
            "This dataset does not declare `supported_derived_bands` in the catalog, "
            "so no derived bands are allowed for it."
        )
    raise ValueError(
        f"Output '{batch_id}': dataset '{dataset_name}' does not support derived "
        f"band(s) {sorted(set(unsupported))}. {supported_msg}"
    )


def _parse_run_config(
    run_config: dict,
    defaults: dict,
    index: int,
    catalog_datasets: dict,
) -> RunSettings:
    """Parse and validate a single output run config dict into a RunSettings instance.

    Raises ValueError for any invalid or missing setting so callers don't need
    to do any further validation on the returned object.
    """
    # Use a numbered fallback batch_id when the user didn't provide one.
    batch_id = run_config.get("batch_id", f"output{index + 1}")

    # datasets is required — must be a non-empty list. Each entry is normalized
    # into a (name, band_overrides) tuple by the helper, which also handles all
    # validation (catalog existence, malformed dicts, derived-on-local, ...).
    raw_datasets = run_config.get("datasets", [])
    if not raw_datasets:
        raise ValueError(f"Output '{batch_id}': missing required 'datasets' list")
    datasets = [_normalize_dataset_entry(e, batch_id, catalog_datasets) for e in raw_datasets]

    # settings is required — must be a non-empty dict.
    settings = run_config.get("settings", {}) or {}
    if not settings:
        raise ValueError(f"Output '{batch_id}': missing required 'settings' dict")

    # output_type controls the entire processing path (stats vs tile export).
    output_type = settings.get("output_type")
    if output_type not in ("tabular", "raster"):
        raise ValueError(f"Unknown or missing output_type: {output_type}")

    # resample_m is only meaningful for raster output.
    resample_m = settings.get("resample_m")
    if output_type == "raster" and resample_m is not None and resample_m <= 0:
        raise ValueError(f"Invalid resample_m: {resample_m}. Must be a positive number.")
    if output_type == "tabular" and resample_m is not None:
        raise ValueError("resample_m is not applicable when output_type is 'tabular'.")

    # output_file_format applies to tabular output only.
    output_file_format = settings.get("output_file_format", defaults["output_file_format"])
    if output_file_format not in ("csv", "parquet", "dataframe"):
        raise ValueError(f"Unknown output_file_format: {output_file_format}")

    # min_coverage_pct is a percentage — must be in [0, 100].
    min_coverage = settings.get("min_coverage_pct", defaults["min_coverage_pct"])
    if min_coverage < 0 or min_coverage > 100:
        raise ValueError(f"Invalid min_coverage_pct: {min_coverage}. Must be between 0 and 100.")

    # statistics — required for tabular, forbidden for raster.
    # Accepts either a flat list (applied to all datasets) or a typed dict
    # {"continuous": [...], "categorical": [...]} for mixed-type runs.
    raw_stats = settings.get("statistics")
    if output_type == "raster" and raw_stats:
        raise ValueError("Statistics cannot be computed when output_type is 'raster'.")
    if output_type == "tabular":
        stats, user_stats = _parse_statistics(raw_stats, batch_id)
    else:
        stats, user_stats = {}, None

    # window_size_m can be either a single positive integer or a list of them.
    # When the user supplies a list, statistics (or tiles) are produced for
    # each window size and the column / filename suffix disambiguates them.
    user_window_size = settings.get("window_size_m", defaults["window_size_m"])
    if isinstance(user_window_size, (list, tuple)):
        window_sizes = list(user_window_size)
        if not window_sizes:
            raise ValueError(f"Output '{batch_id}': window_size_m list must not be empty.")
    else:
        window_sizes = [user_window_size]
    for window_size in window_sizes:
        # Reject non-integers explicitly. window_size feeds f-string column
        # names like "{dataset}_mean_{window_size_m}m", and a float would
        # silently yield columns like "dem_mean_200.0m" — breaking schema
        # expectations downstream. ``bool`` is a subclass of ``int`` in
        # Python, so we filter it out first to avoid accepting True/False.
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(
                f"Output '{batch_id}': invalid window_size_m: {window_size!r}. "
                f"Must be a positive integer (or a list of positive integers)."
            )

    return RunSettings(
        batch_id=batch_id,
        datasets=datasets,
        output_type=output_type,
        output_file_format=output_file_format,
        window_sizes=window_sizes,
        min_coverage=min_coverage,
        stats=stats,
        user_stats=user_stats,
        resample_m=resample_m,
        user_window_size=user_window_size,
    )


def _parse_statistics(
    raw: Any,
    batch_id: str,
) -> tuple[Dict[str, List[str]], Any]:
    """Parse and validate the user's `statistics` setting.

    Accepts two forms:
      * A flat list of reducer names — applied to all datasets regardless of
        type. Normalized internally to {"continuous": list, "categorical": list}.
      * A typed dict {"continuous": [...], "categorical": [...]} — each key is
        optional, but at least one must be present and non-empty.

    Returns (normalized_dict, user_stats) where user_stats is the original
    user-supplied value, preserved verbatim for the metadata sidecar.
    Raises ValueError for any invalid input.
    """
    if not raw:
        raise ValueError(
            f"Output '{batch_id}': 'statistics' must be a non-empty list or "
            f"a dict with 'continuous' and/or 'categorical' keys."
        )

    # ---- flat list (backward-compat) ----
    if isinstance(raw, list):
        if len(raw) == 0:
            raise ValueError(f"Output '{batch_id}': 'statistics' list must not be empty.")
        _validate_reducer_names(raw, batch_id, context="statistics")
        normalized = {"continuous": raw, "categorical": raw}
        return normalized, raw

    # ---- typed dict ----
    if isinstance(raw, dict):
        unknown_keys = set(raw.keys()) - _VALID_STAT_TYPES
        if unknown_keys:
            raise ValueError(
                f"Output '{batch_id}': unknown 'statistics' key(s) {sorted(unknown_keys)}. "
                f"Allowed: {sorted(_VALID_STAT_TYPES)}."
            )
        if not raw:
            raise ValueError(
                f"Output '{batch_id}': 'statistics' dict must contain at least one of "
                f"{sorted(_VALID_STAT_TYPES)}."
            )
        normalized: Dict[str, List[str]] = {}
        for key, reducers in raw.items():
            if not isinstance(reducers, list) or not reducers:
                raise ValueError(
                    f"Output '{batch_id}': 'statistics.{key}' must be a non-empty list."
                )
            _validate_reducer_names(reducers, batch_id, context=f"statistics.{key}")
            normalized[key] = reducers
        return normalized, raw

    raise ValueError(
        f"Output '{batch_id}': 'statistics' must be a list or dict, " f"got {type(raw).__name__}."
    )


def _validate_reducer_names(reducers: list, batch_id: str, context: str) -> None:
    """Raise ValueError if any reducer name is not in the known set."""
    unknown = [r for r in reducers if r not in _ALL_KNOWN_REDUCERS]
    if unknown:
        raise ValueError(
            f"Output '{batch_id}': unknown reducer(s) {unknown} in '{context}'. "
            f"Valid reducers: {sorted(_ALL_KNOWN_REDUCERS)}."
        )


def _resolve_stats_for_dataset(
    data_type: str | None,
    stats: Dict[str, List[str]],
    dataset_name: str,
    batch_id: str,
) -> List[str]:
    """Return the reducer list to use for one dataset based on its data_type.

    Falls back to 'continuous' when data_type is None or unrecognised, since
    most ecological rasters are continuous and users often omit data_type for
    local datasets. Raises ValueError when the resolved list would be empty
    so the user gets a clear message rather than a run with no output columns.
    """
    resolved_type = data_type if data_type in _VALID_STAT_TYPES else "continuous"
    reducers = stats.get(resolved_type)
    if not reducers:
        raise ValueError(
            f"Dataset '{dataset_name}' has data_type='{data_type}' but the "
            f"'{resolved_type}' statistics list is missing or empty. "
            f"Add a '{resolved_type}' key to the 'statistics' dict in the run config "
            f"for output '{batch_id}'."
        )
    return reducers


def _load_yaml(path_or_dict):
    """Load a YAML file if given a path, or return the dict/list if already loaded."""
    if isinstance(path_or_dict, (dict, list)):
        return path_or_dict
    with open(path_or_dict) as f:
        return yaml.safe_load(f)


def _as_config_list(config) -> list[dict]:
    """Load and normalize config into a list of output config dicts."""
    raw = _load_yaml(config)
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    raise ValueError("config must be a dict, list, or path to a YAML file.")
