# src/envoi/_config_parsing.py
"""Validation and normalization of the run config that extract() receives.

Lifted out of ``extract.py`` so the public entry point stays focused on
orchestration. Everything here is pure-Python config-validation logic —
no GEE calls, no file I/O beyond reading a YAML config file.

Public surface re-imported by ``extract.py``:

* :class:`RunSettings` — immutable container of one parsed output config.
* :func:`_parse_run_configs` — parse every run config of one extract() call
  with :func:`_parse_run_config` and reject batch_id values whose output
  files or folders would collide.
* :func:`_parse_run_config` — turn one raw config dict into RunSettings,
  raising ValueError for any malformed setting or unknown key.
* :func:`_as_config_list` — load YAML (if given a path) and normalize the
  result into a list of output config dicts, rejecting an item that is not
  a mapping.

Most other helpers (``_normalize_dataset_entry``, ``_validate_*``,
``_parse_statistics``, ``_resolve_stats_for_dataset``, ``_load_yaml``) are
private to this file.

The checks against the capability declarations of each dataset's adapter
(output types, reducers, derived bands) read the adapter class from the
registry in ``envoi.adapters``. They build no adapter.
"""

from __future__ import annotations

import difflib
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from ._filenames import describe_unsafe_path_component, is_safe_path_component
from .adapters import all_computable_derived_bands, get_adapter
from .adapters.base import _BAND_NAMING_KEY, _PER_BAND_NAMING
from .catalog import _DATA_TYPES
from .reducers import KNOWN_REDUCERS

# Keys allowed inside the full-form dict value
# (e.g. {"sen2": {"bands": [...]}}). Adding more per-call band_overrides in the
# future means extending this set and validating each new key inside the
# entry-normalization helper below.
_ALLOWED_OVERRIDE_KEYS = frozenset({"bands"})

# The keys of one run config, and the keys of its `settings` mapping:
# exactly the keys that _parse_run_config reads. Any other key is rejected,
# because an ignored key, for example a typing error in `window_size_m`,
# would give a silently different run.
_RUN_CONFIG_KEYS = frozenset({"batch_id", "datasets", "settings"})
_SETTINGS_KEYS = frozenset(
    {
        "output_type",
        "output_file_format",
        "min_coverage_pct",
        "statistics",
        "window_size_m",
        "resample_m",
    }
)


@dataclass
class RunSettings:
    """Validated and parsed settings for a single output run."""

    batch_id: str
    # Each entry is a (dataset_name, band_overrides) pair. `band_overrides` is an empty
    # dict for plain-string entries in the user's `datasets` list. When the user
    # supplied a per-call override (e.g. {"sen2": ["B4", "B8"]} or
    # {"sen2": {"bands": ["B4"]}}), it contains `bands` (an empty list when the
    # override lists only derived bands), `derived_bands` when the override
    # lists derived bands, and the mark `band_naming: "per_band"`.
    datasets: list[tuple[str, dict[str, Any]]]
    output_type: str  # "tabular" or "raster"
    output_file_format: str  # "csv", "parquet", or "dataframe"
    # One or more square-sampling-window sizes in metres. Zero is reserved for
    # point-only tabular runs, where no sampling window is constructed.
    window_sizes: list[int]
    min_coverage: float  # 0–100 — threshold for low-coverage QC flag
    # Normalized stats dict: {"continuous": [...], "categorical": [...]}.
    # A flat list from the user is normalized to identical lists on both keys.
    # Downstream code reads the list of each dataset from `stats_by_dataset`
    # rather than reading this dict directly.
    stats: dict[str, list[str]]
    # The reducer list of each dataset of a tabular output, keyed by dataset
    # name: the `stats` list of the dataset's data_type (see
    # _resolve_stats_for_dataset). Every reducer is in the `supported_reducers`
    # of the dataset's adapter. Empty for raster output.
    stats_by_dataset: dict[str, list[str]]
    # Original user-supplied form (flat list or typed dict), stored verbatim
    # so the metadata sidecar round-trips it without normalizing it away.
    user_stats: list | dict[str, list[str]]
    resample_m: float | None  # target pixel size in metres (raster mode only)
    user_window_size: int | list[int]  # original input form, preserved for metadata


def _normalize_dataset_entry(
    entry: Any,
    batch_id: str,
    catalog_datasets: dict,
) -> tuple[str, dict[str, Any]]:
    """Normalize one item from the user's `datasets` list into (name, band_overrides).

    Three accepted shapes:
      * A plain string  -> ("name", {})
      * Shorthand dict  -> {"name": [bands...]}
                          where the list is a unified band list (source +
                          derived bands mixed). Names that a registered
                          adapter can compute (all_computable_derived_bands)
                          are split into the `derived_bands` override; the
                          rest go to `bands`.
      * Full-form dict  -> {"name": {"bands": [...]}}
                          where the inner dict accepts the keys listed in
                          `_ALLOWED_OVERRIDE_KEYS`. Reserved for future
                          per-call band_overrides; today only `bands` is allowed.

    A band list replaces the catalog's bands: `bands` is always set, and is an
    empty list when the list has only derived bands (``{"dem": ["aspect"]}``
    then gives only the aspect band, not the catalog's source band too). The
    band_overrides of a band list also get the mark ``band_naming: "per_band"``
    (``_BAND_NAMING_KEY`` in ``adapters/base.py``), so the output columns are
    named after the bands, also for one band.

    Raises ValueError for any malformed entry or unknown dataset name, and for
    a derived band that the dataset's adapter does not compute, so a single
    ``except ValueError`` covers every config-shape failure mode.
    """
    # ---- shape 1: plain string ----
    if isinstance(entry, str):
        if not entry:
            raise ValueError(f"Output '{batch_id}': dataset name cannot be empty.")
        if entry not in catalog_datasets:
            raise ValueError(f"Output '{batch_id}': dataset(s) ['{entry}'] not found in catalog.")
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
        raise ValueError(f"Output '{batch_id}': dataset(s) ['{name}'] not found in catalog.")

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
    # what the user wrote. A name that any registered adapter can compute
    # counts as a derived band, also for a dataset whose adapter computes
    # none. Without this, "slope" would become a source band of such a
    # dataset and fail late with an unclear error. So a band list cannot
    # select a source band that has the name of a derived band.
    known_derived_bands = all_computable_derived_bands()
    derived_bands = [band for band in unified_bands if band in known_derived_bands]
    source_bands = [band for band in unified_bands if band not in known_derived_bands]

    # Each adapter declares the derived bands that it can compute. Reject
    # the others before any adapter is built.
    dataset_config = catalog_datasets[name]
    data_source = dataset_config.get("data_source")
    computable_derived_bands = get_adapter(data_source).computable_derived_bands
    uncomputable_derived_bands = sorted(set(derived_bands) - computable_derived_bands)
    if uncomputable_derived_bands:
        computable_text = (
            str(sorted(computable_derived_bands)) if computable_derived_bands else "none"
        )
        raise ValueError(
            f"Output '{batch_id}': dataset '{name}' has data_source '{data_source}', whose "
            f"adapter cannot compute the derived band(s) {uncomputable_derived_bands}. "
            f"Derived bands that this adapter computes: {computable_text}. Remove "
            f"{uncomputable_derived_bands} from the band list of '{name}'. In a band list, "
            f"the names {sorted(known_derived_bands)} always mean derived bands."
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

    # The override replaces the catalog's bands, also when it lists only
    # derived bands: an empty `bands` list then replaces the catalog's source
    # bands in the merged config, so the output has only the derived bands.
    band_overrides: dict[str, Any] = {
        "bands": source_bands,
        _BAND_NAMING_KEY: _PER_BAND_NAMING,
    }
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


def _parse_run_configs(
    run_configs: list[dict],
    defaults: dict,
    catalog_datasets: dict,
) -> list[RunSettings]:
    """Parse every run config of one extract() call, in list order.

    Args:
        run_configs: The raw run config dicts, as ``_as_config_list`` returns them.
        defaults: Project-wide defaults from ``load_defaults()``.
        catalog_datasets: The merged catalog's ``datasets`` dict.

    Returns:
        One ``RunSettings`` per run config, in the same order.

    Raises:
        ValueError: If a run config is invalid (see ``_parse_run_config``), if
            two run configs resolve to ``batch_id`` values that are equal when
            letter case is ignored, or if the QC table of a tabular output
            would have the file name of another tabular output.
    """
    all_run_settings = [
        _parse_run_config(run_config, defaults, index, catalog_datasets)
        for index, run_config in enumerate(run_configs)
    ]

    # Each output writes files named after its batch_id, so two run configs
    # with the same batch_id would overwrite each other's outputs. Windows and
    # macOS ignore letter case in file and folder names by default, so
    # "Terrain" and "terrain" are the same name there. A run config without
    # batch_id gets the default "output<N>", which can be the same as an
    # explicit batch_id of another run config, so compare resolved values.
    batch_ids = [str(run_settings.batch_id) for run_settings in all_run_settings]
    folded_batch_ids = [batch_id.casefold() for batch_id in batch_ids]
    duplicate_batch_ids = sorted(
        {
            batch_id
            for batch_id, folded_batch_id in zip(batch_ids, folded_batch_ids)
            if folded_batch_ids.count(folded_batch_id) > 1
        }
    )
    if duplicate_batch_ids:
        raise ValueError(
            f"Duplicate batch_id value(s) {duplicate_batch_ids} in the config list. "
            f"Each output writes files named after its batch_id, so every run config "
            f"needs a different batch_id. Windows and macOS do not tell upper-case and "
            f"lower-case letters apart in file names, so batch_id values that differ "
            f"only in letter case also count as the same. A run config without "
            f"batch_id gets the name 'output<N>', where N is its position in the list "
            f"(starting at 1)."
        )

    # A tabular output writes its QC table to "<batch_id>_qc.<format>", so a
    # tabular batch_id "X" and a tabular batch_id "X_qc" would write the same
    # file. Raster outputs write into a folder named after the batch_id, so
    # they cannot collide with these files.
    tabular_batch_ids = [
        str(run_settings.batch_id)
        for run_settings in all_run_settings
        if run_settings.output_type == "tabular"
    ]
    tabular_batch_id_by_folded = {batch_id.casefold(): batch_id for batch_id in tabular_batch_ids}
    for batch_id in tabular_batch_ids:
        qc_file_stem = f"{batch_id}_qc"
        colliding_batch_id = tabular_batch_id_by_folded.get(qc_file_stem.casefold())
        if colliding_batch_id is not None:
            raise ValueError(
                f"batch_id values '{batch_id}' and '{colliding_batch_id}' in the config "
                f"list would write the same file: output '{batch_id}' writes its QC "
                f"table as '{qc_file_stem}.<format>', the file name of the statistics "
                f"table of output '{colliding_batch_id}'. Rename one of these batch_id "
                f"values."
            )

    return all_run_settings


def _parse_run_config(
    run_config: dict,
    defaults: dict,
    index: int,
    catalog_datasets: dict,
) -> RunSettings:
    """Parse and validate a single output run config dict into a RunSettings instance.

    Raises ValueError for any invalid or missing setting so callers don't need
    to do any further validation on the returned object. This includes a key
    that is not in ``_RUN_CONFIG_KEYS`` or, inside ``settings``, not in
    ``_SETTINGS_KEYS``, and a setting that the adapter of a dataset does not
    support, by its capability declarations: the output_type, a reducer, or a
    derived band.
    """
    # Use a numbered fallback batch_id when the user didn't provide one.
    batch_id = run_config.get("batch_id", f"output{index + 1}")

    # batch_id becomes a directory name (output_dir/<batch_id>/ for raster
    # tiles) and a file stem ("<batch_id>_metadata.json"), so it has to be a
    # valid path component on every OS we support. Unlike sample IDs — which
    # arrive from a data file and get sanitized — batch_id is hand-authored
    # config, so the right response is to tell the author to fix it. This also
    # blocks a "../" batch_id from writing outside the chosen output directory.
    if not is_safe_path_component(batch_id):
        raise ValueError(
            f"Invalid batch_id {batch_id!r}: {describe_unsafe_path_component(batch_id)}. "
            f"batch_id is used as a folder and file name, so it may contain only "
            f"letters, digits, '.', '_' and '-'."
        )

    # Reject unknown keys before the checks of the required keys, so that a
    # typing error such as 'setting' gets a message that names it, not
    # "missing required 'settings' dict".
    _validate_known_keys(
        run_config,
        _RUN_CONFIG_KEYS,
        "the run config",
        batch_id,
        settings_keys=_SETTINGS_KEYS,
    )

    # datasets is required — must be a non-empty list. Each entry is normalized
    # into a (name, band_overrides) tuple by the helper, which also handles all
    # validation (catalog existence, malformed dicts, derived bands that the
    # dataset's adapter cannot compute, ...).
    raw_datasets = run_config.get("datasets", [])
    if not raw_datasets:
        raise ValueError(f"Output '{batch_id}': missing required 'datasets' list")
    _validate_datasets_is_list(raw_datasets, batch_id)
    datasets = [_normalize_dataset_entry(e, batch_id, catalog_datasets) for e in raw_datasets]

    # settings is required — must be a non-empty mapping with known keys.
    settings = run_config.get("settings", {}) or {}
    if not isinstance(settings, Mapping):
        raise ValueError(
            f"Output '{batch_id}': 'settings' must be a mapping of setting names to values "
            f"(a dict in Python, or indented 'key: value' lines in YAML), got "
            f"{type(settings).__name__}: {settings!r}. Allowed keys: {sorted(_SETTINGS_KEYS)}."
        )
    if not settings:
        raise ValueError(f"Output '{batch_id}': missing required 'settings' dict")
    _validate_known_keys(settings, _SETTINGS_KEYS, "'settings'", batch_id)

    # output_type controls the entire processing path (stats vs tile export).
    output_type = settings.get("output_type")
    if output_type not in ("tabular", "raster"):
        raise ValueError(f"Unknown or missing output_type: {output_type}")

    # Each dataset writes columns or a tile folder named after it, so two
    # entries of one dataset, or two raster folders whose names differ only in
    # letter case, would overwrite each other.
    _validate_unique_dataset_names(datasets, output_type, batch_id)

    # Each adapter declares the output types that it supports. Reject the
    # others here, before any adapter is built or any quota is used.
    for dataset_name, _ in datasets:
        data_source = catalog_datasets[dataset_name].get("data_source")
        supported_output_types = get_adapter(data_source).output_types
        if output_type not in supported_output_types:
            raise ValueError(
                f"Output '{batch_id}': dataset '{dataset_name}' has data_source "
                f"'{data_source}', whose adapter does not support output_type "
                f"'{output_type}'. Supported output types: {sorted(supported_output_types)}. "
                f"Set output_type to one of these, or move '{dataset_name}' to a run config "
                f"with a supported output_type."
            )

    # resample_m is only meaningful for raster output.
    resample_m = settings.get("resample_m")
    if output_type == "raster" and resample_m is not None:
        resample_m = _parse_finite_number(resample_m, "resample_m", batch_id)
        if resample_m <= 0:
            raise ValueError(
                f"Output '{batch_id}': invalid resample_m: {resample_m!r}. "
                f"Must be a positive number."
            )
    if output_type == "tabular" and resample_m is not None:
        raise ValueError("resample_m is not applicable when output_type is 'tabular'.")

    # output_file_format applies to tabular output only.
    output_file_format = settings.get("output_file_format", defaults["output_file_format"])
    if output_file_format not in ("csv", "parquet", "dataframe"):
        raise ValueError(f"Unknown output_file_format: {output_file_format}")

    # min_coverage_pct is a percentage — must be in [0, 100].
    min_coverage = _parse_finite_number(
        settings.get("min_coverage_pct", defaults["min_coverage_pct"]),
        "min_coverage_pct",
        batch_id,
    )
    if min_coverage < 0 or min_coverage > 100:
        raise ValueError(
            f"Output '{batch_id}': invalid min_coverage_pct: {min_coverage!r}. "
            f"Must be between 0 and 100."
        )

    # statistics — required for tabular, forbidden for raster.
    # Accepts either a flat list (applied to all datasets) or a typed dict
    # {"continuous": [...], "categorical": [...]} for mixed-type runs.
    raw_stats = settings.get("statistics")
    if output_type == "raster" and raw_stats:
        raise ValueError("Statistics cannot be computed when output_type is 'raster'.")
    if output_type == "tabular":
        stats, user_stats = _parse_statistics(raw_stats, batch_id)
        stats_by_dataset = _resolve_and_validate_stats_by_dataset(
            datasets, stats, catalog_datasets, batch_id
        )
    else:
        stats, user_stats = {}, None
        stats_by_dataset = {}

    # window_size_m can be either one size or a list. Sizes are positive except
    # for the explicit zero sentinel used by point-only tabular sampling.
    # When the user supplies a list, statistics (or tiles) are produced for
    # each window size and the column / filename suffix disambiguates them.
    user_window_size = settings.get("window_size_m", defaults["window_size_m"])
    if isinstance(user_window_size, (list, tuple)):
        window_sizes = list(user_window_size)
        if not window_sizes:
            raise ValueError(f"Output '{batch_id}': window_size_m list must not be empty.")
    else:
        window_sizes = [user_window_size]
    point_only = (
        output_type == "tabular"
        and bool(stats)
        and all(
            reducers and all(reducer == "point" for reducer in reducers)
            for reducers in stats.values()
        )
    )
    for window_size in window_sizes:
        # Reject non-integers explicitly. window_size feeds f-string column
        # names like "{dataset}_mean_{window_size_m}m", and a float would
        # silently yield columns like "dem_mean_200.0m" — breaking schema
        # expectations downstream. ``bool`` is a subclass of ``int`` in
        # Python, so we filter it out first to avoid accepting True/False.
        invalid_type_or_sign = (
            isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 0
        )
        if invalid_type_or_sign or (window_size == 0 and not point_only):
            raise ValueError(
                f"Output '{batch_id}': invalid window_size_m: {window_size!r}. "
                "Must be a positive integer (or a list of positive integers); "
                "zero is allowed only for point-only tabular sampling."
            )

    return RunSettings(
        batch_id=batch_id,
        datasets=datasets,
        output_type=output_type,
        output_file_format=output_file_format,
        window_sizes=window_sizes,
        min_coverage=min_coverage,
        stats=stats,
        stats_by_dataset=stats_by_dataset,
        user_stats=user_stats,
        resample_m=resample_m,
        user_window_size=user_window_size,
    )


def _validate_known_keys(
    config: Mapping,
    allowed_keys: frozenset[str],
    location: str,
    batch_id: str,
    *,
    settings_keys: frozenset[str] = frozenset(),
) -> None:
    """Raise ValueError if ``config`` has a key that is not in ``allowed_keys``.

    ``_parse_run_config`` would ignore such a key, so a typing error
    (``window_size`` for ``window_size_m``) would give a silently different
    run. For each unknown key, the message says that it belongs inside
    ``settings`` when it is in ``settings_keys``, or else suggests the closest
    allowed key when one is close enough.

    Args:
        config: The run config, or its ``settings`` mapping.
        allowed_keys: The keys that ``_parse_run_config`` reads at this level.
        location: Where the keys are, for the message: ``"the run config"``
            or ``"'settings'"``.
        batch_id: The run config's batch_id, used in the error message.
        settings_keys: The keys of ``settings``, when ``config`` is the run
            config. Empty when ``config`` is ``settings``.

    Raises:
        ValueError: If ``config`` has a key that is not in ``allowed_keys``.
    """
    unknown_keys = [key for key in config if key not in allowed_keys]
    if not unknown_keys:
        return

    # Name the unknown keys in config order. A key can be other than text (the
    # YAML key "1:" gives an int), so compare its text. The cutoff 0.6 (the
    # difflib default) suggests 'window_size_m' for 'window_size' and
    # 'statistics' for 'stats', and nothing for an unrelated key such as
    # 'max_workers'.
    key_descriptions = []
    for key in unknown_keys:
        # A setting written next to 'datasets' is a common error, and no
        # run-config key is close to it.
        if key in settings_keys:
            key_descriptions.append(f"{key!r} (this setting belongs inside 'settings')")
            continue
        close_matches = difflib.get_close_matches(str(key), sorted(allowed_keys), n=1, cutoff=0.6)
        if close_matches:
            key_descriptions.append(f"{key!r} (did you mean {close_matches[0]!r}?)")
        else:
            key_descriptions.append(repr(key))
    raise ValueError(
        f"Output '{batch_id}': {location} has the unknown key(s) "
        f"{', '.join(key_descriptions)}. Allowed keys: {sorted(allowed_keys)}. Correct or "
        f"remove the unknown key(s). envoi does not ignore them, because an ignored key "
        f"would silently change the run."
    )


def _validate_datasets_is_list(raw_datasets: Any, batch_id: str) -> None:
    """Raise ValueError if the ``datasets`` value of a run config is not a list or tuple.

    Without this check, a mapping such as ``{"dem": ["slope"]}`` would give its
    keys and silently drop the band lists, and a text value such as ``"dem"``
    would give its letters as dataset names. For a mapping or a text value,
    the message shows the same entries as a Python list, which is also valid
    YAML (flow style).

    Args:
        raw_datasets: The non-empty ``datasets`` value from the run config.
        batch_id: The run config's batch_id, used in the error message.

    Raises:
        ValueError: If ``raw_datasets`` is not a list or tuple.
    """
    if isinstance(raw_datasets, (list, tuple)):
        return

    message = (
        f"Output '{batch_id}': 'datasets' must be a list of dataset entries, got "
        f"{type(raw_datasets).__name__}: {raw_datasets!r}."
    )

    # Show the list form: one single-key dict per dataset of a mapping, or a
    # list with the one dataset name.
    if isinstance(raw_datasets, Mapping):
        list_form = [{name: value} for name, value in raw_datasets.items()]
    elif isinstance(raw_datasets, str):
        list_form = [raw_datasets]
    else:
        raise ValueError(message)
    raise ValueError(f"{message} Write it as a list: datasets: {list_form!r}.")


def _validate_unique_dataset_names(
    datasets: list[tuple[str, dict[str, Any]]],
    output_type: str,
    batch_id: str,
) -> None:
    """Raise ValueError if the datasets of one run config would write the same output.

    A tabular output names its statistic and QC columns after each dataset, and
    a raster output writes the tiles of each dataset to a folder named after
    it. So a dataset listed twice would overwrite its own output in both modes.
    In raster mode, two dataset names that differ only in letter case
    (``dem`` and ``DEM``) are also rejected: Windows and macOS ignore letter
    case in folder names by default, so the two folders would be one. Column
    names keep their letter case, so tabular mode allows such names.

    Args:
        datasets: The normalized ``(dataset_name, band_overrides)`` pairs.
        output_type: ``"tabular"`` or ``"raster"``.
        batch_id: The run config's batch_id, used in the error message.

    Raises:
        ValueError: If a dataset is listed more than once, or if a raster run
            config lists two datasets whose names differ only in letter case.
    """
    dataset_names = [dataset_name for dataset_name, _ in datasets]

    duplicate_dataset_names = sorted(
        {dataset_name for dataset_name in dataset_names if dataset_names.count(dataset_name) > 1}
    )
    if duplicate_dataset_names:
        raise ValueError(
            f"Output '{batch_id}': dataset(s) {duplicate_dataset_names} are listed more "
            f"than once in 'datasets'. Each dataset writes columns or a tile folder named "
            f"after it, so a second entry would overwrite the first. List each dataset "
            f"once, with all the bands you need in one band list."
        )

    if output_type != "raster":
        return

    dataset_names_by_folded: dict[str, list[str]] = {}
    for dataset_name in dataset_names:
        dataset_names_by_folded.setdefault(dataset_name.casefold(), []).append(dataset_name)
    for names_in_group in dataset_names_by_folded.values():
        if len(names_in_group) > 1:
            raise ValueError(
                f"Output '{batch_id}': datasets {names_in_group} differ only in letter "
                f"case. Raster output writes the tiles of each dataset to a folder named "
                f"after the dataset, and Windows and macOS treat these folder names as "
                f"one folder. Put these datasets in separate run configs (each with its "
                f"own batch_id), or register one of them under another name."
            )


def _parse_finite_number(value: Any, setting_name: str, batch_id: str) -> int | float:
    """Return a numeric setting as a Python ``int`` or ``float``, or raise ValueError.

    Accepts any real number, including NumPy numbers such as ``np.int64`` and
    ``np.float32``. These become Python numbers, because the metadata sidecar
    writes settings with ``json``, which cannot write NumPy numbers. Rejects
    ``bool`` (a subclass of ``int`` in Python, so ``True`` would silently mean
    1), text such as ``"30"``, ``None``, NaN, and infinity.

    Args:
        value: The raw setting value from the run config.
        setting_name: The config key, used in the error message.
        batch_id: The run config's batch_id, used in the error message.

    Returns:
        ``value`` as ``int`` when it is an integer, otherwise as ``float``.

    Raises:
        ValueError: If ``value`` is not a finite real number.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(
            f"Output '{batch_id}': invalid {setting_name}: {value!r}. "
            f"Must be a finite number (not text, true/false, null, NaN, or infinity)."
        )
    return int(value) if isinstance(value, numbers.Integral) else float(value)


def _parse_statistics(
    raw: Any,
    batch_id: str,
) -> tuple[dict[str, list[str]], Any]:
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
        unknown_keys = set(raw.keys()) - _DATA_TYPES
        if unknown_keys:
            raise ValueError(
                f"Output '{batch_id}': unknown 'statistics' key(s) {sorted(unknown_keys)}. "
                f"Allowed: {sorted(_DATA_TYPES)}."
            )
        if not raw:
            raise ValueError(
                f"Output '{batch_id}': 'statistics' dict must contain at least one of "
                f"{sorted(_DATA_TYPES)}."
            )
        normalized: dict[str, list[str]] = {}
        for key, reducers in raw.items():
            if not isinstance(reducers, list) or not reducers:
                raise ValueError(
                    f"Output '{batch_id}': 'statistics.{key}' must be a non-empty list."
                )
            _validate_reducer_names(reducers, batch_id, context=f"statistics.{key}")
            normalized[key] = reducers
        return normalized, raw

    raise ValueError(
        f"Output '{batch_id}': 'statistics' must be a list or dict, got {type(raw).__name__}."
    )


def _validate_reducer_names(reducers: list, batch_id: str, context: str) -> None:
    """Raise ValueError if any reducer name is not in the known set."""
    unknown = [r for r in reducers if r not in KNOWN_REDUCERS]
    if unknown:
        raise ValueError(
            f"Output '{batch_id}': unknown reducer(s) {unknown} in '{context}'. "
            f"Valid reducers: {sorted(KNOWN_REDUCERS)}."
        )


def _resolve_stats_for_dataset(
    data_type: str | None,
    stats: dict[str, list[str]],
    dataset_name: str,
    batch_id: str,
) -> list[str]:
    """Return the reducer list to use for one dataset based on its data_type.

    Falls back to 'continuous' when data_type is None, since most ecological
    rasters are continuous and users often omit data_type for local datasets.
    Raises ValueError when the resolved list would be empty so the user gets a
    clear message rather than a run with no output columns.
    """
    resolved_type = data_type if data_type in _DATA_TYPES else "continuous"
    reducers = stats.get(resolved_type)
    if not reducers:
        raise ValueError(
            f"Output '{batch_id}': dataset '{dataset_name}' has data_type='{data_type}' but "
            f"the '{resolved_type}' statistics list is missing or empty. "
            f"Add a '{resolved_type}' key to the 'statistics' dict in the run config."
        )
    return reducers


def _resolve_and_validate_stats_by_dataset(
    datasets: list[tuple[str, dict[str, Any]]],
    stats: dict[str, list[str]],
    catalog_datasets: dict,
    batch_id: str,
) -> dict[str, list[str]]:
    """Return the reducer list of each dataset of a tabular output, and check it.

    Each list is the ``stats`` list of the dataset's data_type (see
    ``_resolve_stats_for_dataset``). Each reducer must be in the
    ``supported_reducers`` declaration of the dataset's adapter. No adapter
    is built.

    Args:
        datasets: The normalized ``(dataset_name, band_overrides)`` pairs.
        stats: The normalized ``statistics`` setting from ``_parse_statistics``.
        catalog_datasets: The merged catalog's ``datasets`` dict.
        batch_id: The run config's batch_id, used in the error messages.

    Returns:
        The reducer list of each dataset, keyed by dataset name.

    Raises:
        ValueError: If the ``statistics`` setting has no list for the data_type
            of a dataset, or if a reducer is not supported by the adapter of
            a dataset.
    """
    stats_by_dataset: dict[str, list[str]] = {}
    for dataset_name, _ in datasets:
        dataset_config = catalog_datasets[dataset_name]
        reducers = _resolve_stats_for_dataset(
            dataset_config.get("data_type"), stats, dataset_name, batch_id
        )

        # Each adapter declares the reducers that it supports. An unsupported
        # reducer would otherwise fail inside the adapter, after other
        # datasets possibly used quota.
        data_source = dataset_config.get("data_source")
        supported_reducers = get_adapter(data_source).supported_reducers
        unsupported_reducers = [
            reducer for reducer in reducers if reducer not in supported_reducers
        ]
        if unsupported_reducers:
            raise ValueError(
                f"Output '{batch_id}': dataset '{dataset_name}' has data_source "
                f"'{data_source}', whose adapter does not support the reducer(s) "
                f"{unsupported_reducers}. Supported reducers: {sorted(supported_reducers)}. "
                f"Remove {unsupported_reducers} from 'statistics', or move '{dataset_name}' "
                f"to a run config without them."
            )
        stats_by_dataset[dataset_name] = reducers
    return stats_by_dataset


def _load_yaml(path_or_dict):
    """Load a YAML file if given a path, or return the dict/list if already loaded."""
    if isinstance(path_or_dict, (dict, list)):
        return path_or_dict
    with open(path_or_dict) as f:
        return yaml.safe_load(f)


def _as_config_list(config) -> list[dict]:
    """Load and normalize config into a list of output config dicts.

    Raises ValueError if ``config`` is not a dict, a list, or a path to a YAML
    file, or if an item of the list is not a mapping.
    """
    raw = _load_yaml(config)
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        # Each item must be one run config. Without this check, an item such
        # as a dataset name fails later with an AttributeError.
        for index, run_config in enumerate(raw):
            if not isinstance(run_config, Mapping):
                raise ValueError(
                    f"Item {index + 1} of the config list (counting from 1) is a "
                    f"{type(run_config).__name__}, not a run config: {run_config!r}. Each "
                    f"run config must be a mapping (a dict in Python, or 'key: value' lines "
                    f"in YAML) with the keys {sorted(_RUN_CONFIG_KEYS)}."
                )
        return raw
    raise ValueError("config must be a dict, list, or path to a YAML file.")
