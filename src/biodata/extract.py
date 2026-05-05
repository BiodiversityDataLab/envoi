# src/biodata/extract.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path

import pandas as pd
import yaml
from pyproj import Transformer

from .adapters import get_adapter
from .adapters.gee_adapter import KNOWN_DERIVED_BANDS
from .config import load_catalogs, load_defaults, BUILTIN_EE_CATALOG
from .qc import attach_quality_control, split_stats_and_qc
from .metadata import write_metadata

# Keys allowed inside the full-form dict value
# (e.g. {"sen2": {"bands": [...]}}). Adding more per-call band_overrides in the
# future means extending this set and validating each new key inside the
# entry-normalization helper below.
_ALLOWED_OVERRIDE_KEYS = frozenset({"bands"})

# Default catalog tuple used when the caller does not supply one.
# The sentinels tell load_catalogs() to load the YAML files that are
# bundled inside the installed package, so this works regardless of
# the user's working directory.
_DEFAULT_CATALOGS = (BUILTIN_EE_CATALOG,)


def extract(
    df: pd.DataFrame,
    config: str | Path | dict | list,
    output_dir: str | Path = "outputs",
    input_crs: str | None = None,
    id_column: str = "id",
    latitude_column: str = "lat",
    longitude_column: str = "lon",
    date_column: str = "date",
) -> Dict[str, Path]:
    """Extract environmental data for a set of geographic sample points.

    For each dataset listed in ``config``, this function samples the environmental
    data at every point in ``df`` and writes the results to ``output_dir``. Two
    output modes are supported:

    - ``"tabular"`` — computes summary statistics (mean, std, etc.) within a
      square window around each point and writes a table (CSV or Parquet). A
      separate QC table is also written with per-point coverage flags.
    - ``"raster"`` — exports a GeoTIFF tile centred on each point and saves
      the tiles to ``output_dir/<batch_id>/<dataset>/``.

    A JSON metadata sidecar is always written alongside the output file(s).

    ``config`` can be a dict (single output), a list of dicts (multiple outputs),
    or a path to a YAML file containing either of those. Each dict must have the
    following structure::

        {
            "batch_id": "terrain",          # used as the output file stem
            "datasets": ["dem_aster"],      # one or more dataset names from the catalog
            "settings": {
                "output_type": "tabular",   # "tabular" or "raster"
                "statistics": ["mean"],     # list (all datasets) or typed dict (see below)
                # Typed-dict form for mixed runs:
                # "statistics": {"continuous": ["mean", "std"], "categorical": ["mode"]},
                "window_size_m": 200,       # sampling window radius in metres
                "output_file_format": "parquet",  # "parquet" or "csv" (tabular only)
                "min_coverage_pct": 80,     # points below this coverage get a QC flag
                "resample_m": 10,           # output pixel size in metres (raster only)
            },
        }

    Each item inside ``datasets`` can take three shapes:

    - A plain dataset name (string) — uses the catalog's default bands.
    - A shorthand single-key dict ``{"sen2": ["B4", "B8"]}`` — the list is a
      unified band list. Names recognised as derived (currently ``"slope"``
      and ``"aspect"``) are split out internally and computed from the
      dataset's first band; the rest are selected as source bands. The
      override **replaces** the catalog's bands for this run only.
    - A full-form single-key dict ``{"sen2": {"bands": ["B4", "B8"]}}`` —
      reserved for future per-call settings; today only ``"bands"`` is
      accepted.

    A single-element list (``{"sen2": ["B4"]}``) keeps the multi-band column
    naming (``sen2_B4_mean_<window>m``); use the catalog if you want the
    bare single-band naming (``sen2_mean_<window>m``).

    Parameters
    ----------
    df : pd.DataFrame
        Input table of sample points. Must contain identifier and coordinate
        columns (named ``id``, ``lat``, ``lon`` by default — override the names
        with the ``*_column`` parameters below). An optional date column
        (YYYY-MM-DD strings) enables nearest-date image selection for
        time-varying datasets.
    config : str, Path, dict, or list
        Output specification — see above. A path string or ``Path`` is loaded as
        YAML before processing.
    output_dir : str or Path, optional
        Directory where all output files are written. Created automatically if it
        does not exist. Defaults to ``"outputs"``.
    input_crs : str or None, optional
        CRS of the coordinates in ``df`` as an EPSG code or proj string
        (e.g. ``"EPSG:32634"``). When provided, the latitude and longitude
        columns are reprojected to WGS84 (EPSG:4326) before extraction. If
        omitted, coordinates are assumed to already be in WGS84.
    id_column : str, optional
        Name of the input column containing each row's identifier.
        Defaults to ``"id"``. The output tables will use the same name.
    latitude_column : str, optional
        Name of the input column containing latitude values. Defaults
        to ``"lat"``. The output tables will use the same name.
    longitude_column : str, optional
        Name of the input column containing longitude values. Defaults
        to ``"lon"``. The output tables will use the same name.
    date_column : str, optional
        Name of the optional input column containing per-row dates.
        Defaults to ``"date"``. If absent from ``df`` the date branch is
        simply skipped — the extractor never errors on a missing date column.
        The output tables will use the same name.

    Returns
    -------
    dict[str, Path]
        Mapping of output key to the file path that was written. For tabular
        outputs the keys are ``"<batch_id>"`` (stats table) and
        ``"<batch_id>_qc"`` (QC table). For raster outputs the key is
        ``"<batch_id>:<dataset>"`` pointing to the tiles folder.
    """
    output_paths: Dict[str, Path] = {}

    df = df.copy()

    # Translate the user's column names to the canonical names used throughout
    # the pipeline. Validation runs against the user-supplied names so error
    # messages mention the columns the user actually has, then we rename to the
    # canonical names ("id", "lat", "lon", "date") for the rest of processing.
    # The output tables are renamed back to the user's names just before being
    # written, so the user's chosen names round-trip through extract().
    column_name_map = {
        id_column: "id",
        latitude_column: "lat",
        longitude_column: "lon",
        date_column: "date",
    }
    _validate_required_columns(df, id_column, latitude_column, longitude_column)
    df = df.rename(columns=column_name_map)

    # Load project-wide defaults from the bundled configs/defaults.yml once per call.
    # These are the fallback values used when settings are not specified
    # in the run config or as keyword arguments to this function.
    defaults = load_defaults()

    # Validate inputs
    df, dates, date_warnings = _parse_and_validate_dates(df)
    df, crs_warnings = _validate_and_reproject_crs(df, input_crs)

    # Merge the built-in catalogs with any datasets registered via update_catalog().
    # The user catalog is always applied last, so user entries override built-ins.
    catalog_dict = load_catalogs(_DEFAULT_CATALOGS)
    catalog_datasets = catalog_dict["datasets"]

    # Normalize config into a list of output configs, whether the user passed a single dict or a list of dicts.
    output_configs = _as_config_list(config)

    for idx, run_config in enumerate(output_configs):
        df_copy = df.copy()

        # Per-dataset metadata accumulator. Each adapter builds its own entry
        # (including any quality stats and date-selection info) and stores it here.
        dataset_metas: Dict[str, Any] = {}
        warnings_backlog: Dict[str, str] = {}

        # Parse and validate all settings from this run's config dict.
        # ValueError is raised for any missing or invalid setting
        run_settings = _parse_run_config(run_config, defaults, idx, catalog_datasets)
        batch_id = run_settings.batch_id
        datasets = run_settings.datasets
        output_type = run_settings.output_type
        output_file_format = run_settings.output_file_format
        window_sizes = run_settings.window_sizes
        min_coverage = run_settings.min_coverage
        resample_m = run_settings.resample_m

        # When the user supplies more than one window size we need to keep the
        # window suffix on raster filenames and on the per-window quality
        # entries; with a single window we keep the legacy naming and metadata
        # layout so existing outputs are byte-for-byte unchanged.
        is_multi_window = len(window_sizes) > 1

        for dataset, band_overrides in datasets:
            # Look up the dataset's catalog entry once; both processing modes need it.
            dataset_config = catalog_datasets[dataset]

            for window_size in window_sizes:
                # Dispatch to the mode-specific helper. Each helper owns adapter
                # instantiation, data fetching, and per-dataset metadata assembly
                # for one (dataset, window_size) pair.

                if output_type == "tabular":
                    df_copy, dataset_meta = _process_dataset_tabular(
                        df_copy,
                        dataset,
                        dataset_config,
                        run_settings,
                        dates,
                        window_size,
                        band_overrides=band_overrides,
                    )

                # In raster mode, the helper exports a folder of GeoTIFF tiles
                # and returns the path plus the per-dataset metadata dict. With
                # multiple windows the filename suffix keeps each window's
                # tiles distinct inside the same dataset folder.
                # TODO: QC for raster mode is not yet implemented. Needed?
                elif output_type == "raster":
                    tiles_root = Path(output_dir) / batch_id
                    suffix = f"{window_size}m" if is_multi_window else None
                    tiles_path, dataset_meta = _process_dataset_raster(
                        dataset,
                        dataset_config,
                        run_settings,
                        dates,
                        window_size,
                        lats=df_copy.lat,
                        lons=df_copy.lon,
                        ids=df_copy["id"].tolist(),
                        tiles_root=tiles_root,
                        filename_suffix=suffix,
                        band_overrides=band_overrides,
                    )
                    # Single-window: same key as before. Multi-window: append
                    # the window so callers can locate each window's tiles.
                    output_key = (
                        f"{batch_id}:{dataset}:{window_size}m"
                        if is_multi_window
                        else f"{batch_id}:{dataset}"
                    )
                    output_paths[output_key] = tiles_path

                # Merge the per-window metadata into the dataset's accumulator.
                # First window for a dataset → store as-is (including static
                # fields like data_source, native_crs). Subsequent windows →
                # only their quality entries are merged in, since static
                # fields are identical across windows for the same dataset.
                if dataset not in dataset_metas:
                    if is_multi_window and output_type == "raster":
                        # Wrap the raster quality dict (currently {"tiles": ...})
                        # under the window key so each window's tile summary
                        # lands at quality["{window}m"]["tiles"].
                        existing_quality = dataset_meta.get("quality", {})
                        dataset_meta["quality"] = {f"{window_size}m": existing_quality}
                    dataset_metas[dataset] = dataset_meta
                else:
                    accumulated_quality = dataset_metas[dataset].setdefault("quality", {})
                    new_quality = dataset_meta.get("quality", {})
                    if is_multi_window and output_type == "raster":
                        accumulated_quality[f"{window_size}m"] = new_quality
                    else:
                        # Tabular quality is already keyed by window size, so
                        # merging dicts gives the right structure for free.
                        accumulated_quality.update(new_quality)

        # --- after all datasets processed for this run, write outputs and metadata ---

        # Build the resolved per-dataset list for the metadata sidecar.
        resolved_datasets = _resolve_dataset_metadata(datasets, catalog_datasets)

        # If any incomplete dates were detected during parsing, add a summary message to the warnings backlog so it gets recorded in the metadata file.
        if date_warnings:
            warnings_backlog["date_parsing"] = "; ".join(date_warnings)

        # If coordinates were reprojected to WGS84, record a note in the metadata file.
        if crs_warnings:
            warnings_backlog["crs"] = "; ".join(crs_warnings)

        # In tabular mode, split stats and QC into separate DataFrames and write them out as CSV or Parquet.
        # In raster mode, the tiles have already been written by the adapter, so just write the metadata.

        if output_type == "tabular":
            # Keep id/lat/lon/date on both output files so each is self-contained.
            core_columns = [c for c in ("id", "lat", "lon", "date") if c in df_copy.columns]
            stats_df, qc_df = split_stats_and_qc(df_copy, core_columns)

            stats_df = _round_stat_columns(
                stats_df, core_columns, defaults["stats_output_decimals"]
            )
            stats_df, qc_df = _restore_user_column_names(stats_df, qc_df, column_name_map)

            stats_path = _write_tabular(stats_df, batch_id, Path(output_dir), output_file_format)
            qc_path = _write_tabular(qc_df, f"{batch_id}_qc", Path(output_dir), output_file_format)

            write_metadata(
                output_dir,
                batch_id,
                output_type=output_type,
                n_points=len(df_copy),
                datasets=dataset_metas,
                config={
                    # Resolved per-dataset settings (name + effective bands /
                    # derived_bands) so the metadata fully captures what was run.
                    "datasets": resolved_datasets,
                    # Preserve the user's original form (flat list or typed dict)
                    # so the metadata round-trips it without normalizing.
                    "statistics": run_settings.user_stats,
                    # Preserve the user's input form: int stays int, list stays list.
                    "window_size_m": run_settings.user_window_size,
                    "min_coverage_pct": min_coverage,
                },
                warnings=warnings_backlog if warnings_backlog else None,
            )

            output_paths[batch_id] = stats_path
            output_paths[f"{batch_id}_qc"] = qc_path

        elif output_type == "raster":
            write_metadata(
                Path(output_dir) / batch_id,
                batch_id,
                output_type=output_type,
                n_points=len(df_copy),
                datasets=dataset_metas,
                config={
                    # Resolved per-dataset settings — see tabular branch above.
                    "datasets": resolved_datasets,
                    # Preserve the user's input form: int stays int, list stays list.
                    "window_size_m": run_settings.user_window_size,
                    **({"resample_m": resample_m} if resample_m else {}),
                },
                warnings=warnings_backlog if warnings_backlog else None,
            )

    return output_paths


def _process_dataset_tabular(
    df: pd.DataFrame,
    dataset: str,
    dataset_config: dict,
    run_settings: "RunSettings",
    dates: list | None,
    window_size: int,
    band_overrides: Dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch stats and QC columns for one dataset/window pair in tabular mode.

    Owns adapter instantiation, statistic computation, QC column assembly,
    and per-dataset metadata construction for a single window size. The
    caller is responsible for looping over multiple window sizes and merging
    the resulting metadata dicts.

    `band_overrides` is a dict of per-call settings (currently `bands` and/or
    `derived_bands`) that replace the catalog values before adapter
    instantiation. The catalog dict itself is never mutated — the override
    is shallow-merged into a fresh dict per call.
    """
    # Build a fresh shallow-merged config so per-call band_overrides replace the
    # catalog values for this run only. The catalog dict itself is not
    # mutated, so concurrent runs / repeat calls remain isolated.
    merged_config = {**dataset_config, **(band_overrides or {})}

    # Instantiate the adapter from the dataset's declared data_source
    # (earth_engine or local). The adapter encapsulates all source-specific
    # fetching and validation logic.
    AdapterClass = get_adapter(merged_config["data_source"])
    adapter = AdapterClass(merged_config)

    # Select the reducer list appropriate for this dataset's data_type.
    # For typed-dict statistics configs, a continuous dataset uses the
    # "continuous" list and a categorical dataset uses "categorical".
    # For flat-list configs both keys map to the same list, so the result
    # is unchanged from the previous behaviour.
    data_type = merged_config.get("data_type")
    resolved_stats = _resolve_stats_for_dataset(
        data_type, run_settings.stats, dataset, run_settings.batch_id
    )

    # Ask the adapter to compute all requested statistics for every sample.
    # Returns a list of (stats_dict, meta_dict) — one tuple per row in df.
    stats_results = adapter.fetch_stats_batch(
        df.lat,
        df.lon,
        window_size,
        resolved_stats,
        dates=dates,
    )

    # Append stat columns to the output DataFrame based on the keys in
    # stats_results, and assign values accordingly.
    df = _append_stat_columns(df, dataset, window_size, stats_results)

    # Split the (stat, meta) tuples so meta_list can feed QC and dataset metadata.
    meta_list = [meta for _, meta in stats_results]

    # Attach per-row QC columns and capture the coverage summary used below
    # when assembling the per-dataset metadata dict.
    df, quality_key, coverage_summary = attach_quality_control(
        df,
        meta_list=meta_list,
        dataset_name=dataset,
        reducer_names=resolved_stats,
        window_size_m=window_size,
        min_coverage_pct=run_settings.min_coverage,
    )

    # Adapter assembles quality stats and date-selection info (GEE) into
    # one per-dataset dict. The quality dict is keyed by window_size (or
    # "point") so multiple window sizes can coexist within the same dataset
    # metadata when the caller loops and merges across windows.
    # Pass the merged spec (catalog + band_overrides) so the metadata reflects
    # the resolved bands the user actually ran with.
    dataset_meta = adapter.build_dataset_meta(
        merged_config,
        meta_list=meta_list,
        quality={quality_key: coverage_summary},
    )

    return df, dataset_meta


def _process_dataset_raster(
    dataset: str,
    dataset_config: dict,
    run_settings: "RunSettings",
    dates: list | None,
    window_size: int,
    *,
    lats,
    lons,
    ids: list,
    tiles_root: Path,
    filename_suffix: str | None = None,
    band_overrides: Dict[str, Any] | None = None,
) -> tuple[Path, dict]:
    """Export GeoTIFF tiles for one dataset/window pair in raster mode.

    When ``filename_suffix`` is set, it is appended to each tile's filename
    (before the extension) so multiple window sizes for the same dataset
    can coexist in the same folder without overwriting each other.

    `band_overrides` is a dict of per-call settings (currently `bands` and/or
    `derived_bands`) that replace the catalog values before adapter
    instantiation. See `_process_dataset_tabular` for details.
    """
    # Build a fresh shallow-merged config so per-call band_overrides replace the
    # catalog values without mutating the catalog dict.
    merged_config = {**dataset_config, **(band_overrides or {})}

    # Instantiate the adapter from the dataset's declared data_source.
    AdapterClass = get_adapter(merged_config["data_source"])
    adapter = AdapterClass(merged_config)

    # Ask the adapter to export a tile per sample, writing GeoTIFFs under
    # tiles_root / dataset / ... and returning the paths it wrote.
    exported_paths, meta_list = adapter.export_tiles(
        lats,
        lons,
        window_size,
        tiles_root,
        ids=ids,
        dates=dates,
        dataset_name=dataset,
        resample_m=run_settings.resample_m,
        filename_suffix=filename_suffix,
    )

    # Adapter assembles the full per-dataset metadata dict, including
    # per-tile export info (paths, bounds, CRS). Pass the merged spec so
    # the metadata reflects the resolved bands the user actually ran with.
    dataset_meta = adapter.build_dataset_meta(
        merged_config,
        meta_list=meta_list,
        exported_paths=exported_paths,
        lats=lats,
        lons=lons,
    )

    return tiles_root / dataset, dataset_meta


def _parse_and_validate_dates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list | None, list[str]]:
    """Parse and validate the 'date' column from the input DataFrame.

    Returns a tuple of (df, dates, date_warnings) where df has rows with missing
    dates removed, dates is a list of YYYY-MM-DD strings (or None if no 'date'
    column is present), and date_warnings is a list of messages the caller can
    print and record in the output metadata.
    """
    date_warnings: list[str] = []

    if "date" not in df.columns:
        message = "No 'date' column found in input DataFrame; proceeding without dates."
        print(message)
        date_warnings.append(message)
        return df, None, date_warnings

    # Drop rows with missing dates rather than raising, so the user still gets
    # results for the valid rows. The dropped ids are recorded in the warnings
    # so the user can see exactly which points were skipped.
    null_date_mask = df["date"].isna()
    if null_date_mask.any():
        null_ids = df.loc[null_date_mask, "id"].tolist()
        message = (
            f"Skipping {len(null_ids)} row(s) with missing dates "
            f"(ids: {null_ids}). Provide a date for every row to include them."
        )
        print(message)
        date_warnings.append(message)
        df = df.loc[~null_date_mask].copy()

    raw_dates = df["date"].tolist()

    # format="mixed" lets pandas infer the format per-element, which is needed
    # to accept a mix of full ("2021-06-15"), year-month ("2021-06"), and
    # year-only ("2021") dates in the same column. Without this, pandas locks
    # onto the first element's format and raises on any entry that doesn't match.
    try:
        parsed_dates = pd.to_datetime(raw_dates, format="mixed")
    except Exception as e:
        raise ValueError(f"Error parsing 'date' column: {e}. Expected dates in YYYY-MM-DD format.")

    # Detect incomplete dates (year-only "2002" or year-month "2002-02") by splitting
    # on "-". A complete date has 3 parts (YYYY-MM-DD); year-only has 1, year-month
    # has 2. This correctly handles single-digit months/days like "2002-1-1"
    # (still 3 parts → complete)
    for raw_date, parsed_date in zip(raw_dates, parsed_dates):
        raw_date_str = str(raw_date).strip()
        if len(raw_date_str.split("-")) < 3:
            message = (
                f"Date '{raw_date_str}' interpreted as {parsed_date.strftime('%Y-%m-%d')}. "
                f"Provide a full YYYY-MM-DD date if you want a specific day."
            )
            print(message)
            date_warnings.append(message)

    return df, parsed_dates.strftime("%Y-%m-%d").tolist(), date_warnings


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
    output_file_format: str  # "csv" or "parquet"
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
    data_source = catalog_datasets[name].get("data_source")
    if derived_bands and data_source != "earth_engine":
        raise ValueError(
            f"Output '{batch_id}': dataset '{name}' is a {data_source!r} raster — "
            f"derived bands {sorted(set(derived_bands))} are only supported for "
            f"earth_engine datasets."
        )

    band_overrides: Dict[str, Any] = {}
    if source_bands:
        band_overrides["bands"] = source_bands
    if derived_bands:
        band_overrides["derived_bands"] = derived_bands
    return name, band_overrides


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
    if output_file_format not in ("csv", "parquet"):
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

    # window_size_m can be either a single positive number or a list of them.
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
        if not isinstance(window_size, (int, float)) or window_size <= 0:
            raise ValueError(f"Invalid window_size_m: {window_size}. Must be a positive number.")

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


_VALID_STAT_TYPES = frozenset({"continuous", "categorical"})
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
        "q05",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "q95",
    }
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


def _round_stat_columns(
    stats_df: pd.DataFrame,
    core_columns: list[str],
    decimals: int,
) -> pd.DataFrame:
    """Round all non-core stat columns to the given number of decimal places.

    Core columns (id, lat, lon, date) are excluded so coordinate precision
    is preserved exactly as the user supplied it.
    """
    stat_columns = [c for c in stats_df.columns if c not in core_columns]
    stats_df[stat_columns] = stats_df[stat_columns].round(decimals)
    return stats_df


def _restore_user_column_names(
    stats_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    column_name_map: Dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rename canonical core columns back to the user's original names.

    `column_name_map` maps user names → canonical names (e.g. "sample_id" →
    "id"). We invert it to rename the output back so the user's chosen names
    round-trip through extract().
    """
    output_rename = {
        canonical: user for user, canonical in column_name_map.items() if user != canonical
    }
    if output_rename:
        stats_df = stats_df.rename(columns=output_rename)
        qc_df = qc_df.rename(columns=output_rename)
    return stats_df, qc_df


def _resolve_dataset_metadata(
    datasets: List[Tuple[str, Dict[str, Any]]],
    catalog_datasets: dict,
) -> List[Dict[str, Any]]:
    """Build the resolved per-dataset list stored in the metadata sidecar.

    For each (name, band_overrides) pair, records the bands and derived_bands
    that were actually used — the per-call override when supplied, otherwise
    the catalog's value, otherwise None. Storing this alongside the run config
    makes a run reproducible without having to consult the catalog separately.
    """
    resolved = []
    for dataset, band_overrides in datasets:
        dataset_config = catalog_datasets[dataset]
        resolved.append(
            {
                "name": dataset,
                "bands": band_overrides.get("bands", dataset_config.get("bands")),
                "derived_bands": band_overrides.get(
                    "derived_bands", dataset_config.get("derived_bands")
                ),
            }
        )
    return resolved


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


def _append_stat_columns(
    df: pd.DataFrame,
    dataset_name: str,
    window_size_m: int,
    stats_results: list[tuple[dict, dict]],
) -> pd.DataFrame:
    """Append adapter statistics columns to the output dataframe.

    Expected stat keys from adapters:
    - window, single-band: "{reducer}"      -> column "{dataset}_{reducer}_{buf}m"
    - window, multi-band:  "b{n}_{reducer}" -> column "{dataset}_b{n}_{reducer}_{buf}m"
    - point,  single-band: "point"          -> column "{dataset}_point"
    - point,  multi-band:  "{band}_point"   -> column "{dataset}_{band}_point"

    Point keys intentionally skip the buffer suffix to preserve the historical schema.
    """
    # Collect keys in first-seen order so output columns stay stable across runs.
    stat_keys = dict.fromkeys(key for stat_dict, _ in stats_results for key in stat_dict)

    for stat_key in stat_keys:
        if stat_key == "point" or stat_key.endswith("_point"):
            column_name = f"{dataset_name}_{stat_key}"
        else:
            column_name = f"{dataset_name}_{stat_key}_{window_size_m}m"

        # Pull one value per sample row, defaulting to None if that key is absent.
        df[column_name] = [stat_dict.get(stat_key) for stat_dict, _ in stats_results]

    return df


def _write_tabular(df: pd.DataFrame, name: str, output_dir: Path, output_file_format: str) -> Path:
    """Write a DataFrame to a CSV or Parquet file and return the path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_file_format == "csv":
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
    else:
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
    return path


def _validate_required_columns(
    df: pd.DataFrame,
    id_column: str,
    latitude_column: str,
    longitude_column: str,
) -> None:
    """Raise ValueError if df is missing any of the required id/lat/lon columns.

    Uses the user-supplied column names so the error message points at the
    columns the caller is actually expecting to find — not the canonical
    internal names.
    """
    required_columns = {id_column, latitude_column, longitude_column}
    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(
            f"Input DataFrame is missing required column(s): {sorted(missing_columns)}.\n"
            f"Expected columns: {id_column}, {latitude_column}, {longitude_column} "
            f"(and optionally a date column).\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )


def _validate_and_reproject_crs(
    df: pd.DataFrame, input_crs: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """Reproject coordinates to WGS84 if needed, and raise if any lat/lon are out of range.

    Returns a tuple of (df, crs_warnings) where crs_warnings is a list of messages
    about CRS handling (e.g. a reprojection notice) that the caller can record in
    the output metadata alongside the warnings_backlog.
    """
    crs_warnings: list[str] = []

    if input_crs is not None:
        input_crs_upper = input_crs.upper()
        if input_crs_upper != "EPSG:4326" and input_crs_upper != "WGS84":
            message = f"Reprojecting coordinates from {input_crs} to EPSG:4326 (WGS84)."
            print(message)
            crs_warnings.append(message)
            transformer = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
            lons, lats = transformer.transform(df["lon"].values, df["lat"].values)
            df["lon"] = lons
            df["lat"] = lats

    bad_lat_mask = df["lat"].abs() > 90
    bad_lon_mask = df["lon"].abs() > 180
    bad_mask = bad_lat_mask | bad_lon_mask
    if bad_mask.any():
        bad_rows = df.loc[bad_mask, ["id", "lat", "lon"]]
        problems = []
        if bad_lat_mask.any():
            problems.append("latitude values outside ±90°")
        if bad_lon_mask.any():
            problems.append("longitude values outside ±180°")
        raise ValueError(
            f"Coordinates appear to not be in WGS84 (EPSG:4326): {'; '.join(problems)}.\n"
            f"Rows with invalid coordinates:\n{bad_rows.to_string(index=False)}\n"
            f"If your coordinates are in a different CRS, pass input_crs='EPSG:XXXX'"
        )

    return df, crs_warnings
