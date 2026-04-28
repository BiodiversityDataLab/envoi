# src/biodata/extract.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
import yaml
from pyproj import Transformer

from .adapters import get_adapter
from .config import load_catalogs, load_defaults, BUILTIN_EE_CATALOG
from .qc import attach_quality_control, split_stats_and_qc
from .metadata import write_metadata

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
                "statistics": ["mean"],     # required for tabular; forbidden for raster
                "window_size_m": 200,       # sampling window radius in metres
                "output_file_format": "parquet",  # "parquet" or "csv" (tabular only)
                "min_coverage_pct": 80,     # points below this coverage get a QC flag
                "resample_m": 10,           # output pixel size in metres (raster only)
            },
        }

    Parameters
    ----------
    df : pd.DataFrame
        Input table of sample points. Must contain columns ``id``, ``lat``, and
        ``lon``. An optional ``date`` column (YYYY-MM-DD strings) enables
        nearest-date image selection for time-varying datasets.
    config : str, Path, dict, or list
        Output specification — see above. A path string or ``Path`` is loaded as
        YAML before processing.
    output_dir : str or Path, optional
        Directory where all output files are written. Created automatically if it
        does not exist. Defaults to ``"outputs"``.
    input_crs : str or None, optional
        CRS of the coordinates in ``df`` as an EPSG code or proj string
        (e.g. ``"EPSG:32634"``). When provided, ``lat`` and ``lon`` are
        reprojected to WGS84 (EPSG:4326) before extraction. If omitted,
        coordinates are assumed to already be in WGS84.

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

    # Load project-wide defaults from the bundled configs/defaults.yml once per call.
    # These are the fallback values used when settings are not specified
    # in the run config or as keyword arguments to this function.
    defaults = load_defaults()

    # Validate inputs
    _validate_required_columns(df)
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
        window_size = run_settings.window_size
        min_coverage = run_settings.min_coverage
        stats = run_settings.stats
        resample_m = run_settings.resample_m

        for dataset in datasets:
            # Look up the dataset's catalog entry once; both processing modes need it.
            dataset_config = catalog_datasets[dataset]

            # Dispatch to the mode-specific helper. Each helper owns adapter
            # instantiation, data fetching, and per-dataset metadata assembly.

            # In tabular mode, the helper returns the input DataFrame with calculated summary statistics and QC columns appended, plus the per-dataset metadata dict.
            if output_type == "tabular":
                df_copy, dataset_meta = _process_dataset_tabular(
                    df_copy,
                    dataset,
                    dataset_config,
                    run_settings,
                    dates,
                )

            # In raster mode, the helper exports a folder of GeoTIFF tiles and returns the path plus the per-dataset metadata dict.
            # TODO: QC for raster mode is not yet implemented. Needed?
            elif output_type == "raster":
                tiles_root = Path(output_dir) / batch_id
                tiles_path, dataset_meta = _process_dataset_raster(
                    dataset,
                    dataset_config,
                    run_settings,
                    dates,
                    lats=df_copy.lat,
                    lons=df_copy.lon,
                    ids=df_copy["id"].tolist(),
                    tiles_root=tiles_root,
                )
                output_paths[f"{batch_id}:{dataset}"] = tiles_path

            # Store the per-dataset metadata assembled by the adapter in the dataset_metas dict for later inclusion in the output metadata file.
            dataset_metas[dataset] = dataset_meta

        # --- after all datasets processed for this run, write outputs and metadata ---

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

            # Round stat columns to the specified number of decimals so the output stays readable.
            # Core columns (id/lat/lon/date) are excluded so coordinate
            # precision is preserved exactly as the user supplied it.
            stat_columns = [c for c in stats_df.columns if c not in core_columns]
            stats_df[stat_columns] = stats_df[stat_columns].round(defaults["stats_output_decimals"])

            stats_path = _write_tabular(stats_df, batch_id, Path(output_dir), output_file_format)
            qc_path = _write_tabular(qc_df, f"{batch_id}_qc", Path(output_dir), output_file_format)

            write_metadata(
                output_dir,
                batch_id,
                output_type=output_type,
                n_points=len(df_copy),
                datasets=dataset_metas,
                config={
                    "statistics": stats,
                    "window_size_m": window_size,
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
                    "window_size_m": window_size,
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
) -> tuple[pd.DataFrame, dict]:
    """Fetch stats and QC columns for one dataset in tabular mode.

    Owns adapter instantiation, statistic computation, QC column assembly,
    and per-dataset metadata construction. Returns the input DataFrame with
    new stat + QC columns appended, plus the per-dataset metadata dict.
    """
    # Instantiate the adapter from the dataset's declared data_source
    # (earth_engine or local). The adapter encapsulates all source-specific
    # fetching and validation logic.
    AdapterClass = get_adapter(dataset_config["data_source"])
    adapter = AdapterClass(dataset_config)

    # Ask the adapter to compute all requested statistics for every sample.
    # Returns a list of (stats_dict, meta_dict) — one tuple per row in df.
    stats_results = adapter.fetch_stats_batch(
        df.lat,
        df.lon,
        run_settings.window_size,
        run_settings.stats,
        dates=dates,
    )

    # Append stat columns to the output DataFrame based on the keys in
    # stats_results, and assign values accordingly.
    df = _append_stat_columns(df, dataset, run_settings.window_size, stats_results)

    # Split the (stat, meta) tuples so meta_list can feed QC and dataset metadata.
    meta_list = [meta for _, meta in stats_results]

    # Attach per-row QC columns and capture the coverage summary used below
    # when assembling the per-dataset metadata dict.
    df, quality_key, coverage_summary = attach_quality_control(
        df,
        meta_list=meta_list,
        dataset_name=dataset,
        reducer_names=run_settings.stats,
        window_size_m=run_settings.window_size,
        min_coverage_pct=run_settings.min_coverage,
    )

    # Adapter assembles quality stats and date-selection info (GEE) into
    # one per-dataset dict. The quality dict is keyed by window_size (or
    # "point") so the metadata layout supports future multi-window runs.
    dataset_meta = adapter.build_dataset_meta(
        dataset_config,
        meta_list=meta_list,
        quality={quality_key: coverage_summary},
    )

    return df, dataset_meta


def _process_dataset_raster(
    dataset: str,
    dataset_config: dict,
    run_settings: "RunSettings",
    dates: list | None,
    *,
    lats,
    lons,
    ids: list,
    tiles_root: Path,
) -> tuple[Path, dict]:
    """Export GeoTIFF tiles for one dataset in raster mode.

    Owns adapter instantiation, tile export, and per-dataset metadata
    construction. Returns the path to the per-dataset tiles folder and
    the per-dataset metadata dict.
    """
    # Instantiate the adapter from the dataset's declared data_source.
    AdapterClass = get_adapter(dataset_config["data_source"])
    adapter = AdapterClass(dataset_config)

    # Ask the adapter to export a tile per sample, writing GeoTIFFs under
    # tiles_root / dataset / ... and returning the paths it wrote.
    exported_paths, meta_list = adapter.export_tiles(
        lats,
        lons,
        run_settings.window_size,
        tiles_root,
        ids=ids,
        dates=dates,
        dataset_name=dataset,
        resample_m=run_settings.resample_m,
    )

    # Adapter assembles the full per-dataset metadata dict, including
    # per-tile export info (paths, bounds, CRS).
    dataset_meta = adapter.build_dataset_meta(
        dataset_config,
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
    datasets: List[str]
    output_type: str  # "tabular" or "raster"
    output_file_format: str  # "csv" or "parquet"
    window_size: int  # square sampling window in metres
    min_coverage: float  # 0–100 — threshold for low-coverage QC flag
    stats: list | None  # requested reducer names, or None to use the default
    resample_m: float | None  # target pixel size in metres (raster mode only)


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

    # datasets is required — must be a non-empty list of dataset names.
    datasets = run_config.get("datasets", [])
    if not datasets:
        raise ValueError(f"Output '{batch_id}': missing required 'datasets' list")

    # Validate that every requested dataset exists in the catalog.
    missing_datasets = [d for d in datasets if d not in catalog_datasets]
    if missing_datasets:
        raise KeyError(f"Output '{batch_id}': dataset(s) {missing_datasets} not found in catalog.")

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

    # statistics (reducer names) are not supported for raster output.
    stats = settings.get("statistics")
    if output_type == "raster" and stats:
        raise ValueError("Statistics cannot be computed when output_type is 'raster'.")
    if output_type == "tabular" and (not isinstance(stats, list) or len(stats) == 0):
        raise ValueError("For tabular output, 'statistics' must be provided as a non-empty list.")

    # window_size_m must be a positive number of metres.
    window_size = settings.get("window_size_m", defaults["window_size_m"])
    if window_size <= 0:
        raise ValueError(f"Invalid window_size_m: {window_size}. Must be a positive number.")

    return RunSettings(
        batch_id=batch_id,
        datasets=datasets,
        output_type=output_type,
        output_file_format=output_file_format,
        window_size=window_size,
        min_coverage=min_coverage,
        stats=stats,
        resample_m=resample_m,
    )


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


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if df is missing any of the required id/lat/lon columns."""
    required_columns = {"id", "lat", "lon"}
    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(
            f"Input DataFrame is missing required column(s): {sorted(missing_columns)}.\n"
            f"Expected columns: id, lat, lon (and optionally: date).\n"
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
