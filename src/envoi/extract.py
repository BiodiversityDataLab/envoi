# src/envoi/extract.py
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path

import pandas as pd

from .adapters import get_adapter
from .catalog import load_catalogs, load_defaults, BUILTIN_EE_CATALOG
from .qc import attach_quality_control, split_stats_and_qc
from .reducers import validate_reducers
from .metadata import write_metadata_sidecar
from ._config_parsing import (
    RunSettings,
    _as_config_list,
    _parse_run_config,
    _resolve_stats_for_dataset,
)
from ._input_validation import (
    _parse_and_validate_dates,
    _validate_and_reproject_crs,
    _validate_required_columns,
)
from ._output_assembly import (
    _append_stat_columns,
    _resolve_dataset_metadata,
    _restore_user_column_names,
    _round_stat_columns,
    _write_tabular,
)

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
    id_column: str = "gbifID",
    latitude_column: str = "decimalLatitude",
    longitude_column: str = "decimalLongitude",
    date_column: str = "eventDate",
    write_metadata: bool = True,
) -> Dict[str, Path | pd.DataFrame]:
    """Extract environmental data for a set of geographic sample points.

    For each dataset listed in ``config``, this function samples the environmental
    data at every point in ``df`` and writes the results to ``output_dir``. Two
    output modes are supported:

    - ``"tabular"`` — computes summary statistics (mean, std, etc.) within a
      square window around each point and writes a table (CSV or Parquet). A
      separate QC table is also written with per-point coverage flags.
    - ``"raster"`` — exports a GeoTIFF tile centred on each point and saves
      the tiles to ``output_dir/<batch_id>/<dataset>/``.

    A JSON metadata sidecar is written alongside the output file(s) by default.
    Pass ``write_metadata=False`` to suppress it.

    ``config`` can be a dict (single output), a list of dicts (multiple outputs),
    or a path to a YAML file containing either of those. Each dict must have the
    following structure::

        {
            "batch_id": "terrain",          # used as the output file stem
            "datasets": ["dem_copernicus_glo30"],  # one or more dataset names from the catalog
            "settings": {
                "output_type": "tabular",   # "tabular" or "raster"
                "statistics": ["mean"],     # list (all datasets) or typed dict (see below)
                # Typed-dict form for mixed runs:
                # "statistics": {"continuous": ["mean", "std"], "categorical": ["mode"]},
                "window_size_m": 200,       # sampling window radius in metres
                "output_file_format": "parquet",  # "parquet", "csv", or "dataframe" (tabular only)
                "min_coverage_pct": 0,     # points below this coverage get a QC flag
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
        columns (named ``gbifID``, ``decimalLatitude``, ``decimalLongitude``
        by default, following the GBIF / Darwin Core convention — override the
        names with the ``*_column`` parameters below). An optional date column
        (``eventDate``, YYYY-MM-DD strings) enables nearest-date image selection
        for time-varying datasets.
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
        Defaults to ``"gbifID"`` (GBIF / Darwin Core). The output
        tables will use the same name.
    latitude_column : str, optional
        Name of the input column containing latitude values. Defaults
        to ``"decimalLatitude"`` (GBIF / Darwin Core). The output tables
        will use the same name.
    longitude_column : str, optional
        Name of the input column containing longitude values. Defaults
        to ``"decimalLongitude"`` (GBIF / Darwin Core). The output tables
        will use the same name.
    date_column : str, optional
        Name of the optional input column containing per-row dates.
        Defaults to ``"eventDate"`` (GBIF / Darwin Core). If absent from
        ``df`` the date branch is simply skipped — the extractor never
        errors on a missing date column. The output tables will use the
        same name.
    write_metadata : bool, optional
        Whether to write the auxiliary files alongside the main output(s).
        When ``True`` (the default), the JSON metadata sidecar is written for
        every output and, in tabular mode, the per-point QC table is also
        written to disk. Set to ``False`` to suppress both — useful when
        running interactively and the output will not be kept on disk.

    Returns
    -------
    dict[str, Path | pd.DataFrame]
        Mapping of output key to the result. For tabular outputs the key is
        ``"<batch_id>"`` pointing to the stats table. For raster outputs the key is
        ``"<batch_id>:<dataset>"`` pointing to the tiles folder. When
        ``output_file_format`` is ``"dataframe"``, the stats value is a pandas
        DataFrame rather than a file path.
    """
    output_paths: Dict[str, Path | pd.DataFrame] = {}

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

        # Accumulate QC column names across all datasets and window sizes so that
        # split_stats_and_qc can use the explicit list rather than substring matching.
        all_qc_columns: list[str] = []

        for dataset, band_overrides in datasets:
            # Look up the dataset's catalog entry once; both processing modes need it.
            dataset_config = catalog_datasets[dataset]

            for window_size in window_sizes:
                # Dispatch to the mode-specific helper. Each helper owns adapter
                # instantiation, data fetching, and per-dataset metadata assembly
                # for one (dataset, window_size) pair.

                if output_type == "tabular":
                    (
                        df_copy,
                        dataset_meta,
                        dataset_qc_columns,
                        reducer_warning,
                    ) = _process_dataset_tabular(
                        df_copy,
                        dataset,
                        dataset_config,
                        run_settings,
                        dates,
                        window_size,
                        band_overrides=band_overrides,
                    )
                    all_qc_columns.extend(dataset_qc_columns)
                    # Record the reducer/data_type mismatch (if any) in the
                    # per-dataset warnings backlog. The message is identical
                    # across windows for the same dataset, so a dict
                    # assignment naturally deduplicates repeat windows.
                    if reducer_warning:
                        warnings_backlog[f"reducer_{dataset}"] = reducer_warning

                # In raster mode, the helper exports a folder of GeoTIFF tiles
                # and returns the path plus the per-dataset metadata dict. With
                # multiple windows the filename suffix keeps each window's
                # tiles distinct inside the same dataset folder.
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
            stats_df, qc_df = split_stats_and_qc(df_copy, core_columns, all_qc_columns)

            stats_df = _round_stat_columns(
                stats_df, core_columns, defaults["stats_output_decimals"]
            )
            stats_df, qc_df = _restore_user_column_names(stats_df, qc_df, column_name_map)

            # When format is "dataframe", return the DataFrames directly instead
            # of writing them to disk. CSV/Parquet still write as before.
            if output_file_format == "dataframe":
                output_paths[batch_id] = stats_df

            else:
                stats_path = _write_tabular(
                    stats_df, batch_id, Path(output_dir), output_file_format
                )
                output_paths[batch_id] = stats_path

            if write_metadata:
                _write_tabular(qc_df, f"{batch_id}_qc", Path(output_dir), output_file_format)
                write_metadata_sidecar(
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

        elif output_type == "raster":
            if write_metadata:
                write_metadata_sidecar(
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
) -> tuple[pd.DataFrame, dict, list[str], str | None]:
    """Fetch stats and QC columns for one dataset/window pair in tabular mode.

    Owns adapter instantiation, statistic computation, QC column assembly,
    and per-dataset metadata construction for a single window size. The
    caller is responsible for looping over multiple window sizes and merging
    the resulting metadata dicts.

    `band_overrides` is a dict of per-call settings (currently `bands` and/or
    `derived_bands`) that replace the catalog values before adapter
    instantiation. The catalog dict itself is never mutated — the override
    is shallow-merged into a fresh dict per call.

    Returns ``(df, dataset_meta, qc_column_names, reducer_warning)`` where
    ``reducer_warning`` is the message from :func:`validate_reducers` when
    the requested stats don't match the dataset's data_type (e.g. ``mean``
    on a categorical raster) — or ``None`` when everything is fine. The
    caller threads this into the metadata sidecar's ``warnings`` section.
    """
    # Build a fresh shallow-merged config so per-call band_overrides replace the
    # catalog values for this run only. The catalog dict itself is not
    # mutated, so concurrent runs / repeat calls remain isolated.
    merged_config = {**dataset_config, **(band_overrides or {})}

    # Instantiate the adapter from the dataset's declared data_source
    # (earth_engine or local). The adapter encapsulates all source-specific
    # fetching and validation logic. The `with` block guarantees that any
    # adapter-held resources (e.g. an open rasterio dataset) are released
    # even if the per-point batch raises mid-flight.
    AdapterClass = get_adapter(merged_config["data_source"])
    with AdapterClass(merged_config) as adapter:
        # Select the reducer list appropriate for this dataset's data_type.
        # For typed-dict statistics configs, a continuous dataset uses the
        # "continuous" list and a categorical dataset uses "categorical".
        # For flat-list configs both keys map to the same list, so the result
        # is unchanged from the previous behaviour.
        data_type = merged_config.get("data_type")
        resolved_stats = _resolve_stats_for_dataset(
            data_type, run_settings.stats, dataset, run_settings.batch_id
        )

        # Flag reducer/data_type mismatches (e.g. `mean` on a categorical
        # raster, or `class_count` on a continuous one). The message is
        # emitted via warnings.warn inside validate_reducers and also
        # returned here so the caller can persist it in the metadata
        # sidecar alongside the date / CRS warnings.
        reducer_warning = validate_reducers(resolved_stats, data_type, dataset)

        # Ask the adapter to compute all requested statistics for every sample.
        # Returns a list of (stats_dict, meta_dict) — one tuple per row in df.
        # progress_desc is shown in the per-point tqdm bar so the user can tell at
        # a glance which dataset/window the bar belongs to when several run in series.
        stats_results = adapter.fetch_stats_batch(
            df.lat,
            df.lon,
            window_size,
            resolved_stats,
            dates=dates,
            progress_desc=f"{dataset} | {window_size}m | tabular",
        )

        # Append stat columns to the output DataFrame based on the keys in
        # stats_results, and assign values accordingly.
        df = _append_stat_columns(df, dataset, window_size, stats_results)

        # Split the (stat, meta) tuples so meta_list can feed QC and dataset metadata.
        meta_list = [meta for _, meta in stats_results]

        # Attach per-row QC columns and capture the coverage summary used below
        # when assembling the per-dataset metadata dict.
        df, quality_key, coverage_summary, qc_column_names = attach_quality_control(
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

    return df, dataset_meta, qc_column_names, reducer_warning


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

    # Instantiate the adapter from the dataset's declared data_source. The
    # `with` block guarantees that any adapter-held resources (e.g. an open
    # rasterio dataset) are released after tile export, even if it raises.
    AdapterClass = get_adapter(merged_config["data_source"])
    with AdapterClass(merged_config) as adapter:
        # Ask the adapter to export a tile per sample, writing GeoTIFFs under
        # tiles_root / dataset / ... and returning the paths it wrote.
        # progress_desc is shown in the per-tile tqdm bar so the user can tell at
        # a glance which dataset/window the bar belongs to when several run in series.
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
            progress_desc=f"{dataset} | {window_size}m | raster",
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
