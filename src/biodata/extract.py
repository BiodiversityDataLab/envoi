# src/biodata/extract.py
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path

import logging

import pandas as pd
import yaml
import numpy as np
from pyproj import Transformer

from .adapters import get_adapter
from .reducers import get_reducer, validate_reducers
from .output import OutputManager
from .config import load_catalogs
from .qc import compute_qc_flags, extract_date_columns, extract_crs_column
from .metadata import (
    build_dataset_meta,
    build_tile_crs_zones,
    write_metadata,
    summarize_date_info,
    summarize_tile_export,
)


logger = logging.getLogger(__name__)


def _load_yaml(path_or_dict):
    if isinstance(path_or_dict, (dict, list)):
        return path_or_dict
    with open(path_or_dict) as f:
        return yaml.safe_load(f)


def _normalize_cfg(cfg) -> list[dict]:
    """Load and normalize cfg into a list of output config dicts."""
    raw = _load_yaml(cfg)
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    raise ValueError("cfg must be a dict, list, or path to a YAML file.")


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if df is missing any of the required id/lat/lon columns."""
    required = {"id", "lat", "lon"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(
            f"Input DataFrame is missing required column(s): {sorted(missing)}.\n"
            f"Expected columns: id, lat, lon (and optionally: date).\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )


def _validate_crs(df: pd.DataFrame, input_crs: str | None) -> pd.DataFrame:
    """Reproject coordinates to WGS84 if needed, and raise if any lat/lon are out of range."""
    if input_crs is not None:
        input_crs_upper = input_crs.upper()
        if input_crs_upper != "EPSG:4326" and input_crs_upper != "WGS84":
            logger.info(f"Reprojecting coordinates from {input_crs} to WGS84.")
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

    return df


def extract(
    df: pd.DataFrame,
    cfg: str | Path | dict | list,
    catalog: str | Path | dict | list | tuple = (
        "configs/ee_catalog.yml",
        "configs/local_catalog.yml",
    ),
    extra_catalog: str | Path | dict | None = None,
    out_dir: str | Path = "out",
    input_crs: str | None = None,
) -> Dict[str, Path]:
    """Extract environmental data for sample points.

    ``cfg`` is a dict (single output) or list of dicts (multiple output_paths),
    each specifying run_id, datasets, and settings.  Example::

        extract(df, {
            "run_id": "terrain",
            "datasets": ["dem_aster"],
            "settings": {"output_type": "tabular", "statistics": ["mean"], "window_size_m": 200},
        })

    Parameters
    ----------
    input_crs : str, optional
        EPSG code or proj string for the input coordinates (e.g. "EPSG:32634").
        If provided, lat/lon columns are reprojected to WGS84 (EPSG:4326)
        before extraction. If omitted, coordinates are assumed to be WGS84.

    Returns a mapping of output-key -> Path written.
    """
    df_copy = df.copy()
    output_paths: Dict[str, Path] = {}

    _validate_required_columns(df_copy)
    df_copy = _validate_crs(df_copy, input_crs)

    catalog_dict = load_catalogs(catalog, extra_catalog)
    catalog_datasets = catalog_dict["datasets"]

    output_cfgs_list = _normalize_cfg(cfg)

    for idx, output_cfg in enumerate(output_cfgs_list):
        dataset_metas: Dict[str, Any] = {}
        coverage_backlog: Dict[str, Dict[str, Dict[str, int]]] = {}
        date_backlog: Dict[str, Any] = {}
        warnings_backlog: Dict[str, str] = {}

        run_id = output_cfg.get("run_id", f"output{idx+1}")
        datasets: List[str] = output_cfg.get("datasets", [])
        if not datasets:
            raise ValueError(f"Output '{run_id}': missing required 'datasets' list")
        settings = output_cfg.get("settings", {}) or {}
        output_type = settings.get("output_type", None)
        if output_type not in ("tabular", "raster"):
            raise ValueError(f"Unknown or missing output_type: {output_type}")
        resample_m = settings.get("resample_m")
        output_format = settings.get("output_format", "csv")
        min_coverage = settings.get("min_coverage_pct", 0)
        stats = settings.get("statistics")
        window_size = settings.get("window_size_m", 500)

        for dataset in datasets:
            coverage_backlog[dataset] = {}

            if dataset not in catalog_datasets:
                raise KeyError(f"Dataset '{dataset}' not found in catalog {catalog}")
            dataset_cfg = catalog_datasets[dataset]
            source = dataset_cfg.get("source")
            data_type = dataset_cfg.get("data_type")
            dates = df_copy.date.tolist() if "date" in df_copy.columns else None

            AdapterClass = get_adapter(source)
            adapter = AdapterClass(dataset_cfg)

            # ----- output_type: "raster" — export GeoTIFF tiles, skip stats -----
            if output_type == "raster":
                tiles_root = Path(out_dir) / run_id

                # ----- export tiles for this dataset -----
                exported_paths, meta_list = adapter.export_tiles(
                    df_copy.lat,
                    df_copy.lon,
                    window_size,
                    tiles_root,
                    ids=df_copy["id"].tolist(),
                    dates=dates,
                    dataset_name=dataset,
                    resample_m=resample_m,
                )

                # ----- build metadata for tile export -----
                coverage_backlog[dataset]["tiles"] = summarize_tile_export(
                    exported_paths, len(df_copy)
                )
                tile_crs_zones = build_tile_crs_zones(df_copy.lat, df_copy.lon)
                date_summary = summarize_date_info(meta_list)
                if date_summary is not None:
                    date_backlog[dataset] = date_summary
                dataset_metas[dataset] = build_dataset_meta(
                    dataset_cfg,
                    adapter,
                    tile_crs_zones=tile_crs_zones,
                )
                output_paths[f"{run_id}:{dataset}"] = tiles_root / dataset

                continue  # skip stats for this dataset

            elif output_type == "tabular" and stats and "point" in stats:
                # remove point only and put that in adapters
                # ----- "point" reducer: exact pixel at coordinate, no window -----
                if not hasattr(adapter, "fetch_points_batch"):
                    raise ValueError(f"Source '{source}' does not support the 'point' reducer.")
                pt_results = adapter.fetch_points_batch(
                    df_copy.lat,
                    df_copy.lon,
                    dates=dates,
                )
                all_keys = {k for vals, _ in pt_results for k in vals}
                for bk in sorted(all_keys):
                    col = f"{dataset}_{bk}_point" if len(all_keys) > 1 else f"{dataset}_point"
                    df_copy[col] = [r[0].get(bk) for r in pt_results]

                meta_list = [r[1] for r in pt_results]
                qc_df = compute_qc_flags(meta_list, min_coverage_pct=min_coverage)
                qc_df = qc_df.add_prefix(f"{dataset}_").add_suffix("_point")
                extra_dfs = []
                date_df = extract_date_columns(meta_list)
                if not date_df.empty:
                    extra_dfs.append(date_df.add_prefix(f"{dataset}_").add_suffix("_point"))
                    date_backlog[dataset] = summarize_date_info(meta_list)
                crs_df = extract_crs_column(meta_list)
                if not crs_df.empty:
                    extra_dfs.append(crs_df.add_prefix(f"{dataset}_").add_suffix("_point"))
                if extra_dfs:
                    qc_df = pd.concat(
                        [qc_df.reset_index(drop=True)]
                        + [d.reset_index(drop=True) for d in extra_dfs],
                        axis=1,
                    )
                df_copy = pd.concat(
                    [df_copy.reset_index(drop=True), qc_df.reset_index(drop=True)],
                    axis=1,
                )
                coverage_backlog[dataset]["point"] = {
                    "n_zero": int((qc_df[f"{dataset}_coverage_pct_point"] == 0).sum()),
                    "n_full": int((qc_df[f"{dataset}_coverage_pct_point"] == 100).sum()),
                    "total": int(len(pt_results)),
                }

            else:
                # ----- summary stats with window (both server-side and Python-side) -----
                # TODO: this needs some work, have the same name for the fetch function in the adapters
                # validate requested reducers per dataset and data type.
                if stats:
                    warning = validate_reducers(list(stats), data_type, dataset)
                    if warning:
                        warnings_backlog[dataset] = warning

                # ----- Earth Engine -----
                if source == "earth_engine":
                    reducer_names = list(stats)
                    stats_results = adapter.fetch_stats_batch(
                        df_copy.lat,
                        df_copy.lon,
                        window_size,
                        reducer_names,
                        dates=dates,
                    )
                    # For multi-band datasets keys are "{dataset}_{band}_{reducer}" (e.g. "bio01_mean");
                    # for single-band just "{dataset}_{reducer}". dict.fromkeys preserves insertion order.
                    stat_keys = dict.fromkeys(key for stats, _ in stats_results for key in stats)
                    for stat_key in stat_keys:
                        col = f"{dataset}_{stat_key}_{window_size}m"
                        df_copy[col] = [stats.get(stat_key) for stats, _ in stats_results]

                    meta_list = [meta for _, meta in stats_results]

                else:
                    # ----- Python-side stats (local raster path) -----
                    vals_list: List[np.ndarray] = []
                    meta_list: List[Dict[str, Any]] = []

                    results = [
                        adapter.fetch_values(lat, lon, window_size, return_meta=True)
                        for lat, lon in zip(df_copy.lat, df_copy.lon)
                    ]

                    for arr, meta in results:
                        arr = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
                        vals_list.append(arr)
                        meta_list.append(meta)

                    # Apply Python reducers
                    # Multi-band local: vals are shape (n_bands, n_pixels); reduce per band.
                    is_multiband_local = vals_list and vals_list[0].ndim == 2
                    band_nums = adapter.band if is_multiband_local else None

                    default = "point" if data_type == "categorical" else "mean"
                    reducer_names_iter = (
                        list(stats) if stats else [dataset_cfg.get("default_reducer", default)]
                    )
                    for rname in reducer_names_iter:
                        reducer = get_reducer(rname)
                        if is_multiband_local:
                            for b_idx, band_num in enumerate(band_nums):
                                col = f"{dataset}_b{band_num}_{rname}_{window_size}m"
                                df_copy[col] = [
                                    (reducer(v[b_idx]) if v.size else None) for v in vals_list
                                ]
                        else:
                            col = f"{dataset}_{rname}_{window_size}m"
                            df_copy[col] = [(reducer(v) if v.size else None) for v in vals_list]

                # --- QA columns (also window_size-suffixed) ---
                qc_df = compute_qc_flags(meta_list, min_coverage_pct=min_coverage)
                qc_df = qc_df.add_prefix(f"{dataset}_").add_suffix(f"_{window_size}m")
                extra_dfs = []
                date_df = extract_date_columns(meta_list)
                if not date_df.empty:
                    extra_dfs.append(
                        date_df.add_prefix(f"{dataset}_").add_suffix(f"_{window_size}m")
                    )
                    date_backlog[dataset] = summarize_date_info(meta_list)
                crs_df = extract_crs_column(meta_list)
                if not crs_df.empty:
                    extra_dfs.append(
                        crs_df.add_prefix(f"{dataset}_").add_suffix(f"_{window_size}m")
                    )
                if extra_dfs:
                    qc_df = pd.concat(
                        [qc_df.reset_index(drop=True)]
                        + [d.reset_index(drop=True) for d in extra_dfs],
                        axis=1,
                    )
                df_copy = pd.concat(
                    [df_copy.reset_index(drop=True), qc_df.reset_index(drop=True)],
                    axis=1,
                )

                # --- coverage summary for metadata ---
                cov = qc_df[f"{dataset}_coverage_pct_{window_size}m"].fillna(0)
                coverage_backlog[dataset][str(window_size)] = {
                    "n_zero": int((cov == 0).sum()),
                    "n_partial": int(((cov > 0) & (cov < 100)).sum()),
                    "n_full": int((cov == 100).sum()),
                    "total": int(cov.shape[0]),
                }

            # Build dataset metadata after fetching so band_names are cached
            dataset_metas[dataset] = build_dataset_meta(dataset_cfg, adapter)

        # --- after all datasets processed ---
        if output_type == "tabular":
            core_cols = [c for c in ("id", "lat", "lon", "date") if c in df_copy.columns]

            qc_keywordataset = (
                "_in_extent_",
                "_n_pixels_",
                "_had_nodata_",
                "_coverage_pct_",
                "_image_date_used_",
                "_date_clamped_",
                "_date_source_",
                "_region_crs_",
            )
            qc_cols = [c for c in df_copy.columns if any(kw in c for kw in qc_keywordataset)]
            stats_cols = [c for c in df_copy.columns if c not in qc_cols]

            stats_df = df_copy[core_cols + [c for c in stats_cols if c not in core_cols]].copy()
            qc_df = df_copy[core_cols + [c for c in qc_cols if c not in core_cols]].copy()

            output_cfg_om = OutputManager(out_dir, fmt=output_format)
            stats_path = output_cfg_om.write_tabular(stats_df, run_id)
            qc_path = output_cfg_om.write_tabular(qc_df, f"{run_id}_qc")

            write_metadata(
                out_dir,
                run_id,
                output_type=output_type,
                n_points=len(df_copy),
                datasets=dataset_metas,
                config={
                    "statistics": stats,
                    "window_size_m": window_size,
                    "min_coverage_pct": min_coverage,
                },
                quality=coverage_backlog,
                date_info=date_backlog if date_backlog else None,
                warnings=warnings_backlog if warnings_backlog else None,
            )

            output_paths[run_id] = stats_path
            output_paths[f"{run_id}_qc"] = qc_path

        elif output_type == "raster":
            write_metadata(
                Path(out_dir) / run_id,
                run_id,
                output_type=output_type,
                n_points=len(df_copy),
                datasets=dataset_metas,
                config={
                    "window_size_m": window_size,
                    **({"resample_m": resample_m} if resample_m else {}),
                },
                quality=coverage_backlog if coverage_backlog else None,
                date_info=date_backlog if date_backlog else None,
                warnings=warnings_backlog if warnings_backlog else None,
            )

    return output_paths
