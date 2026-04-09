# src/biodata/enrich.py
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
import yaml
import numpy as np

from .adapters import get_adapter
from .reducers import get_reducer
from .output import OutputManager
from .config import load_catalogs
from .qc import compute_qc_flags
from .metadata import build_feature_meta, build_tile_crs_zones, write_metadata


def _load_yaml(path_or_dict):
    if isinstance(path_or_dict, (dict, list)):
        return path_or_dict
    with open(path_or_dict) as f:
        return yaml.safe_load(f)


def enrich(
    df: pd.DataFrame,
    cfg: str | Path | dict | list,
    catalog: str | Path | dict | list | tuple = (
        "configs/ee_catalog.yml",
        "configs/local_catalog.yml",
    ),
    extra_catalog: str | Path | dict | None = None,
    out_dir: str | Path = "out",
) -> Dict[str, Path]:
    """Enrich sample points with environmental data.

    ``cfg`` is a dict (single output) or list of dicts (multiple outputs),
    each specifying name, predictors, and output settings.  Example::

        enrich(df, {
            "name": "terrain",
            "predictors": ["dem_aster"],
            "output": {"kind": "tabular", "reducers": ["mean"], "window_m": 200},
        })

    Returns a mapping of output-key -> Path written.
    """
    required = {"id", "lat", "lon"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(
            f"Input DataFrame is missing required column(s): {sorted(missing)}.\n"
            f"Expected columns: id, lat, lon (and optionally: date).\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )

    catalog_dict = load_catalogs(catalog, extra_catalog)
    cat = catalog_dict["datasets"]

    outputs: Dict[str, Path] = {}

    # --- Normalise cfg into a list of output dicts ---
    raw = _load_yaml(cfg)
    if isinstance(raw, dict):
        output_list = [raw]
    elif isinstance(raw, list):
        output_list = raw
    else:
        raise ValueError("cfg must be a dict, list, or path to a YAML file.")

    for idx, g in enumerate(output_list):
        gname = g.get("name", f"output{idx+1}")
        feats: List[str] = g.get("features") or g.get("predictors", [])
        out_cfg = g.get("output", {}) or {}
        kind = out_cfg.get("kind", "tabular")

        resample_m = out_cfg.get("resample_m")
        fmt = out_cfg.get("format", "csv")
        min_cov = out_cfg.get("min_coverage_pct", 80)

        stats = out_cfg.get("reducers")
        buffers = [out_cfg.get("window_m", 500)]

        work = df.copy()
        feature_metas: Dict[str, Any] = {}
        coverage_backlog: Dict[str, Dict[str, Dict[str, int]]] = {}

        for p in feats:
            if p not in cat:
                raise KeyError(f"Feature '{p}' not found in catalog {catalog}")
            spec = cat[p]
            source = spec.get("source")

            AdapterCls = get_adapter(source)
            adapter = AdapterCls(spec)
            coverage_backlog[p] = {}

            feature_meta_first: Dict[str, Any] | None = None
            dates = work.date.tolist() if "date" in work.columns else None

            # ----- kind: "raster" — export GeoTIFF tiles, skip stats -----
            if kind == "raster":
                buf = buffers[0]
                ids = work["id"].tolist() if "id" in work.columns else None
                dates_list = work.date.tolist() if "date" in work.columns else None
                tiles_root = Path(out_dir) / gname

                if hasattr(adapter, "export_images"):
                    adapter.export_images(
                        work.lat, work.lon, buf, tiles_root,
                        ids=ids, dates=dates_list, feature_name=p,
                        resample_m=resample_m,
                    )
                elif hasattr(adapter, "export_windows"):
                    adapter.export_windows(
                        work.lat, work.lon, buf, tiles_root,
                        ids=ids, feature_name=p,
                        resample_m=resample_m,
                    )
                else:
                    raise ValueError(
                        f"Source '{source}' does not support raster export."
                    )
                tile_crs_zones = build_tile_crs_zones(work.lat, work.lon)
                feature_metas[p] = build_feature_meta(
                    spec, adapter,
                    tile_crs_zones=tile_crs_zones,
                )
                outputs[f"{gname}:{p}"] = tiles_root / p
                continue  # skip stats for this feature

            # ----- "point" reducer: exact pixel at coordinate, no window -----
            if stats and "point" in list(stats):
                if not hasattr(adapter, "fetch_points_batch"):
                    raise ValueError(
                        f"Source '{source}' does not support the 'point' reducer."
                    )
                pt_results = adapter.fetch_points_batch(
                    work.lat, work.lon, dates=dates,
                )
                all_keys = {k for vals, _ in pt_results for k in vals}
                for bk in sorted(all_keys):
                    col = f"{p}_{bk}_point" if len(all_keys) > 1 else f"{p}_point"
                    work[col] = [r[0].get(bk) for r in pt_results]

                meta_list = [r[1] for r in pt_results]
                if feature_meta_first is None:
                    for m in meta_list:
                        if m and m.get("in_extent"):
                            feature_meta_first = m
                            break

                qc_df = compute_qc_flags(meta_list, min_coverage_pct=min_cov)
                qc_df = qc_df.add_prefix(f"{p}_").add_suffix("_point")
                work = pd.concat(
                    [work.reset_index(drop=True), qc_df.reset_index(drop=True)],
                    axis=1,
                )
                coverage_backlog[p]["point"] = {
                    "n_zero": int((qc_df[f"{p}_coverage_pct_point"] == 0).sum()),
                    "n_full": int((qc_df[f"{p}_coverage_pct_point"] == 100).sum()),
                    "total": int(len(pt_results)),
                }

            for buf in buffers:
                if stats and "point" in list(stats):
                    break  # already handled above

                # ----- Server-side stats (GEE fast path) -----
                use_server_stats = (
                    hasattr(adapter, "fetch_stats_batch")
                    and stats
                )

                if use_server_stats:
                    reducer_names = list(stats)
                    ss_results = adapter.fetch_stats_batch(
                        work.lat, work.lon, buf, reducer_names, dates=dates,
                    )
                    # Use actual result keys — for multi-band datasets these are
                    # "{band}_{reducer}" (e.g. "bio01_mean"); for single-band just
                    # "{reducer}" (e.g. "mean"). dict.fromkeys preserves insertion order.
                    all_stat_keys = dict.fromkeys(k for r, _ in ss_results for k in r)
                    for key in all_stat_keys:
                        col = f"{p}_{key}_{buf}m"
                        work[col] = [r[0].get(key) for r in ss_results]

                    meta_list = [r[1] for r in ss_results]
                    if feature_meta_first is None:
                        for m in meta_list:
                            if m and m.get("in_extent"):
                                feature_meta_first = m
                                break

                else:
                    # ----- Python-side stats (local raster path) -----
                    if hasattr(adapter, "fetch_batch"):
                        results = adapter.fetch_batch(
                            work.lat, work.lon, buf,
                            dates=dates, return_meta=True,
                        )
                    else:
                        results = [
                            adapter.fetch_values(lat, lon, buf, return_meta=True)
                            for lat, lon in zip(work.lat, work.lon)
                        ]

                    vals_list: List[np.ndarray] = []
                    meta_list: List[Dict[str, Any]] = []
                    for arr, meta in results:
                        arr = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
                        vals_list.append(arr)
                        meta_list.append(meta)
                        if feature_meta_first is None and meta:
                            feature_meta_first = meta

                    # Apply Python reducers
                    # Multi-band local: vals are shape (n_bands, n_pixels); reduce per band.
                    is_multiband_local = vals_list and vals_list[0].ndim == 2
                    band_nums = adapter.band if is_multiband_local else None

                    reducer_names_iter = list(stats) if stats else [spec.get("default_reducer", "mean")]
                    for rname in reducer_names_iter:
                        reducer = get_reducer(rname)
                        if is_multiband_local:
                            for b_idx, band_num in enumerate(band_nums):
                                col = f"{p}_b{band_num}_{rname}_{buf}m"
                                work[col] = [
                                    (reducer(v[b_idx]) if v.size else None) for v in vals_list
                                ]
                        else:
                            col = f"{p}_{rname}_{buf}m"
                            work[col] = [(reducer(v) if v.size else None) for v in vals_list]

                # --- QA columns (also buffer-suffixed) ---
                qc_df = compute_qc_flags(meta_list, min_coverage_pct=min_cov)
                qc_df = qc_df.add_prefix(f"{p}_").add_suffix(f"_{buf}m")
                work = pd.concat(
                    [work.reset_index(drop=True), qc_df.reset_index(drop=True)],
                    axis=1,
                )

                # --- coverage summary for metadata ---
                cov = qc_df[f"{p}_coverage_pct_{buf}m"].fillna(0)
                coverage_backlog[p][str(buf)] = {
                    "n_zero": int((cov == 0).sum()),
                    "n_partial": int(((cov > 0) & (cov < 100)).sum()),
                    "n_full": int((cov == 100).sum()),
                    "total": int(cov.shape[0]),
                }

            # Build feature metadata after fetching so band_names are cached
            feature_metas[p] = build_feature_meta(spec, adapter)

        # --- after all features processed ---
        if kind == "tabular":
            core_cols = [c for c in ("id", "lat", "lon", "date") if c in work.columns]

            qc_keywords = ("_in_extent_", "_n_pixels_", "_had_nodata_", "_coverage_pct_")
            qc_cols = [c for c in work.columns if any(kw in c for kw in qc_keywords)]
            stats_cols = [c for c in work.columns if c not in qc_cols]

            stats_df = work[core_cols + [c for c in stats_cols if c not in core_cols]].copy()
            qc_df = work[core_cols + [c for c in qc_cols if c not in core_cols]].copy()

            group_om = OutputManager(out_dir, fmt=fmt)
            stats_path = group_om.write_tabular(stats_df, gname)
            qc_path = group_om.write_tabular(qc_df, f"{gname}_qc")

            write_metadata(
                out_dir, gname,
                kind=kind,
                n_points=len(work),
                features=feature_metas,
                config={
                    "reducers": stats,
                    "window_m": buffers[0],
                    "min_coverage_pct": min_cov,
                },
                quality=coverage_backlog,
            )

            outputs[gname] = stats_path
            outputs[f"{gname}_qc"] = qc_path

        elif kind == "raster":
            write_metadata(
                Path(out_dir) / gname, gname,
                kind=kind,
                n_points=len(work),
                features=feature_metas,
                config={
                    "window_m": buffers[0],
                    **({"resample_m": resample_m} if resample_m else {}),
                },
            )
        else:
            raise ValueError(f"Unknown output kind: {kind}")

    return outputs
