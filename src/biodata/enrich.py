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
from .provenance import build_provenance
from .output import write_group_parquet


def _load_yaml(path_or_dict) -> Dict[str, Any]:
    if isinstance(path_or_dict, (dict, list)):
        return path_or_dict
    with open(path_or_dict) as f:
        return yaml.safe_load(f)


def enrich(
    df: pd.DataFrame,
    predictors: List[str] | None = None,  # legacy flat mode
    catalog: str | Path | dict = "configs/catalog.yml",
    extra_catalog: str | Path | dict | None = None,  # user-level additions/overrides
    groups: str | dict | None = None,  # group mode
    out_dir: str | Path = "out",
    window_m: int = 500,
    temporal: str = "nearest_month",  # reserved for future sources
    cache_dir: str = "~/.biodata_cache",
    out_path: str | Path | None = None,  # reserved for future sources
) -> Dict[str, Path]:
    """
    Enrich points either:
      A) with a flat predictor list (legacy), writing a single tabular 'flat' output, or
      B) using 'groups' that specify outputs per group (tabular or demo raster).

    Returns: mapping of output-key -> Path written (groups) or a DataFrame (legacy flat mode with out_path=None).
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

    om = OutputManager(out_dir)
    outputs: Dict[str, Path] = {}

    # -------- Mode B: groups specified --------
    if groups is not None:
        gcfg = _load_yaml(groups)
        groups_list = gcfg.get("groups", [])
        min_cov = gcfg.get("min_coverage_pct", 80)

        for idx, g in enumerate(groups_list):
            gname = g.get("name", f"group{idx+1}")
            feats: List[str] = g.get("features") or g.get("predictors", [])
            out_cfg = g.get("output", {}) or {}
            kind = out_cfg.get("kind", "tabular")

            resample_m = out_cfg.get("resample_m")  # target resolution for CNN-ready tiles

            stats = g.get("summary_statistics") or out_cfg.get("reducers")
            buffers = g.get("buffer_sizes") or [out_cfg.get("window_m", window_m)]

            work = df.copy()
            provenance: Dict[str, Any] = {}
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
                    tiles_root = Path(out_dir) / "tiles" / gname

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
                        # ss_results: list of (stats_dict, meta_dict)
                        for rname in reducer_names:
                            col = f"{p}_{rname}_b{buf}"
                            work[col] = [r[0].get(rname) for r in ss_results]

                        meta_list = [r[1] for r in ss_results]
                        if feature_meta_first is None:
                            for m in meta_list:
                                if m and m.get("in_extent"):
                                    feature_meta_first = m
                                    break

                    else:
                        # ----- Python-side stats (local raster path / tile dumps) -----
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
                        if stats:
                            for rname in stats:
                                reducer = get_reducer(rname)
                                col = f"{p}_{rname}_b{buf}"
                                work[col] = [(reducer(v) if v.size else None) for v in vals_list]
                        else:
                            default_r = spec.get("default_reducer", "mean")
                            reducer = get_reducer(default_r)
                            col = f"{p}_{default_r}_b{buf}"
                            work[col] = [(reducer(v) if v.size else None) for v in vals_list]

                    # --- QA columns (also buffer-suffixed) ---
                    qc_df = compute_qc_flags(meta_list, min_coverage_pct=min_cov)
                    qc_df = qc_df.add_prefix(f"{p}_").add_suffix(f"_b{buf}")
                    work = pd.concat(
                        [work.reset_index(drop=True), qc_df.reset_index(drop=True)],
                        axis=1,
                    )

                    # --- coverage summary for metadata ---
                    cov = qc_df[f"{p}_coverage_pct_b{buf}"].fillna(0)
                    coverage_backlog[p][str(buf)] = {
                        "n_zero": int((cov == 0).sum()),
                        "n_partial": int(((cov > 0) & (cov < 100)).sum()),
                        "n_full": int((cov == 100).sum()),
                        "total": int(cov.shape[0]),
                    }

# --- provenance for this feature ---
                provenance[p] = build_provenance(
                    spec,
                    stats or [spec.get("default_reducer", "mean")],
                    buffers,
                    temporal,
                    feature_meta_first or {},
                )

        # --- after all features for this group are processed ---
        if kind == "tabular":
            # Core ID columns we always keep
            core_cols = [c for c in ("id", "lat", "lon", "date") if c in work.columns]

            # QC columns: the *_in_extent_b*, *_n_pixels_b*, *_had_nodata_b*, *_coverage_pct_b* ones
            qc_suffixes = (
                "_in_extent_b",
                "_n_pixels_b",
                "_had_nodata_b",
                "_coverage_pct_b",
            )
            qc_cols = [c for c in work.columns if any(suf in c for suf in qc_suffixes)]

            # Stats columns = everything that's not QC
            stats_cols = [c for c in work.columns if c not in qc_cols]

            stats_df = work[core_cols + [c for c in stats_cols if c not in core_cols]].copy()
            qc_df = work[core_cols + [c for c in qc_cols if c not in core_cols]].copy()

            # Metadata: include provenance + coverage_backlog
            meta_info = {
                "provenance": provenance,
                "coverage_backlog": coverage_backlog,
            }
            cfg_for_meta = {
                **gcfg,
                "min_coverage_pct": min_cov,
                "summary_statistics": stats,
                "buffer_sizes": buffers,
                "out_dir": out_dir,
            }

            # Write stats parquet + metadata JSON
            stats_path = write_group_parquet(
                stats_df,
                gname,
                meta_info,
                cfg_for_meta,
            )

            # Write QC parquet (no separate metadata file for now)
            qc_path = om.write_tabular(qc_df, f"{gname}_qc")

            outputs[gname] = stats_path
            outputs[f"{gname}_qc"] = qc_path

        elif kind == "raster":
            pass  # outputs already populated inside the per-feature loop above
        else:
            raise ValueError(f"Unknown output kind: {kind}")

        return outputs

    # -------- Mode A: flat predictor list (back-compat) --------
    if groups is None:
        if predictors is None:
            raise ValueError("Provide either `groups` or a flat `predictors` list.")

        out = df.copy()
        for p in predictors:
            if p not in cat:
                raise KeyError(f"Predictor '{p}' not found in catalog {catalog}")
            spec = cat[p]

            source = spec.get("source")
            AdapterCls = get_adapter(source)
            adapter = AdapterCls(spec)

            default_r = spec.get("default_reducer", "mean")
            dates = out.date.tolist() if "date" in out.columns else None

            if hasattr(adapter, "fetch_stats_batch"):
                # Server-side single-stat (GEE fast path)
                ss_results = adapter.fetch_stats_batch(
                    out.lat, out.lon, window_m, [default_r], dates=dates,
                )
                out[p] = [r[0].get(default_r) for r in ss_results]
            elif hasattr(adapter, "fetch_batch"):
                reducer = get_reducer(default_r)
                batch_vals = adapter.fetch_batch(
                    out.lat, out.lon, window_m, dates=dates, return_meta=False,
                )
                out[p] = [reducer(np.asarray(v)) for v in batch_vals]
            else:
                reducer = get_reducer(default_r)
                out[p] = [
                    reducer(adapter.fetch_values(lat, lon, window_m))
                    for lat, lon in zip(out.lat, out.lon)
                ]

        # write tabular: honor out_path if provided, else return the DataFrame (legacy)
        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if str(out_path).endswith(".parquet"):
                out.to_parquet(out_path, index=False)
            else:
                out.to_csv(out_path, index=False)
            return out
        else:
            return out
