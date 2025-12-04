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
from .output import write_group_parquet, write_window_tiff


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
        raise ValueError(f"Missing required columns: {missing}")

    catalog_dict = load_catalogs(catalog, extra_catalog)
    cat = catalog_dict["datasets"]

    om = OutputManager(out_dir)
    outputs: Dict[str, Path] = {}

    # -------- Mode B: groups specified --------
    if groups is not None:
        gcfg = _load_yaml(groups)
        groups_list = gcfg.get("groups", [])
        min_cov = gcfg.get("min_coverage_pct", 80)
        proj_crs = gcfg.get("project_crs", "EPSG:3006")

        for idx, g in enumerate(groups_list):
            gname = g.get("name", f"group{idx+1}")
            feats: List[str] = g.get("features") or g.get("predictors", [])
            out_cfg = g.get("output", {}) or {}
            kind = out_cfg.get("kind", "tabular")

            # Optional: dump per-point raster windows as GeoTIFF tiles
            dump_windows = bool(out_cfg.get("dump_windows", False))
            tiles_root = Path(out_dir) / "tiles" / gname  # out/tiles/<group>/

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

                for buf in buffers:
                    vals_list: List[np.ndarray] = []
                    meta_list: List[Dict[str, Any]] = []

                    # --- fetch values + meta for each point (for this buffer) ---
                    for lat, lon in zip(work.lat, work.lon):
                        arr, meta = adapter.fetch_values(lat, lon, buf, return_meta=True)
                        arr = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
                        vals_list.append(arr)
                        meta_list.append(meta)
                        if feature_meta_first is None and meta:
                            feature_meta_first = meta

                    # --- reducers → columns (ALWAYS buffer-suffixed) ---
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

                    # --- optional: dump tiles for debugging/inspection ---
                    if dump_windows:
                        tiles_dir = tiles_root / p / f"b{buf}"
                        tiles_dir.mkdir(parents=True, exist_ok=True)

                        for ridx, vmeta in enumerate(meta_list):
                            # use id col if present, else row index
                            pid = work.iloc[ridx]["id"] if "id" in work.columns else ridx
                            tile_path = tiles_dir / f"id{pid}.tif"

                            arr2d = vmeta.get("window_arr")
                            if arr2d is None or getattr(arr2d, "ndim", 1) != 2 or arr2d.size == 0:
                                continue

                            write_window_tiff(
                                arr=arr2d,
                                transform=vmeta.get("transform"),
                                crs=vmeta.get("raster_crs"),
                                dtype=vmeta.get("dtype", "float32"),
                                nodata=vmeta.get("nodata"),
                                path=tile_path,
                            )

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
                "project_crs": proj_crs,
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
            for p in feats:
                val_cols = [c for c in work.columns if c.startswith(f"{p}_") and "_b" in c]
                vcol = val_cols[0] if val_cols else None
                vals = work[vcol].tolist() if vcol else [None] * len(work)
                outputs[f"{gname}:{p}"] = om.write_raster_demo(vals, work.lat, work.lon, gname, p)
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

            reducer = get_reducer(spec.get("default_reducer", "mean"))
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
