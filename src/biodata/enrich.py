from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
import yaml
import numpy as np

from .adapters.local_raster import LocalRasterAdapter
from .reducers import get_reducer
from .output import OutputManager
from .config import load_catalog
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
    catalog: str | Path = "configs/catalog.yml",
    groups: str | dict | None = None,  # group mode
    out_dir: str | Path = "out",
    window_m: int = 500,
    temporal: str = "nearest_month",  # reserved for future sources
    cache_dir: str = "~/.biodata_cache",
    out_path: str | Path | None = None,  # reserved for future sources
) -> Dict[str, Path]:
    print("enrich catalog type:", type(catalog), catalog)  # Debug print
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

    catalog_dict = load_catalog(catalog)
    cat = catalog_dict["datasets"]

    om = OutputManager(out_dir)
    outputs: Dict[str, Path] = {}

    # -------- Mode B: groups specified --------
    if groups is not None:
        gcfg = _load_yaml(groups)
        groups_list = gcfg.get("groups", [])
        # policy defaults (keep old behavior if not provided)
        min_cov = gcfg.get("min_coverage_pct", 80)
        proj_crs = gcfg.get("project_crs", "EPSG:3006")

        for idx, g in enumerate(groups_list):
            gname = g.get("name", f"group{idx+1}")
            feats: List[str] = g.get("features") or g.get("predictors", [])
            out_cfg = g.get("output", {}) or {}
            kind = out_cfg.get("kind", "tabular")
            stats = g.get("summary_statistics") or out_cfg.get("reducers")  # list[str] | None
            buffers = g.get("buffer_sizes")
            if not buffers:
                buffers = [out_cfg.get("window_m", window_m)]

            work = df.copy()
            provenance: Dict[str, Any] = {}
            coverage_backlog: Dict[str, Dict[str, Dict[str, int]]] = {}

            for p in feats:
                if p not in cat:
                    raise KeyError(f"Feature '{p}' not found in catalog {catalog}")
                spec = cat[p]
                if spec.get("source") == "local_raster":
                    adapter = LocalRasterAdapter(spec)
                    coverage_backlog[p] = {}
                    # Loop over all requested buffer sizes
                    for buf in buffers:
                        vals_list: List[np.ndarray] = []
                        meta_list: List[Dict[str, Any]] = []
                        # One pass: fetch values + meta for each row (use buf!)
                        for lat, lon in zip(work.lat, work.lon):
                            arr, meta = adapter.fetch_values(lat, lon, buf, return_meta=True)
                            arr = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
                            vals_list.append(arr)
                            meta_list.append(meta)
                        # --- reducers → columns (buffer-suffixed) ---
                        created_value_cols: list[str] = []
                        if stats:
                            for rname in stats:
                                reducer = get_reducer(rname)
                                col = f"{p}_{rname}_b{buf}"
                                work[col] = [(reducer(v) if v.size else None) for v in vals_list]
                                created_value_cols.append(col)
                        else:
                            default_r = spec.get("default_reducer", "mean")
                            reducer = get_reducer(default_r)
                            col = f"{p}_{default_r}_b{buf}"
                            work[col] = [(reducer(v) if v.size else None) for v in vals_list]
                            created_value_cols.append(col)
                        # --- QA columns (prefixed + buffer-suffixed) ---
                        qc_df = compute_qc_flags(meta_list, min_coverage_pct=min_cov)
                        qc_df = qc_df.add_prefix(f"{p}_").add_suffix(f"_b{buf}")
                        work = pd.concat(
                            [work.reset_index(drop=True), qc_df.reset_index(drop=True)], axis=1
                        )
                        # ---- Back-compat aliases when only one buffer is used ----
                        if len(buffers) == 1:
                            # alias value columns (drop _b{buf})
                            if stats:
                                for rname in stats:
                                    src = f"{p}_{rname}_b{buf}"
                                    dst = f"{p}_{rname}"
                                    if src in work.columns and dst not in work.columns:
                                        work[dst] = work[src]
                            else:
                                default_r = spec.get("default_reducer", "mean")
                                src = f"{p}_{default_r}_b{buf}"
                                dst = f"{p}_{default_r}"
                                if src in work.columns and dst not in work.columns:
                                    work[dst] = work[src]
                            # alias QA columns (drop _b{buf})
                            for qc_col in ["in_extent", "n_pixels", "had_nodata", "coverage_pct"]:
                                src = f"{p}_{qc_col}_b{buf}"
                                dst = f"{p}_{qc_col}"
                                if src in work.columns and dst not in work.columns:
                                    work[dst] = work[src]
                        # Coverage summary for this feature + buffer
                        cov = qc_df[f"{p}_coverage_pct_b{buf}"].fillna(0)
                        coverage_backlog[p][str(buf)] = {
                            "n_zero": int((cov == 0).sum()),
                            "n_partial": int(((cov > 0) & (cov < 100)).sum()),
                            "n_full": int((cov == 100).sum()),
                            "total": int(cov.shape[0]),
                        }
                    provenance[p] = build_provenance(
                        spec,
                        stats or [spec.get("default_reducer", "mean")],
                        buffers,  # list now
                        temporal,
                        meta_list[0] if meta_list else {},
                    )
                else:
                    # Placeholder for non-local adapters in MVP
                    for buf in buffers:
                        work[f"{p}_value_b{buf}"] = None
            if kind == "tabular":
                outputs[gname] = write_group_parquet(
                    work,
                    gname,
                    {"provenance": provenance, "coverage_backlog": coverage_backlog},
                    {
                        **gcfg,
                        "project_crs": proj_crs,
                        "min_coverage_pct": min_cov,
                        "summary_statistics": stats,
                        "buffer_sizes": buffers,
                        "out_dir": out_dir,
                    },
                )
            elif kind == "raster":
                for p in feats:
                    val_cols = [c for c in work.columns if c.startswith(f"{p}_") and "_b" in c]
                    vcol = val_cols[0] if val_cols else None
                    vals = work[vcol].tolist() if vcol else [None] * len(work)
                    outputs[f"{gname}:{p}"] = om.write_raster_demo(
                        vals, work.lat, work.lon, gname, p
                    )
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
            if spec.get("source") == "local_raster":
                adapter = LocalRasterAdapter(spec)
                reducer = get_reducer(spec.get("default_reducer", "mean"))
                out[p] = [
                    reducer(adapter.fetch_values(lat, lon, window_m))
                    for lat, lon in zip(out.lat, out.lon)
                ]
            else:
                out[p] = None
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
