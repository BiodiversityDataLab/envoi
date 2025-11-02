# src/biodata/provenance.py (new)
from __future__ import annotations


def build_provenance(ds: dict, reducers, window_m, temporal, meta) -> dict:
    return {
        "dataset": ds.get("name") or ds,
        "source": ds.get("source"),
        "type": ds.get("type"),
        "crs": ds.get("crs"),
        "path": ds.get("path"),
        "license": ds.get("license", "unknown"),
        "reducers": reducers,
        "window_m": window_m,
        "temporal": temporal,
        "adapter_meta": {
            k: meta.get(k) for k in ("raster_crs", "n_pixels", "coverage_pct", "had_nodata")
        },
    }
