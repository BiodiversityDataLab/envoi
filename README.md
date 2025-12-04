# biodata-enricher — Python usage

Enrich a Pandas DataFrame of points (`id`, `lat`, `lon`, optional `date`) with features sampled from local GeoTIFF rasters.  
Outputs model-ready **Parquet** plus **QA** columns and **provenance** (metadata JSON).

---

## Install

```bash
make install
make test
```

## Sampling policy

Current behavior includes all pixels in the pixel-aligned window. We plan to expose a sampling_policy (e.g., `centroid`, `all_touched`, `fractional`) to control edge inclusion.

## Recommended: groups mode (one call → multiple features + QA + metadata)

```python
import pandas as pd
from biodata.enrich import enrich

df = pd.read_csv("data/points_sample.csv")

# One run, two groups, multiple reducers
cfg = {
    "groups": [
        {
            "name": "dem_100m",
            # catalog keys (can be local rasters or GEE-backed)
            "predictors": ["dem_mini"],
            "output": {
                "kind": "tabular",
                # any subset of the registered reducers:
                # mean, median, std, var, min, max, q10, q90, count, sum
                "reducers": ["mean", "std", "q10", "q90"],
                "window_m": 100,
            },
        },
        {
            "name": "site_stats",
            "predictors": ["dem_mini"],
            "output": {
                "kind": "tabular",
                "reducers": ["mean", "min", "max", "count"],
                "window_m": 500,
            },
        },
    ],
    "min_coverage_pct": 80,          # QA threshold
    "project_crs": "EPSG:3006",      # working CRS for meter-based windows
}

outputs = enrich(
    df,
    groups=cfg,
    catalog="configs/catalog.yml",          # base catalog shipped with the library
    extra_catalog="configs/local_catalog.yml",  # optional: user’s own extra sources
    out_dir="out",
)

# Parquet paths (per group)
print(outputs["dem_100m"])   # -> out/dem_100m.parquet
print(outputs["site_stats"]) # -> out/site_stats.parquet
# Each has a matching metadata JSON:
# out/dem_100m_metadata.json, out/site_stats_metadata.json
```

## What you’ll see in the Parquet

Reducer columns: `dem_mini_mean`, `dem_mini_std`, `dem_mini_q10`, `dem_mini_q90`

QA columns: `dem_mini_in_extent`, `dem_mini_n_pixels`, `dem_mini_had_nodata`, `dem_mini_coverage_pct`

## Catalog (tell the library where rasters live)

`configs/catalog.yml`:
```yaml
datasets:
  dem_mini:
    type: raster
    source: local_raster
    path: tests/data/mini_dem.tif   # any GeoTIFF with a valid CRS
    crs: EPSG:4326
    default_reducer: mean
```
Add more predictors by adding more entries to `datasets:` and listing them in your `predictors`/`features`.

## Re-run previous runs (History)

Every `enrich` run writes a manifest with all the knobs you used (input CSV, catalog, groups/predictors, reducers, window, etc.):

- Latest run: `out/last_run.json`
- Archive of all runs: `out/runs/run_<YYYYMMDD_HHMMSS>.json`

### CLI

```bash
# Run once (writes out/last_run.json and out/runs/run_*.json)
biodata enrich \
  --in data/points_sample.csv \
  --out out \
  --catalog configs/catalog.yml \
  --groups configs/run.yml

# Re-run the latest
biodata rerun --from out/last_run.json

# Re-run a specific past run
biodata rerun --from out/runs/run_20251113_121530.json
```
### Python
```python
from biodata.history import replay_last_run

outputs = replay_last_run()  # or replay_last_run("out/runs/run_20251113_121530.json")
print(outputs)
```

### Where do the TIFF files go?
```
out/tiles/<group>/<feature>/b<buffer>/id<point_id>.tif
# e.g. out/tiles/terrain_features/dem_elev/b100/id42.tif
```

## Notes & limits

- Windows are in meters using EPSG:3006 internally (robust for Sweden; reprojected to the raster CRS when sampling).

- Works with GeoTIFFs readable by rasterio and having a valid CRS; assumes numeric single-band by default.

- Low coverage is flagged, not fatal; filter by *_coverage_pct as needed.
