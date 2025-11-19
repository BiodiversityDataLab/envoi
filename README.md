# biodata-enricher — Python usage

Enrich a Pandas DataFrame of points (`id`, `lat`, `lon`, optional `date`) with features sampled from local GeoTIFF rasters.  
Outputs model-ready **Parquet** plus **QA** columns and **provenance** (metadata JSON).

---

## Install

```bash
make install
make test
```

## Recommended: groups mode (one call → multiple features + QA + metadata)
```python
import pandas as pd
from biodata.enrich import enrich

df = pd.read_csv("data/points_sample.csv")

cfg = {
  "groups": [{
    "name": "dem_100m",
    "predictors": ["dem_mini"],             # or "features": [...]
    "output": {
      "kind": "tabular",
      "reducers": ["mean", "std", "q10", "q90"],
      "window_m": 100
    }
  }],
  "min_coverage_pct": 80,                   # QA threshold
  "project_crs": "EPSG:3006"
}

outputs = enrich(df, groups=cfg, catalog="configs/catalog.yml", out_dir="out")
# Parquet path:
print(outputs["dem_100m"])                  # -> out/dem_100m.parquet
# Metadata JSON sits next to it:
# out/dem_100m_metadata.json

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


## Notes & limits

- Windows are in meters using EPSG:3006 internally (robust for Sweden; reprojected to the raster CRS when sampling).

- Works with GeoTIFFs readable by rasterio and having a valid CRS; assumes numeric single-band by default.

- Low coverage is flagged, not fatal; filter by *_coverage_pct as needed.
