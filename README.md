# biodata-enricher

Enrich a Pandas DataFrame of sample points (`id`, `lat`, `lon`, optional `date`) with environmental data from local GeoTIFFs or Google Earth Engine.

Outputs **tabular** data (Parquet/CSV) with summary statistics and QC columns, or **raster** tiles (GeoTIFF) — plus sidecar metadata JSON.

---

## Install

```bash
make install
make test
```

## Quick start

```python
import pandas as pd
from biodata.enrich import enrich

df = pd.read_csv("data/points_sample.csv")

# Single output — tabular stats
outputs = enrich(df, {
    "name": "terrain",
    "predictors": ["dem_local"],
    "output": {
        "kind": "tabular",
        "reducers": ["mean", "std"],
        "window_m": 200,
    },
}, catalog="configs/catalog.yml", out_dir="out")

print(outputs["terrain"])      # -> out/terrain.parquet
print(outputs["terrain_qc"])   # -> out/terrain_qc.parquet
# Metadata: out/terrain_metadata.json
```

## Multiple outputs

Pass a list to process several configurations in one call:

```python
outputs = enrich(df, [
    {
        "name": "terrain_stats",
        "predictors": ["dem_local"],
        "output": {"kind": "tabular", "reducers": ["mean", "std"], "window_m": 200},
    },
    {
        "name": "terrain_tiles",
        "predictors": ["dem_local"],
        "output": {"kind": "raster", "window_m": 200, "resample_m": 10},
    },
], catalog="configs/catalog.yml", out_dir="out")
```

## Output kinds

### Tabular (`kind: "tabular"`)
Produces Parquet (default) or CSV with reducer columns and a separate QC file.

Available reducers: `mean`, `median`, `std`, `var`, `min`, `max`, `q10`, `q90`, `count`, `sum`, `point`

The `point` reducer samples the exact pixel at each coordinate (no window).

Options: `format: "csv"` to write CSV instead of Parquet.

### Raster (`kind: "raster"`)
Exports GeoTIFF tiles per point, cropped to the specified window.

Option: `resample_m` resamples all tiles to a consistent resolution (e.g. for CNN input).

## Output columns

**Stats file**: `dem_local_mean_b200`, `dem_local_std_b200`, ...

**QC file**: `dem_local_in_extent_b200`, `dem_local_n_pixels_b200`, `dem_local_had_nodata_b200`, `dem_local_coverage_pct_b200`

## Catalog

The catalog tells the library where data lives. `configs/catalog.yml`:

```yaml
datasets:
  dem_local:
    source: local
    path: data/dem/my_dem.tif

  dem_aster:
    source: earth_engine
    path: NASA/ASTER_GED/AG100_003
    bands: [elevation]
```

Sources: `local` (GeoTIFF via rasterio) or `earth_engine` (Google Earth Engine).

## Notes

- Windows are in meters, using each point's UTM zone for global coverage.
- Works with any GeoTIFF readable by rasterio with a valid CRS.
- CRS and resolution are detected from the data source — no manual configuration needed.
- Low coverage is flagged in QC columns, not fatal. Filter by `*_coverage_pct` as needed.
