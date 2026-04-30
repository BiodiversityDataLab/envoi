# biodata-enricher

Enrich a Pandas DataFrame of sample points (`id`, `lat`, `lon`, optional `date`) with environmental data from Google Earth Engine or local GeoTIFFs.

Outputs **tabular** data (Parquet/CSV) with summary statistics and QC columns, or **raster** tiles (GeoTIFF) — plus sidecar metadata JSON.

---

## Install

```bash
pip install biodata-enricher
```

## Quick start

```python
import pandas as pd
from biodata import extract

df = pd.read_csv("data/points_sample.csv")

# Single output — tabular stats
outputs = extract(df, {
    "batch_id": "terrain",
    "datasets": ["dem_aster"],
    "settings": {
        "output_type": "tabular",
        "statistics": ["mean", "std"],
        "window_size_m": 200,
    },
}, output_dir="out")

print(outputs["terrain"])      # -> out/terrain.parquet
print(outputs["terrain_qc"])   # -> out/terrain_qc.parquet
# Metadata: out/terrain_metadata.json
```

## Multiple outputs

Pass a list to process several configurations in one call:

```python
outputs = extract(df, [
    {
        "batch_id": "terrain_stats",
        "datasets": ["dem_aster"],
        "settings": {"output_type": "tabular", "statistics": ["mean", "std"], "window_size_m": 200},
    },
    {
        "batch_id": "terrain_tiles",
        "datasets": ["dem_aster"],
        "settings": {"output_type": "raster", "window_size_m": 200, "resample_m": 10},
    },
], output_dir="out")
```

## Adding custom datasets

Datasets not in the built-in catalog can be registered once with `update_catalog()` and are then available in all subsequent `extract()` calls:

```python
from biodata import extract, update_catalog

# Register a local raster
update_catalog({"datasets": {
    "my_dem": {"data_source": "local", "path": "data/dem.tif"},
}})

# Or load from a YAML file
update_catalog("my_catalog.yml")

# Now use the dataset normally
outputs = extract(df, {"batch_id": "terrain", "datasets": ["my_dem"], ...})
```

See `examples/run.yml` for an example catalog YAML structure.

## Output kinds

### Tabular (`output_type: "tabular"`)
Produces Parquet (default) or CSV with reducer columns and a separate QC file.

Available reducers: `mean`, `median`, `std`, `var`, `min`, `max`, `q10`, `q90`, `count`, `sum`, `point`

The `point` reducer samples the exact pixel at each coordinate (no window).

### Raster (`output_type: "raster"`)
Exports GeoTIFF tiles per point, cropped to the specified window.

Option: `resample_m` resamples all tiles to a consistent resolution (e.g. for CNN input).

## Output columns

**Stats file**: `dem_aster_mean_b200`, `dem_aster_std_b200`, ...

**QC file**: `dem_aster_in_extent_b200`, `dem_aster_n_pixels_b200`, `dem_aster_had_nodata_b200`, `dem_aster_coverage_pct_b200`

## Notes

- Windows are in meters, using each point's UTM zone for global coverage.
- CRS and resolution are detected from the data source — no manual configuration needed.
- Low coverage is flagged in QC columns, not fatal. Filter by `*_coverage_pct` as needed.
