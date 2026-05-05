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

## Mixing categorical and continuous datasets

When a run includes both continuous (e.g. elevation, climate) and categorical (e.g. land cover) datasets, use a typed dict for `statistics` to assign the right reducers to each type:

```python
outputs = extract(df, {
    "batch_id": "mixed",
    "datasets": ["dem_glo30", "lulc_esa_worldcover_2021"],
    "settings": {
        "output_type": "tabular",
        "statistics": {
            "continuous":  ["mean", "std", "q10", "q90"],
            "categorical": ["mode", "count"],
        },
        "window_size_m": 200,
    },
})
```

Each dataset's `data_type` field in the catalog controls which list is used. A reducer that makes sense for both types (e.g. `mode`) can appear in both lists. Datasets without a `data_type` default to `continuous`. A flat list (the existing form) still works and applies to all datasets — no changes needed for single-type runs.

## Selecting bands per call

The catalog defines each dataset's default bands. To override them for a single `extract()` call without re-registering the catalog, replace a string entry in `datasets` with a single-key dict whose value is the unified band list:

```python
outputs = extract(df, {
    "batch_id": "satellite",
    "datasets": [
        "dem_glo30",                           # catalog defaults
        {"sentinel2": ["B4", "B8"]},           # narrow bands for this run only
        {"dem_aster": ["DEM", "slope"]},       # mix source + derived names
    ],
    "settings": {"output_type": "tabular", "statistics": ["mean"], "window_size_m": 200},
})
```

Names recognised as derived bands (currently `slope` and `aspect`, computed from the dataset's first band) are split out automatically — you don't need a separate key for them. Derived bands are only supported for Earth Engine datasets; supplying one for a local raster raises a `ValueError`.

A one-element list (`{"sentinel2": ["B4"]}`) keeps the multi-band column naming (`sentinel2_B4_mean_<window>m`); use the catalog if you want the bare single-band form (`sentinel2_mean_<window>m`).

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
