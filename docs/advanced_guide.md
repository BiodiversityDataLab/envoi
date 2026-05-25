# Advanced usage

## Multiple outputs in one call

Pass a list of configs to produce several outputs from one `extract()` call:

```python
outputs = extract(sample_points, [
    {
        "batch_id": "terrain_stats",
        "datasets": ["dem_copernicus_glo30"],
        "settings": {
            "output_type": "tabular",
            "statistics": ["mean", "std"],
            "window_size_m": 200,
        },
    },
    {
        "batch_id": "terrain_tiles",
        "datasets": ["dem_copernicus_glo30"],
        "settings": {
            "output_type": "raster",
            "window_size_m": 200,
            "resample_m": 10,
        },
    },
])
```

## Date-aware extraction

If your DataFrame has an `eventDate` column, envoi uses it when querying time-varying Earth Engine ImageCollections:

```python
sample_points = pd.DataFrame({
    "gbifID":     ["a", "b"],
    "decimalLatitude":  [59.85, 59.86],
    "decimalLongitude": [17.63, 17.64],
    "eventDate":        ["2022-06-15", "2023-08-01"],
})

extract(sample_points, {
    "batch_id": "ndvi",
    "datasets": ["ndvi_landsat_annual"],
    "settings": {"output_type": "tabular", "statistics": ["mean"], "window_size_m": 200},
})
```

For each point, envoi selects a single image from the collection. How that image is chosen is controlled per-dataset by `collection_date_policy` in the catalog:

- `"nearest"` (default) — the image whose start timestamp is closest to the point's date.
- `"contains"` — the image whose time range contains the point's date. Use this for interval-based collections (e.g. monthly aggregates) where "contains" is more meaningful than "nearest".

Dates outside the collection's range are silently clamped to the nearest boundary and recorded in the metadata sidecar — they do not raise.

Without an `eventDate` column, envoi falls back to the most recent image in each collection and emits a warning so you know it happened.

## Mixing categorical and continuous datasets

When a run combines continuous datasets (e.g. elevation, climate) with categorical ones (e.g. land cover), pass a typed dict for `statistics` so each type gets the appropriate reducers:

```python
extract(sample_points, {
    "batch_id": "mixed",
    "datasets": ["dem_copernicus_glo30", "lulc_worldcover_2021"],
    "settings": {
        "output_type": "tabular",
        "statistics": {
            "continuous":  ["mean", "std", "q10", "q90"],
            "categorical": ["mode", "class_fraction"],
        },
        "window_size_m": 200,
    },
})
```

Each dataset's `data_type` in the catalog (`continuous` or `categorical`) decides which list applies. A reducer valid for both types (e.g. `mode`) can appear in both lists. Datasets without an explicit `data_type` default to `continuous`. A flat list (the original form) still works and applies to every dataset.

## Selecting bands per call

The catalog defines each dataset's default bands. To override them for a single call without re-registering the catalog, replace a string entry in `datasets` with a single-key dict whose value is the band list:

```python
extract(sample_points, {
    "batch_id": "satellite",
    "datasets": [
        "dem_copernicus_glo30",                # catalog defaults
        {"sr_landsat_annual": ["B4", "B5"]},   # narrow bands for this run only
        {"dem_copernicus_glo30": ["DEM", "slope"]},       # mix source + derived names
    ],
    "settings": {"output_type": "tabular", "statistics": ["mean"], "window_size_m": 200},
})
```

Names recognised as derived bands (currently `slope` and `aspect`, computed from the first band of the dataset) are split out automatically — no separate key needed. Derived bands are only supported for Earth Engine datasets; supplying one for a local raster raises `ValueError`.

A one-element list (`{"sr_landsat_annual": ["B4"]}`) keeps the multi-band column naming (`sr_landsat_annual_B4_mean_200m`); use the catalog default if you want the bare single-band form.

> **Band identifiers.** Earth Engine datasets identify bands by *string* names (`"B4"`, `"DEM"`, `"slope"`) because GEE exposes named bands. Local GeoTIFFs identify bands by *integer* index (`1`, `2`, `[1, 2, 3]`) because rasterio reads bands positionally. So a GEE catalog entry might set `bands: ["DEM"]` and a local one `bands: 1`, and a per-call override follows the same convention as the dataset it targets.

## Custom datasets

Datasets not in the built-in catalog can be registered once with `update_catalog()`. Registered datasets are then available in every subsequent `extract()` call:

```python
from envoi import update_catalog

# From a dict — quick for one-off local rasters.
update_catalog({
    "datasets": {
        "my_dem": {"data_source": "local", "path": "data/dem.tif"},
    },
})

# From a YAML file — better for multi-dataset projects under version control.
update_catalog("my_catalog.yml")
```

For the full catalog schema, see the commented reference block at the top of [src/envoi/configs/ee_catalog.yml](../src/envoi/configs/ee_catalog.yml) — it's a copy-paste starting point for adding a new entry.
