# envoi

[![PyPI version](https://img.shields.io/pypi/v/envoi)](https://pypi.org/project/envoi/)
[![Python versions](https://img.shields.io/pypi/pyversions/envoi)](https://pypi.org/project/envoi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Enrich a pandas DataFrame of sample points with environmental data from Google Earth Engine and/or local GeoTIFFs — one unified interface, identical output shape across data sources.

---

## Table of contents

- [Why envoi?](#why-envoi)
- [Install](#install)
- [Earth Engine setup](#earth-engine-setup)
- [Quick start](#quick-start)
- [Outputs](#outputs)
  - [Tabular](#tabular)
  - [Raster](#raster)
- [Advanced](#advanced)
  - [Multiple outputs in one call](#multiple-outputs-in-one-call)
  - [Date-aware extraction](#date-aware-extraction)
  - [Mixing categorical and continuous datasets](#mixing-categorical-and-continuous-datasets)
  - [Selecting bands per call](#selecting-bands-per-call)
  - [Custom datasets](#custom-datasets)
- [Reference](#reference)
  - [Built-in datasets](#built-in-datasets)
  - [Notes](#notes)
- [Project links](#project-links)

---

## envoi - Environmental variables for observational instances

Ecological and spatial models need environmental variables attached to field sample points — climate, terrain, land cover, vegetation indices. The usual workflow involves stitching together one-off scripts for each data source (Earth Engine for satellite data, rasterio for local files, ad-hoc projections to get distances right), and the outputs rarely line up.

envoi exposes a single `extract(df, config)` call that runs against both Google Earth Engine and local GeoTIFFs and returns the same shape of output. No pre-downloading, sensible defaults for users who'd rather not think about CRS or UTM zones, and the same reducers and QC columns across data sources so results are directly comparable.

---

## Install

```bash
pip install envoi
```

Requires Python 3.10 or newer.

---

## Earth Engine setup

Datasets that come from Google Earth Engine (most of the built-in catalog — `dem_copernicus_glo30`, `ndvi_landsat_annual`, etc.) need a service account key from Google. If you only plan to use your own local rasters you can skip this section.

**Step 1 — get the key file.** In the [Google Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts), create a service account that has Earth Engine access and download its JSON key. You'll end up with a file like `my-project-1234-abcdef.json`. See the [official guide](https://developers.google.com/earth-engine/guides/service_account) for the full walkthrough.

**Step 2 — put the file somewhere envoi can find it.** Pick whichever of these is easiest:

- **In your project folder:** save it as `credentials/ee_credentials.json` next to your script or notebook. This is the simplest option if you only use Earth Engine for one project.
- **In your user folder:** save it as `~/.config/envoi/ee_credentials.json` (you may need to create the folder). Useful if you want the same key available to every project on your computer.
- **Anywhere else:** pass the path explicitly in your code before calling `extract()`:

  ```python
  from envoi import init_gee

  init_gee(credentials_path="/path/to/my-project-1234-abcdef.json")
  ```

> **Advanced:** if you're running envoi in CI or a Docker container, set the `ENVOI_EE_CREDENTIALS` environment variable to the path of the JSON file. This is checked before the two folders above.

---

## Quick start

```python
import pandas as pd
from envoi import extract

# Input: any DataFrame with id, lat, lon. Coordinates are assumed to be in
# WGS84 (EPSG:4326). If yours are in a different CRS, pass `input_crs=...`
# to extract() (e.g. `input_crs="EPSG:32634"`) and envoi will reproject them
# to WGS84 internally. A `date` column is optional — see "Date-aware
# extraction" below for how envoi uses it when present.
sample_points = pd.DataFrame({
    "id":  ["a", "b", "c"],
    "lat": [59.85, 59.86, 59.87],
    "lon": [17.63, 17.64, 17.65],
})

# Single output: mean and std of elevation in a 200 m window around each point.
outputs = extract(sample_points, {
    "batch_id": "terrain",
    "datasets": ["dem_copernicus_glo30"],
    "settings": {
        "output_type": "tabular",
        "statistics": ["mean", "std"],
        "window_size_m": 200,
    },
})

# Files land in outputs/ by default:
#   outputs/terrain.parquet           ← reducer columns
#   outputs/terrain_qc.parquet        ← per-point coverage / nodata flags
#   outputs/terrain_metadata.json     ← per-run dataset metadata
```

Override the output location with `extract(df, config, output_dir="my_dir")`.

---

## Outputs

### Tabular

`output_type: "tabular"` produces a table with one row per input point and one column per reducer × dataset × window. A separate QC file flags coverage and nodata.

**Reducer columns** look like: `dem_copernicus_glo30_mean_200m`, `dem_copernicus_glo30_std_200m`.

**QC columns** look like: `dem_copernicus_glo30_in_extent_200m`, `dem_copernicus_glo30_n_pixels_200m`, `dem_copernicus_glo30_had_nodata_200m`, `dem_copernicus_glo30_coverage_pct_200m`.

**Available reducers:**

- Core stats: `mean`, `median`, `min`, `max`, `sum`, `std`, `var`, `count`, `mode`
- Quantiles: `q05`, `q10`, `q25`, `q50`, `q75`, `q90`, `q95`
- Categorical: `class_count`, `class_fraction` (expanded per-class downstream)
- Special: `point` — samples the exact pixel at each coordinate (no window)

For the current authoritative list, run:

```python
from envoi import list_reducers
list_reducers()
```

**Output file format.** Set `output_file_format` in the settings block:

| Value         | Result                                                   |
| ------------- | -------------------------------------------------------- |
| `"parquet"`   | `outputs/<batch_id>.parquet` (default)                   |
| `"csv"`       | `outputs/<batch_id>.csv`                                 |
| `"dataframe"` | Returns the DataFrame in-memory, skips writing to disk.  |

```python
extract(sample_points, {
    "batch_id": "terrain",
    "datasets": ["dem_copernicus_glo30"],
    "settings": {
        "output_type": "tabular",
        "statistics": ["mean"],
        "window_size_m": 200,
        "output_file_format": "csv",
    },
})
```

### Raster

`output_type: "raster"` exports a GeoTIFF tile per point, cropped to the requested window:

```python
extract(sample_points, {
    "batch_id": "terrain_tiles",
    "datasets": ["dem_copernicus_glo30"],
    "settings": {
        "output_type": "raster",
        "window_size_m": 200,
        "resample_m": 10,   # optional — resample all tiles to a common resolution
    },
})
```

Tiles land at `outputs/<batch_id>/<dataset>/<id>-<dataset>.tif`.

**Without `resample_m`,** tiles are written in the source raster's native CRS at native resolution. The tile boundary snaps to the source pixel grid, so the actual extent is `window_size_m` rounded to whole pixels — any pixel touched by the requested window is included, and tile dimensions can vary slightly across points (especially for global datasets where pixel size depends on latitude).

**With `resample_m`,** every tile is reprojected to the point's UTM zone at exactly `resample_m` metres per pixel, on a grid snapped to that resolution. All tiles end up the same size (`round(window_size_m / resample_m)` pixels per side) and are spatially aligned across data sources — useful when feeding tiles to a CNN that expects a fixed input size or when comparing GEE and local rasters pixel-for-pixel.

---

## Advanced

### Multiple outputs in one call

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

### Date-aware extraction

If your DataFrame has a `date` column, envoi uses it when querying time-varying Earth Engine ImageCollections:

```python
sample_points = pd.DataFrame({
    "id":   ["a", "b"],
    "lat":  [59.85, 59.86],
    "lon":  [17.63, 17.64],
    "date": ["2022-06-15", "2023-08-01"],
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

Without a `date` column, envoi falls back to the most recent image in each collection and emits a warning so you know it happened.

### Mixing categorical and continuous datasets

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

### Selecting bands per call

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

### Custom datasets

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

For the full catalog schema, see the commented reference block at the top of [src/envoi/configs/ee_catalog.yml](src/envoi/configs/ee_catalog.yml) — it's a copy-paste starting point for adding a new entry.

---

## Reference

### Built-in datasets

envoi ships with a curated set of Earth Engine datasets spanning terrain, climate, land cover, satellite imagery, vegetation indices, and human-impact themes. Inspect what's available — including any datasets you've registered with `update_catalog()` — using `list_datasets()`:

```python
from envoi import list_datasets

list_datasets()          # just the names, one per line
list_datasets("info")    # name + description, citation, source URLs
list_datasets("full")    # the complete catalog entry for each dataset
```

`list_datasets()` both prints the listing and returns the same data as a list (of strings for `"names"`, of dicts for `"info"` / `"full"`), so you can keep using it programmatically.

A representative subset of the built-in catalog:

- **Terrain** — `dem_copernicus_glo30`
- **Climate** — `climate_worldclim_v1_bioclim`, `climate_era5_monthly`, `climate_terraclimate_monthly`
- **Land cover** — `lulc_worldcover_2021`, `lulc_copernicus_lc100`, `lulc_naturallands_2020`
- **Satellite imagery** — `sr_landsat_8day`, `sr_landsat_32day`, `sr_landsat_annual`
- **Vegetation / productivity** — `ndvi_landsat_annual`, `evi_landsat_annual`, `npp_modis_terra`, `agb_esa_cci`
- **Human impact** — `human_impact_index` plus eight `hii_driver_*` subcomponents
- **Embeddings** — `aef_satellite_embeddings`

The source, including descriptions, citations, and URLs for every entry, is [src/envoi/configs/ee_catalog.yml](src/envoi/configs/ee_catalog.yml).

### Notes

- **Input CRS.** Coordinates in the input DataFrame are assumed to be in **WGS84 (EPSG:4326)**. If yours are in a different CRS, pass `input_crs="EPSG:XXXX"` to `extract()` and envoi reprojects them to WGS84 before extraction.
- **Window units.** `window_size_m` is in meters. Each window is projected into the point's local UTM zone so distances are correct globally.
- **Data source CRS and resolution.** Both are detected automatically from each dataset — no manual configuration needed.
- **QC, not failure.** Low pixel coverage is flagged in QC columns rather than raising. Filter on `<dataset>_coverage_pct_<window>m` to drop unreliable rows downstream.

---

## Project links

- **License** — [MIT](LICENSE)
- **Contributing** — [CONTRIBUTORS.md](CONTRIBUTORS.md)
- **Issues / bug reports** — [github.com/BiodiversityDataLab/envoi/issues](https://github.com/BiodiversityDataLab/envoi/issues)
- **Repository** — [github.com/BiodiversityDataLab/envoi](https://github.com/BiodiversityDataLab/envoi)
