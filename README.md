# envoi

[![PyPI version](https://img.shields.io/pypi/v/envoi-geospatial)](https://pypi.org/project/envoi-geospatial/)
[![Python versions](https://img.shields.io/pypi/pyversions/envoi-geospatial)](https://pypi.org/project/envoi-geospatial/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automated feature extraction from environmental data sources for ecological and spatial analysis.

---

## Table of contents

- [Install](#install)
- [Earth Engine setup](#earth-engine-setup)
- [Quick start](#quick-start)
  - [Walkthrough](#walkthrough)
- [Outputs](#outputs)
  - [Tabular](#tabular)
  - [Raster](#raster)
- [Advanced usage](#advanced-usage)
- [Reference](#reference)
  - [Built-in datasets](#built-in-datasets)
  - [Notes](#notes)
- [How to cite](#how-to-cite)
- [Contributors](#contributors)
- [Project links](#project-links)

---

## envoi - ENvironmental Variables for Observational Instances

Ecological and spatial models need environmental variables attached to field sample points — climate, terrain, land cover, vegetation indices. The usual workflow involves stitching together one-off scripts for each data source (Earth Engine for satellite data, rasterio for local files, ad-hoc projections to get distances right), and the outputs rarely line up.

envoi exposes a single `extract(df, config)` call that runs against both Google Earth Engine and local GeoTIFFs and returns the same shape of output. No pre-downloading, sensible defaults for users who'd rather not think about CRS or UTM zones, and the same reducers and QC columns across data sources so results are directly comparable.

envoi is developed at the [Biodiversity Data Lab](https://biodiversity.se/) at Uppsala University.

---

## Install

```bash
pip install envoi-geospatial
```

Requires Python 3.10 or newer.

---

## Earth Engine setup

Datasets that come from Google Earth Engine (most of the built-in catalog — `dem_copernicus_glo30`, `ndvi_landsat_annual`, etc.) need a service account key from Google. If you only plan to use your own local rasters you can skip this section.

**Step 1 — get the key file.** In the [Google Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts), create a service account that has Earth Engine access and download its JSON key. You'll end up with a file like `my-project-1234-abcdef.json`. See the [official guide](https://developers.google.com/earth-engine/guides/service_account) for the full walkthrough.

**Step 2 — put the file somewhere envoi can find it.** Pick whichever of these is easiest:

- **In your project folder:** save it as `credentials/ee_credentials.json` next to your script or notebook. This is the simplest option if you only use Earth Engine for one project.
- **In your user folder:** save it as `~/.config/envoi/ee_credentials.json` (macOS/Linux) or `%APPDATA%\envoi\ee_credentials.json` (Windows). You may need to create the folder. Useful if you want the same key available to every project on your computer.
- **At a custom path via environment variable:** set `ENVOI_EE_CREDENTIALS` to the file's path. envoi checks this before the two folders above, so it's the right choice when the key lives outside the defaults, when you swap between several credential files, or in CI / Docker.
- **Anywhere else:** pass the path explicitly in your code before calling `extract()`:

  ```python
  from envoi import init_gee

  init_gee(credentials_path="/path/to/my-project-1234-abcdef.json")
  ```

---

## Quick start

Pass any DataFrame with an identifier column and a latitude/longitude pair. By default envoi expects the GBIF / Darwin Core names `gbifID`, `decimalLatitude`, `decimalLongitude` and treats coordinates as WGS84 (EPSG:4326). If yours differ, override on the call with `id_column=`, `latitude_column=`, `longitude_column=`, and `input_crs=` (e.g. `"EPSG:32634"`) — envoi reprojects to WGS84 internally. An optional `eventDate` column (or any column passed via `date_column=`) enables [date-aware extraction](docs/advanced_usage.md#date-aware-extraction).

```python
import pandas as pd
from envoi import extract

sample_points = pd.DataFrame({
    "gbifID":     ["a", "b", "c"],
    "decimalLatitude":  [59.85, 59.86, 59.87],
    "decimalLongitude": [17.63, 17.64, 17.65],
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
#   outputs/terrain.csv               ← reducer columns
#   outputs/terrain_qc.csv            ← per-point coverage / nodata flags
#   outputs/terrain_metadata.json     ← per-run dataset metadata
```

Override the output location with `extract(df, config, output_dir="my_dir")`.

The same config can also live in a YAML file — see [examples/run.yml](examples/run.yml) for a runnable template.

### Walkthrough

For a guided end-to-end tutorial — tabular and raster extraction, local rasters, multi-dataset runs, date-aware extraction, and catalog discovery — see the [walkthrough notebook](examples/walkthrough.ipynb).

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
| `"csv"`       | `outputs/<batch_id>.csv` (default)                       |
| `"parquet"`   | `outputs/<batch_id>.parquet`                             |
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

**With `resample_m`,** every tile is reprojected to the point's UTM zone at exactly `resample_m` meters per pixel, on a grid snapped to that resolution. All tiles end up the same size (`round(window_size_m / resample_m)` pixels per side) and are spatially aligned across data sources — useful when feeding tiles to a CNN that expects a fixed input size or when comparing GEE and local rasters pixel-for-pixel.

---

## Advanced usage

Multiple outputs in one call, date-aware extraction, mixing categorical and continuous datasets, per-call band selection, multiple window sizes, and custom dataset registration are covered in [docs/advanced_usage.md](docs/advanced_usage.md). A starter custom catalog (local raster and Earth Engine entries) lives at [examples/catalog.yml](examples/catalog.yml).

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

`list_datasets()` both prints the listing and returns the same data as a list (of strings for the default call, of dicts for `"info"` / `"full"`), so you can keep using it programmatically.

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

## How to cite

A paper describing envoi is currently in preparation. In the meantime, please cite the software directly:

> Baggström, A., Nyström, J., & Andermann, T. (*in prep.*). envoi: automated environmental feature extraction for ecological analysis. Retrieved from https://github.com/BiodiversityDataLab/envoi

This entry will be updated with a DOI and full citation when the paper is published.

---

## Contributors

**Primary authors and maintainers** — Adrian Baggström, Jakob Nyström.

**Past contributors** — Miguel Redondo at [NBIS](https://nbis.se); Shaheryar, Thant Zin Bo, and Per Vincent Ankarbåge (Uppsala University Data Science MSc students).

**Acknowledgements** — Tobias Andermann (Conceptualization and PhD supervision for A.B. and J.N.). A.B., J.N., and T.A. received financial support from the SciLifeLab & Wallenberg Data Driven Life Science Program (grant: KAW 2020.0239) and from the Swedish Research Council (2023-05366). We are grateful to the maintainers of Google Earth Engine, rasterio, geopandas, and pyproj.


---

## Project links

- **License** — [MIT](LICENSE)
- **Contributing** — [CONTRIBUTING.md](CONTRIBUTING.md)
- **Issues / bug reports** — [github.com/BiodiversityDataLab/envoi/issues](https://github.com/BiodiversityDataLab/envoi/issues)
- **Repository** — [github.com/BiodiversityDataLab/envoi](https://github.com/BiodiversityDataLab/envoi)

---

*Take these points: cross sky and stone;  
return them clothed, no longer alone.*