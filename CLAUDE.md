# CLAUDE.md — EDDP / biodata-enricher

This file gives you context on the project vision, architecture, and decisions.
Update it as the project evolves.

---

## What this project is

**biodata-enricher (EDDP)** is a Python package that enriches geographic point data
with environmental datasets. The input is a table of sample points (`id`, `lat`, `lon`,
optionally `date`); the output is either that same table with appended environmental columns 
or images of environmental datasets, ready for spatial ecological modeling or similar analyses.

The primary use case is ecological research where you have field sample
locations and want to attach climate, terrain, vegetation, or other environmental
variables to each point.

---

## Core vision

> Access environmental data from Google Earth Engine **and/or** local rasters through a
> single, unified, easy-to-use interface — so results from both sources are directly comparable.

Key priorities:
**GEE access first** — fast, flexible, server-side queries without pre-downloading data
**Local raster parity** — same interface, same output format, so you can add rasters that
are not available on GEE and have the same processing of the data
**Flexible for different data sources** - should have a general code structure, with specific 
adapters for the different data sources (e.g. local data, Google Earth Engine)
**User friendly interface** - since many users of this Python package will not have much programming
experiance, the user interface should be as intuitive as possible.
**Global coverage** - should be able to correctly download images and calculate statistics globally
---

## Architecture

```
extract(df, cfg)               ← main entry point
    ↓
catalog.yml                   ← defines available datasets (source + path required)
    ↓
Adapter (per dataset)
    ├── GeeRasterAdapter      ← queries GEE directly, parallel via ThreadPoolExecutor
    └── LocalRasterAdapter    ← reads GeoTIFF via rasterio, dynamic UTM per point
```

## API

The primary interface is `extract(df, cfg)` where `cfg` is a dict (single output)
or list of dicts (multiple outputs):

```python
extract(df, {
    "run_id": "terrain",
    "datasets": ["dem_local"],
    "settings": {
        "output_type": "tabular",          # "tabular" or "raster"
        "statistics": ["mean", "std"],
        "window_size_m": 200,
        "output_format": "parquet",        # "parquet" or "csv"
        "resample_m": 10,           # optional, for CNN-ready tiles
        "min_coverage_pct": 80,     # QC threshold
    },
})
```

## Catalog design

The catalog (`configs/catalog.yml`) is the source of available datasets.
Only `source` and `path` are required — everything else is auto-detected or optional.

```yaml
dem_aster:
  source: earth_engine
  path: projects/sat-io/open-datasets/ASTER/GDEM   # asset type auto-detected via ee.data.getAsset()

dem_local:
  source: local
  path: data/for_testing/dem/TG4NHB-dem.tif
  band: 1                                           # optional, defaults to band 1
```

**GEE auto-detection:** asset type (IMAGE vs IMAGE_COLLECTION) is resolved via
`ee.data.getAsset()` — no `asset_type` key needed in catalog.

**Local auto-detection:** CRS, resolution, nodata, and band count are read from the
file via rasterio (`_inspect_raster()`).

**`feature_spec` block** (optional, GEE only): for advanced config like
cloud masking, collection reducer, and derived bands:

```yaml
sen2_ndvi:
  source: earth_engine
  path: COPERNICUS/S2_SR_HARMONIZED
  feature_spec:
    cloud_pct_max: 20
    cloud_mask: s2
    derived_band: NDVI
```

**Automatic date handling for ImageCollections:** when the input DataFrame
has a `date` column, the adapter fetches the collection's available timestamps
and selects the single nearest image to each point's date. Out-of-range dates
are clamped to the closest boundary. When no `date` column is provided, the
most recent image is used. Date decisions are recorded in the output metadata.

---

## Files overview

```
src/biodata/
    extract.py           ← main entry point
    config.py           ← catalog loading + local raster auto-detection
    metadata.py         ← per-feature metadata + sidecar JSON writer
    auth.py             ← GEE authentication from credentials/ee_credentials.json
    reducers.py         ← Python-side reducer registry (mean, std, quantiles, ...)
    output.py           ← parquet/csv writing
    qc.py               ← coverage QC flags
    adapters/
        base.py         ← BaseAdapter
        gee_adapter.py  ← GeeRasterAdapter + all image-building utilities
        local_adapter.py← LocalRasterAdapter

configs/
    ee_catalog.yml      ← GEE dataset registry
    local_catalog.yml   ← local raster dataset registry
    run.yml             ← example single-output config

credentials/
    ee_credentials.json ← GEE service account key (gitignored)
```

---
