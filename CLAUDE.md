# CLAUDE.md — EDDP / biodata-enricher

This file gives you context on the project vision, architecture, and decisions.
Update it as the project evolves.

---

## What this project is

**biodata-enricher (EDDP)** is a Python package that enriches geographic point data
with environmental predictors. The input is a table of sample points (`id`, `lat`, `lon`,
optionally `date`); the output is either that same table with appended environmental columns 
or images of environmental predictors, ready for spatial ecological modeling or similar analyses.

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
enrich()                  ← main entry point (gateway to all adapters)
    ↓
catalog.yml               ← defines available datasets (source + path required)
    ↓
Adapter (per dataset)
    ├── GeeRasterAdapter  ← queries GEE directly, parallel via ThreadPoolExecutor
    └── LocalRasterAdapter ← reads GeoTIFF via rasterio, dynamic UTM per point
```

## Catalog design

The catalog (`configs/catalog.yml`) is the  source of available datasets.
Only `source` and `path` are required — everything else is auto-detected or optional.

```yaml
dem_aster:
  source: gee_raster
  path: projects/sat-io/open-datasets/ASTER/GDEM   # asset type auto-detected via ee.data.getAsset()

dem_local:
  source: local_raster
  path: data/for_testing/dem/TG4NHB-dem.tif
  band: 1                                           # optional, defaults to band 1
```

**GEE auto-detection:** asset type (IMAGE vs IMAGE_COLLECTION) is resolved via
`ee.data.getAsset()` — no `asset_type` key needed in catalog.

**Local auto-detection:** CRS, resolution, nodata, and band count are read from the
file via rasterio (`_inspect_raster()`).

**`feature_spec` block** (optional, GEE only): for advanced config like date filtering,
cloud masking, collection reducer, and derived bands:

```yaml
sen2_ndvi:
  source: gee_raster
  path: COPERNICUS/S2_SR_HARMONIZED
  feature_spec:
    collection: COPERNICUS/S2_SR_HARMONIZED
    temporal_window_days: 30
    cloud_pct_max: 20
    cloud_mask: s2
    derived_band: NDVI
```

---


## Files overview

```
src/biodata/
    enrich.py           ← main entry point
    config.py           ← catalog loading + local raster auto-detection
    auth.py             ← GEE authentication from credentials/ee_credentials.json
    reducers.py         ← Python-side reducer registry (mean, std, quantiles, ...)
    adapters/
        base.py         ← BaseAdapter
        gee_adapter.py   ← GeeRasterAdapter + all image-building utilities
        local_adapter.py ← LocalRasterAdapter
    output.py           ← parquet writing, manifest, window TIF export
    qc.py               ← coverage QC flags
    provenance.py       ← provenance metadata per feature
    history.py          ← replay last run from manifest

configs/
    catalog.yml         ← dataset registry
    groups.yml          ← example groups config
    run.yml             ← example run config

credentials/
    ee_credentials.json ← GEE service account key (gitignored)
```

---
