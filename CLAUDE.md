# CLAUDE.md — EDDP / biodata-enricher

This file gives you context on the project vision, architecture, and decisions.
Update it as the project evolves.

---

## Code style

- Write inline comments liberally. Explain *what* non-trivial blocks do, not only the *why*.
  Many users/contributors of this project are not experienced programmers, so err on the side
  of more comments rather than fewer. This overrides the default "only comment the non-obvious WHY".
- Use full, descriptive variable names — avoid abbreviations unless they are universally understood
  (e.g. `df` for a pandas DataFrame is fine; `cfg`, `out_dir`, `cov`, `col` are not).
  Prefer `run_config` over `cfg`, `output_dir` over `out_dir`, `coverage_values` over `cov`.

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
extract(df, config)            ← main entry point
    ↓
ee_catalog.yml                ← built-in GEE dataset registry (bundled with package)
update_catalog(source)        ← user registers local/custom datasets at runtime
    ↓
Adapter (per dataset)
    ├── GeeRasterAdapter      ← queries GEE directly, parallel via ThreadPoolExecutor
    └── LocalRasterAdapter    ← reads GeoTIFF via rasterio, dynamic UTM per point
```

## API

The primary interface is `extract(df, config)` where `config` is a dict (single output)
or list of dicts (multiple outputs):

```python
extract(df, {
    "batch_id": "terrain",
    "datasets": ["dem_aster"],
    "settings": {
        "output_type": "tabular",          # "tabular" or "raster"
        "statistics": ["mean", "std"],
        "window_size_m": 200,
        "output_file_format": "parquet",        # "parquet" or "csv"
        "resample_m": 10,           # optional, for CNN-ready tiles
        "min_coverage_pct": 80,     # QC threshold
    },
})
```

To add custom datasets (local rasters or GEE assets not in the built-in catalog),
call `update_catalog()` once before extracting:

```python
from biodata import update_catalog
update_catalog("my_catalog.yml")          # from a YAML file
update_catalog({"datasets": {...}})       # or a dict
```

## Catalog design

The built-in catalog (`src/biodata/configs/ee_catalog.yml`) is bundled with the
package and loaded automatically. Only `data_source` and `path` are required —
everything else is auto-detected or optional.

```yaml
datasets:
  dem_aster:
    data_source: earth_engine
    path: projects/sat-io/open-datasets/ASTER/GDEM   # asset type auto-detected via ee.data.getAsset()
```

**GEE auto-detection:** asset type (IMAGE vs IMAGE_COLLECTION) is resolved via
`ee.data.getAsset()` — no `asset_type` key needed in catalog.

**Local auto-detection:** CRS, resolution, nodata, and band count are read from the
file via rasterio (`_inspect_raster()`).

**`feature_spec` block** (optional, GEE only): for advanced config like
cloud masking, collection reducer, and derived bands:

```yaml
sen2_ndvi:
  data_source: earth_engine
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
    config.py            ← catalog loading, update_catalog(), local raster auto-detection
    metadata.py          ← per-feature metadata + sidecar JSON writer
    auth.py              ← GEE authentication from credentials/ee_credentials.json
    reducers.py          ← Python-side reducer registry (mean, std, quantiles, ...)
    output.py            ← parquet/csv writing
    qc.py                ← coverage QC flags
    configs/
        ee_catalog.yml   ← built-in GEE dataset registry (bundled with package)
        local_catalog.yml← template for user-defined local datasets
        defaults.yml     ← project-wide setting defaults
    adapters/
        base.py          ← BaseAdapter
        gee_adapter.py   ← GeeRasterAdapter + all image-building utilities
        local_adapter.py ← LocalRasterAdapter

examples/
    run.yml              ← example run config

credentials/
    ee_credentials.json  ← GEE service account key (gitignored)
```

---
