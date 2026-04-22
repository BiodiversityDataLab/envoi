# Architecture Overview

## Module map

```
                          User code
                             |
                     enrich(df, cfg)                 <- src/biodata/enrich.py
                             |
              +--------------+--------------+
              |                             |
         config.py                     catalog.yml
     load & validate catalog          dataset registry
     auto-detect local raster         (source + path)
              |                             |
              +-------------+---------------+
                            |
                    For each dataset:
                            |
              +-------------+-------------+
              |                           |
        adapters/__init__.py         reducers.py
        get_adapter(source)         get_reducer(name)
        adapter registry            mean, std, q10, ...
              |
     +--------+--------+
     |                  |
 GeeRasterAdapter  LocalRasterAdapter
 gee_adapter.py    local_adapter.py
     |                  |
     | Google Earth     | rasterio
     | Engine API       | (local GeoTIFF)
     |                  |
     +--------+---------+
              |
              |  returns pixel values + QC metadata
              |
     +--------+---------+-----------+
     |                  |           |
   qc.py           output.py   metadata.py
  coverage       Parquet/CSV    sidecar JSON
  flags          writer         (run/config/
                                 datasets/quality)
```

## Data flow

```
 Input DataFrame                    Config (dict or YAML)
 +------------------+               +-------------------------+
 | id | lat | lon   |               | run_id: "terrain"       |
 |    | (date)      |               | datasets: [dem_aster]   |
 +--------+---------+               | settings:               |
          |                          |   output_type: tabular  |
          v                          |   statistics: [mean,std]|
 +--------+---------+               |   window_size_m: 200    |
 |    enrich()      | <-------------+-------------------------+
 +--------+---------+
          |
          |  1. Load catalog -> resolve adapter per dataset
          |  2. For each dataset:
          |
          +---> [tabular + server stats]  GEE fast path
          |     adapter.fetch_stats_batch()
          |     -> server-side reduceRegion
          |     -> {mean: 121.0, std: 4.2}
          |
          +---> [tabular + local stats]   Python reducers
          |     adapter.fetch_batch()
          |     -> raw pixel arrays
          |     -> reducers.py computes stats
          |
          +---> [tabular + point]         Single pixel
          |     adapter.fetch_points_batch()
          |     -> value at exact (lat, lon)
          |
          +---> [raster]                  GeoTIFF tiles
          |     adapter.export_images()   (GEE via geemap)
          |     adapter.export_windows()  (local via rasterio)
          |
          v
 +-----------------+  +------------------+  +---------------------+
 | stats.parquet   |  | stats_qc.parquet |  | metadata.json       |
 | (or .csv)       |  | coverage flags   |  | run / config /      |
 +-----------------+  +------------------+  | datasets / quality  |
                                            +---------------------+
    OR (raster mode):
 +------------------------------+
 | out/{name}/{dataset}/        |
 |   A-dem_aster.tif            |
 |   B-dem_aster.tif            |
 +------------------------------+
```

## Module responsibilities

| Module | Role |
|---|---|
| `enrich.py` | Orchestrator. Parses config, loops over datasets, dispatches to adapters, assembles outputs. |
| `config.py` | Loads and validates `catalog.yml`. Auto-detects CRS/resolution for local rasters via rasterio. |
| `adapters/__init__.py` | Adapter registry. Maps source names (`earth_engine`, `local`) to adapter classes. |
| `gee_adapter.py` | GEE adapter. Handles asset type detection (IMAGE vs IMAGE_COLLECTION), image building (date filtering, cloud masking, mosaicking), server-side stats, point sampling, and raster export. Uses `filterBounds` for tiled collections and caches native projection. |
| `local_adapter.py` | Local raster adapter. Reads GeoTIFF via rasterio, crops to UTM window per point, returns pixel arrays. Supports `resample_m` via rasterio `reproject`. |
| `reducers.py` | Python-side reducer registry (mean, std, quantiles, etc.). Used for local rasters; GEE uses server-side reducers instead. |
| `output.py` | Writes tabular results as Parquet or CSV. |
| `qc.py` | Computes QC flags (in_extent, n_pixels, had_nodata, coverage_pct) from adapter metadata. |
| `metadata.py` | Writes sidecar JSON with run info, config, per-dataset source details, and coverage quality summary. |
| `auth.py` | Initializes GEE from a service account credentials JSON file. |

## Adapter interface

Both adapters expose the same methods so `enrich.py` can treat them uniformly:

| Method | Mode | Returns |
|---|---|---|
| `fetch_batch(lats, lons, window_m)` | Raw pixels | List of `(values_array, meta_dict)` |
| `fetch_stats_batch(lats, lons, window_m, reducers)` | Server stats (GEE only) | List of `(stats_dict, meta_dict)` |
| `fetch_points_batch(lats, lons)` | Point sampling | List of `(values_dict, meta_dict)` |
| `export_images(...)` / `export_windows(...)` | Raster tiles | List of output file paths |

## GEE image building pipeline

```
catalog path
    |
    v
ee.data.getAsset() -> IMAGE or IMAGE_COLLECTION?
    |
    +-- IMAGE: ee.Image(path)
    |
    +-- IMAGE_COLLECTION:
            |
            +-- filterBounds(point)     <- spatial constraint
            +-- filterDate(window)      <- if temporal_window_days set
            +-- filter(cloud_pct)       <- if cloud_pct_max set
            +-- map(cloud_mask)         <- if cloud_mask set
            +-- reduce:
            |     no dates  -> mosaic()
            |     windowed  -> mean()
            |     closest   -> first()
            +-- select(band)            <- if specific band requested
            +-- derived_band            <- NDVI, EVI, slope, aspect
```
