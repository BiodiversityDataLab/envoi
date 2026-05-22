# Architecture Overview

## Module map

```
                          User code
                             |
                     extract(df, config)              <- src/envoi/extract.py
                             |
              +--------------+--------------+
              |                             |
         config.py                     configs/ee_catalog.yml
     load + merge catalogs            built-in GEE dataset registry
     update_catalog() / defaults      (data_source + path + ...)
     auto-detect local raster
              |                             |
              +-------------+---------------+
                            |
                    For each output config:
                    For each dataset:
                            |
              +-------------+-------------+
              |                           |
        adapters/__init__.py         reducers.py
        get_adapter(data_source)    get_reducer(name)
        adapter registry            mean, std, q05..q95, mode, ...
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
              |  returns stats + per-point QC metadata,
              |  or writes GeoTIFF tiles
              |
     +--------+---------+-----------+
     |                              |
   qc.py                       metadata.py
  coverage flags,             sidecar JSON
  per-band coverage,          (run / config /
  date / CRS columns          datasets / warnings)
```

## Data flow

```
 Input DataFrame                    Config (dict, list, or YAML path)
 +------------------+               +-------------------------+
 | id | lat | lon   |               | batch_id: "terrain"     |
 |    | (date)      |               | datasets: [dem_glo30]   |
 +--------+---------+               | settings:               |
          |                          |   output_type: tabular  |
          v                          |   statistics: [mean,std]|
 +--------+---------+               |   window_size_m: 200    |
 |    extract()      | <-------------+-------------------------+
 +--------+---------+
          |
          |  1. Validate/rename id/lat/lon/date columns; reproject to WGS84.
          |  2. Parse dates (mixed-format aware) and warn on incomplete ones.
          |  3. Load + merge catalogs (built-in EE + update_catalog() entries).
          |  4. For each output run config, for each (dataset, window) pair:
          |
          +---> [tabular]                  server stats (GEE) /
          |     adapter.fetch_stats_batch()  python reducers (local)
          |     -> {mean: 121.0, std: 4.2, ...} + QC meta
          |     ("point" reducer adds an exact-pixel sample in the same call)
          |
          +---> [raster]                   GeoTIFF tiles
                adapter.export_tiles()
                -> per-point .tif files + per-tile meta
                (GEE via geemap; local via rasterio.warp)
          |
          v
 +-----------------+  +------------------+  +---------------------+
 | stats.csv       |  | stats_qc.csv     |  | metadata.json       |
 | (or .parquet)   |  | coverage flags,  |  | run / config /      |
 +-----------------+  | date/CRS cols    |  | datasets / warnings |
                     +------------------+  +---------------------+
    OR (raster mode):
 +------------------------------+
 | out/{batch_id}/{dataset}/    |
 |   A-dem_glo30.tif            |
 |   B-dem_glo30.tif            |
 |   ...                        |
 | out/{batch_id}/              |
 |   {batch_id}_metadata.json   |
 +------------------------------+
```

## Module responsibilities

| Module | Role |
|---|---|
| `extract.py` | Orchestrator. Validates inputs, parses each run config into `RunSettings`, loops over (dataset, window) pairs, dispatches to adapters, assembles stats/QC DataFrames and metadata. |
| `config.py` | Loads and validates catalog YAMLs (built-in + user-registered via `update_catalog()`); auto-detects CRS/resolution/nodata/band count for local rasters via rasterio. |
| `adapters/__init__.py` | Adapter registry. Maps `data_source` strings (`earth_engine`, `local`) to adapter classes via `register()` / `get_adapter()`. |
| `adapters/base.py` | `BaseAdapter` with the shared context-manager lifecycle and method signatures (`fetch_values`, `fetch_batch`, `fetch_stats_batch`, `build_dataset_meta`). |
| `gee_adapter.py` | GEE adapter. Auto-detects asset type (IMAGE vs IMAGE_COLLECTION) via `ee.data.getAsset()`, builds per-point images with date selection (nearest / contains policies) and optional UTM-zone filtering, computes server-side stats via combined `reduceRegion` reducers, exports tiles via geemap. Patches the urllib3 connection pool for parallel workers. |
| `local_adapter.py` | Local raster adapter. Reads GeoTIFFs via rasterio, crops to a UTM-zone square per point, supports per-band nodata (including NaN), exports tiles either at native resolution or resampled+UTM-snapped via `rasterio.warp.reproject` for GEE parity. |
| `reducers.py` | Python-side reducer registry (mean, std, var, min, max, sum, count, median, mode, q05..q95). Used by the local adapter; the GEE adapter uses server-side reducers instead. |
| `qc.py` | Builds per-dataset QC columns from adapter meta dicts (core flags + optional date / region_crs / per-band coverage), and splits the merged DataFrame into stats vs QC files. |
| `metadata.py` | Writes the sidecar JSON (`run` / `config` / `datasets` / optional `warnings`), and provides helpers used by adapters (date-info summary, UTM-zone collection, tile-export summary). |
| `auth.py` | Initializes GEE from a service-account credentials JSON. |

## Adapter interface

Both adapters expose the same methods so `extract.py` can treat them uniformly:

| Method | Mode | Returns |
|---|---|---|
| `fetch_values(lat, lon, window_m, *, return_meta=False)` | Raw window pixels (one point) | `values_array` or `(values_array, meta_dict)` |
| `fetch_batch(lats, lons, window_m, *, dates=None, return_meta=False)` | Raw window pixels (many points) | List of values (or `(values, meta)`) — default impl loops `fetch_values`; GEE overrides with parallelism |
| `fetch_stats_batch(lats, lons, window_m, reducer_names, *, dates=None, progress_desc=None)` | Window stats + optional exact-pixel `"point"` sample | List of `(stats_dict, meta_dict)` — one per point |
| `export_tiles(lats, lons, window_m, output_dir, *, ids=None, dates=None, dataset_name=..., resample_m=None, filename_suffix=None, progress_desc=None)` | GeoTIFF tiles | `(paths, meta_list)` — per-point output paths and per-tile meta |
| `build_dataset_meta(spec, meta_list=None, exported_paths=None, quality=None, lats=None, lons=None)` | Per-dataset metadata for the sidecar | `dict` — source info, native CRS/resolution, band names, date-selection summary, quality stats |

The `"point"` reducer is a special name handled inside `fetch_stats_batch`: it samples the exact pixel at each `(lat, lon)` and is merged into the same `stats_dict` as the window reducers (single round-trip in GEE, single read in local).

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
            +-- filterBounds(point)             <- when called per-point with coords
            +-- filter(UTM_ZONE == zone)        <- when dataset_spec.use_utm_zone
            +-- date selection:
            |     no date     -> mosaic() (most recent non-masked pixel)
            |     date given  -> filterDate(start, end).first(), with start/end
            |                    resolved from cached system:time_start/time_end
            |                    using collection_date_policy: "nearest" (default)
            |                    or "contains"
            +-- _apply_derived_bands             <- slope, aspect (from KNOWN_DERIVED_BANDS)
            +-- select(source_bands + derived)   <- final output band list
```

Cached on the adapter for the lifetime of one `extract()` call: native projection (`_native_proj`), native scale (`_cached_native_scale`), native CRS (`_cached_native_crs`), band name(s) and count (`_cached_band_name(s)`, `_cached_band_count`), and collection start/end timestamp indices (`_collection_time_starts/_ends`). The cache is populated lazily on first use and shared across all per-point worker threads.
