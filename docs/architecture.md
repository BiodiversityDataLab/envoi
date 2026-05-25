# Architecture Overview

## Module map

```
                          User code
                             |
                     extract(df, config)              <- src/envoi/extract.py
                             |
              +--------------+--------------+
              |                             |
        _input_validation.py           _config_parsing.py
     required-column check          load YAML, normalize to
     date parsing & cleanup         list of RunSettings,
     CRS reprojection               resolve per-dataset stats
              |                             |
              +-------------+---------------+
                            |
                       catalog.py
              load + merge catalogs (built-in + user),
              update_catalog() / list_datasets(),
              auto-detect local raster CRS/res/nodata
                            |
                    configs/ee_catalog.yml
                    configs/defaults.yml
                            |
                    For each output run config:
                    For each (dataset, window_size):
                            |
              +-------------+-------------+
              |                           |
        adapters/__init__.py         reducers.py
        get_adapter(data_source)    get_reducer(name)
        adapter registry            mean, std, q05..q95, mode, ...
              |
     +--------+-----------------+
     |                          |
 GeeRasterAdapter         LocalRasterAdapter
 adapters/earth_engine/   adapters/local_adapter.py
   adapter.py
   _image.py  (per-point ee.Image build)
   _reducers.py (combined reducers + parsing)
   _tiles.py  (size guard + tile download)
     |                          |
     | Google Earth Engine API  | rasterio (local GeoTIFF)
     |                          | + geo.py (UTM helpers)
     +--------+-----------------+
              |
              |  returns stats + per-point QC metadata,
              |  or writes GeoTIFF tiles
              |
     +--------+--------------------+
     |              |              |
   qc.py     _output_assembly.py   metadata.py
  coverage   append stat columns,  sidecar JSON
  flags,     round, rename core    (run / config /
  splits     columns, write CSV/   datasets / warnings),
  stats/QC   Parquet               date + tile summaries
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
          |  1. _validate_required_columns(): fail fast if id/lat/lon missing
          |     (under whichever names the user supplied).
          |  2. Rename user columns -> canonical id/lat/lon/date for the rest
          |     of the pipeline; restored before the output is written.
          |  3. _parse_and_validate_dates(): accept GBIF / ISO 8601 flexibility
          |     (intervals, time-of-day, year-only), collapse to YYYY-MM-DD,
          |     warn on incomplete dates. _validate_and_reproject_crs(): if
          |     input_crs is not WGS84, reproject lat/lon to EPSG:4326.
          |  4. load_catalogs() merges built-in EE catalog with any entries
          |     registered at runtime via update_catalog().
          |  5. _as_config_list() + _parse_run_config() turn the raw config
          |     into a list of RunSettings (one per output).
          |  6. For each RunSettings, for each (dataset, window_size) pair:
          |
          +---> [tabular]                  server stats (GEE) /
          |     adapter.fetch_stats_batch()  python reducers (local)
          |     -> {mean: 121.0, std: 4.2, ...} + QC meta
          |     ("point" reducer adds an exact-pixel sample in the same call)
          |
          +---> [raster]                   GeoTIFF tiles
                adapter.export_tiles()
                -> per-point .tif files + per-tile meta
                (GEE via getDownloadURL + requests; local via rasterio.warp)
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
| `extract.py` | Orchestrator. Renames user columns to canonical names, delegates input validation, parses each run config into `RunSettings`, loops over (dataset, window) pairs, dispatches to adapters, assembles stats/QC DataFrames and metadata. |
| `_input_validation.py` | Validates the input DataFrame: required id/lat/lon columns, GBIF/ISO 8601 date parsing (intervals, year-only, time-of-day, timezones), and CRS reprojection to WGS84. Returns warnings the orchestrator persists into the metadata sidecar. |
| `_config_parsing.py` | Pure config-validation. Defines `RunSettings`, normalizes a single dict / list / YAML path into a list of run configs (`_as_config_list`), parses one raw dict into `RunSettings` (`_parse_run_config`), and resolves the right reducer list per dataset data_type (`_resolve_stats_for_dataset`). Also owns the dataset-entry shorthand expansion (string / `{name: [bands]}` / `{name: {bands: [...]}}`). |
| `_output_assembly.py` | Tabular-output post-processing. Turns per-point `(stats_dict, meta_dict)` results into named DataFrame columns (including the per-class `class_count` / `class_fraction` expansion), rounds stat columns, restores the user's original core column names, builds the resolved per-dataset entry for the sidecar, and writes the final CSV / Parquet file. |
| `catalog.py` | Loads and validates catalog YAMLs (built-in `ee_catalog.yml` + user-registered via `update_catalog()`), caches `defaults.yml`, exposes `list_datasets()` / `reset_catalog()`, and auto-detects CRS/resolution/nodata/band count for local rasters via rasterio. |
| `geo.py` | General-purpose CRS / UTM helpers used by both adapters and metadata: `get_utm_crs()`, `get_utm_zone_label()`, `build_tile_crs_zones()`. Resolves WGS84 lon/lat to the right UTM zone for meter-accurate window construction. |
| `adapters/__init__.py` | Adapter registry. Maps `data_source` strings (`earth_engine`, `local`) to adapter classes via `register()` / `get_adapter()`. Imports the built-in adapters so they self-register. |
| `adapters/base.py` | `BaseAdapter` with the shared context-manager lifecycle and method signatures (`fetch_values`, `fetch_batch`, `fetch_stats_batch`, `build_dataset_meta`). |
| `adapters/local_adapter.py` | Local raster adapter. Reads GeoTIFFs via rasterio, crops to a UTM-zone square per point, supports per-band nodata (including NaN), exports tiles either at native resolution or resampled+UTM-snapped via `rasterio.warp.reproject` for GEE parity. |
| `adapters/earth_engine/adapter.py` | `GeeRasterAdapter` class. Orchestrates per-point work via `ThreadPoolExecutor`, composes the sibling helpers, builds the per-dataset metadata, and caches collection-level state (native projection / scale / CRS, band names, timestamp indices) for the lifetime of one `extract()` call. |
| `adapters/earth_engine/_image.py` | GEE SDK init + session-pool patch, pixel-grid snapping, collection timestamp fetching, nearest / contains date selection, derived-band registration (`KNOWN_DERIVED_BANDS` — slope, aspect), and the central `_build_image` pipeline (load -> spatial filter -> date select -> derive -> select bands). |
| `adapters/earth_engine/_reducers.py` | Server-side reducer registry, `_build_combined_reducer` (so multiple stats resolve in one round-trip), `reduceRegion` result parsing (single-band, multi-band, per-class histogram unpack), and per-band coverage summary aggregation. |
| `adapters/earth_engine/_tiles.py` | Synchronous-download size guard (`_check_tile_size`) and retrying tile downloader (`_download_tile_via_url`, wrapping `ee.Image.getDownloadURL` + `requests` with exponential backoff for 429/5xx). |
| `reducers.py` | Python-side reducer registry (mean, std, var, min, max, sum, count, median, mode, q05..q95). Used by the local adapter; the GEE adapter uses server-side reducers instead. Also exposes `list_reducers()` and `validate_reducers()` (warns on reducer / data_type mismatches like `mean` on a categorical raster). |
| `qc.py` | Builds per-dataset QC columns from adapter meta dicts (core flags + optional date / region_crs / per-band coverage), warns on points below `min_coverage_pct`, and splits the merged DataFrame into stats vs QC files. |
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

Adapters are context managers (`BaseAdapter.__enter__` / `__exit__`). The orchestrator uses `with AdapterClass(spec) as adapter:` so any held resources (e.g. an open rasterio dataset) are released even if the per-point batch raises.

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

This pipeline lives in `adapters/earth_engine/_image.py` (`_build_image`, `_find_nearest_timestamp`, `_get_collection_time_bounds`, `_snap_to_grid`). The `GeeRasterAdapter` class in `adapter.py` composes it together with the reducer / tile helpers.

Cached on the adapter for the lifetime of one `extract()` call: native projection (`_native_proj`), native scale (`_cached_native_scale`), native CRS (`_cached_native_crs`), band name(s) and count (`_cached_band_name(s)`, `_cached_band_count`), and collection start/end timestamp indices (`_collection_time_starts/_ends`). The cache is populated lazily on first use and shared across all per-point worker threads.
