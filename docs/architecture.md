# Architecture Overview

This document describes how envoi is put together: which module owns what, how data flows
through `extract()`, the adapter interface, and how the Earth Engine adapter builds an image.
For coding conventions, see `docs/coding_guidelines.md`. For user-facing output details
(column names, return values), see `README.md` and `docs/advanced_usage.md`.
Update this document when you add a module or change the adapter interface.

## Module map

```
                       User code, or the web app (src/envoi_webapp/)
                                         |
                               extract(df, config)               <- extract.py
                                         |
        +--------------------------------+--------------------------------+
        |                                |                                |
 _input_validation.py           _config_parsing.py                   catalog.py
 required columns,              load run-config YAML,                load + validate catalogs
 sample IDs (raster mode),      normalize to RunSettings,            (built-in + update_catalog()),
 dates, CRS reprojection        validate batch_id, bands, reducers   cache defaults.yml,
                                                                     inspect local rasters
                                                                           |
                                                                configs/ee_catalog.yml
                                                                configs/defaults.yml
                                         |
                   For each run config, for each (dataset, window_size):
                                         |
                               adapters/__init__.py
                               get_adapter(data_source)
                                         |
              +--------------------------+--------------------------+
              |                                                     |
      GeeRasterAdapter                                      LocalRasterAdapter
      adapters/earth_engine/                                adapters/local_adapter.py
        adapter.py    batches, worker threads, metadata       rasterio windows and tiles
        _image.py     ee.Image for each point                 reducers.py (Python-side stats)
        _reducers.py  server-side statistics
        _tiles.py     tile size guard, download
              |                                                     |
       Google Earth Engine API                              local GeoTIFF files
              |                                                     |
              +--------------------------+--------------------------+
                                         |
           Shared by both adapters: geo.py (UTM zones), _filenames.py (tile names),
           metadata.py (date and tile summaries), progress.py (progress events)
                                         |
             returns (stats, meta) for each point, or writes GeoTIFF tiles
                                         |
        +--------------------------------+--------------------------------+
        |                                |                                |
      qc.py                    _output_assembly.py                   metadata.py
 QC columns,                    stat columns, rounding,              sidecar JSON
 low-coverage warning,          user column names,                   (run / config /
 stats vs QC split              CSV / Parquet write                  datasets / warnings)
```

## Data flow

```
 Input DataFrame                       Config (dict, list, or YAML path)
 +--------------------+                +----------------------------------+
 | occurrenceID       |                | batch_id: "terrain"              |
 | decimalLatitude    |                | datasets: [dem_copernicus_glo30] |
 | decimalLongitude   |                | settings:                        |
 | (eventDate)        |                |   output_type: tabular           |
 +---------+----------+                |   statistics: [mean, std]        |
           |                           |   window_size_m: 200             |
           v                           +----------------+-----------------+
 +---------+----------+                                 |
 |     extract()      | <-------------------------------+
 +---------+----------+
           |
           |  1. _validate_required_columns(): fail fast if the id/lat/lon columns are
           |     missing (reported under the names the user supplied).
           |  2. Rename user columns to the canonical id/lat/lon/date for the rest of
           |     the pipeline. They are renamed back before output.
           |     The same rename gives each unmapped input column that has a working-
           |     column name (id, lat, lon, date, lat_original, lon_original) a
           |     temporary name from _build_reserved_column_renames(). The output
           |     tables restore its name.
           |  3. load_defaults() reads configs/defaults.yml (cached per process).
           |  4. _parse_and_validate_dates(): drop rows without a date (with a warning),
           |     accept only the listed forms (YYYY, YYYY-MM, YYYY-MM-DD, YYYYMMDD,
           |     each with an optional time of day and time zone; ISO 8601
           |     intervals; YYYY/MM/DD; date and datetime values; whole numbers
           |     as years or YYYYMMDD), and reduce each to YYYY-MM-DD. Raise ValueError for every
           |     other value (such as 15/06/2021 or 05.06.2021) and for dates that
           |     do not exist.
           |     _validate_and_reproject_crs(): reject missing, non-numeric, and
           |     infinite coordinates, reproject from input_crs to WGS84, and
           |     reject coordinates outside ±90° / ±180°.
           |  5. load_catalogs() merges the built-in catalog with datasets registered
           |     through update_catalog(). User entries override built-in entries.
           |  6. _as_config_list() turns the config into a list of run configs.
           |     _parse_run_configs() builds a RunSettings for every run config, and
           |     rejects duplicate batch_id values, before the first adapter exists.
           |     _validate_no_output_column_names() rejects input columns (original
           |     names) that a tabular run config would add to its output tables.
           |  7. For each RunSettings: raster mode runs _validate_sample_ids(), before
           |     the adapters of that run config exist.
           |     For every local dataset of every run config (catalog entry plus band
           |     overrides): _resolve_local_raster_settings() checks the catalog crs
           |     and nodata against the file, before the first adapter exists.
           |  8. For each (dataset, window_size) pair:
           |
           +---> [tabular] _process_dataset_tabular()
           |       adapter.fetch_stats_batch() -> (stats, meta) for each point
           |         GEE: server-side reducers; local: Python reducers (reducers.py)
           |         The "point" reducer adds an exact-pixel sample in the same call.
           |       _raise_if_every_point_failed(): RuntimeError when every point failed
           |       _append_stat_columns() -> attach_quality_control()
           |       -> adapter.build_dataset_meta() -> _warn_about_failed_points()
           |
           +---> [raster]  _process_dataset_raster()
           |       adapter.export_tiles() -> one GeoTIFF for each point
           |         GEE: one tile-size check for the batch (an oversized request
           |         raises ValueError before any download), then getDownloadURL +
           |         requests; local: rasterio (native grid, or
           |         rasterio.warp.reproject onto a UTM grid when resample_m is set)
           |       _raise_if_every_point_failed(): RuntimeError when every tile failed
           |       adapter.build_dataset_meta() -> _write_tile_manifest()
           |       -> _warn_about_failed_points()
           |
           v
  9. Write outputs and the metadata sidecar:

  Tabular mode, in output_dir/:
    <batch_id>.csv or .parquet         statistics (returned in memory instead
                                       when output_file_format is "dataframe")
    <batch_id>_qc.csv or .parquet      QC columns (CSV in "dataframe" mode);
                                       only when write_metadata=True
    <batch_id>_metadata.json           sidecar; only when write_metadata=True

  Raster mode, in output_dir/<batch_id>/:
    <dataset>/<id>-<dataset>.tif       one tile for each point
    <dataset>/tiles_manifest.csv       maps each input row to its tile (and the
                                       error of each failed tile)
    <batch_id>_metadata.json           sidecar (only when write_metadata=True)
    Multi-window runs add the window to tile and manifest names:
    <id>-<dataset>-200m.tif, tiles_manifest-200m.csv
```

`extract()` returns a dict keyed by `<batch_id>` (tabular) or `<batch_id>:<dataset>` (raster;
`<batch_id>:<dataset>:<window>m` in multi-window runs). A single run config with
`output_file_format: dataframe` returns the DataFrame itself.

## Module responsibilities

| Module | Role |
|---|---|
| `__init__.py` | The public API (`__all__`): `extract`, `update_catalog`, `list_datasets`, `reset_catalog`, `list_reducers`, `catalog_markdown`, `init_gee`, `CatalogError`, `ProgressEvent`, `__version__`. |
| `extract.py` | Orchestrator. Renames user columns to canonical names, delegates input validation, parses every run config into `RunSettings` before the first adapter is built, loops over (dataset, window) pairs, dispatches to the mode helpers (`_process_dataset_tabular`, `_process_dataset_raster`), writes the tile manifest, and assembles the stats and QC tables and the metadata sidecar. Wraps adapter progress updates into public `ProgressEvent`s. |
| `_input_validation.py` | Validates the input DataFrame: required id/lat/lon columns, temporary names for unmapped columns that have a working-column name (`_build_reserved_column_renames`), sample IDs for raster mode (`_validate_sample_ids`, also used by the web app), GBIF / ISO 8601 date parsing (only the listed forms; every other value raises), coordinates: missing, non-numeric, and infinite values are rejected, then CRS reprojection to WGS84 with a coordinate range check, and input columns with the name of an output column of the call (`_validate_no_output_column_names`). Returns warnings that the orchestrator records in the metadata sidecar. |
| `_config_parsing.py` | Run-config validation. Defines `RunSettings`, normalizes a dict / list / YAML path into a list of run configs (`_as_config_list`), parses one raw dict into `RunSettings` (`_parse_run_config`), parses all run configs of a call and rejects `batch_id` values whose output files would collide, also when letter case is ignored (`_parse_run_configs`), and resolves the reducer list for a dataset's `data_type` (`_resolve_stats_for_dataset`). Expands the dataset-entry shorthand (string / `{name: [bands]}` / `{name: {bands: [...]}}`; a band list replaces the catalog's bands, also when it has only derived bands, and marks the dataset for band-named columns with `band_naming: per_band`), rejects unsafe `batch_id` values, rejects a dataset listed twice in one run config (and, in raster mode, dataset names that differ only in letter case), and rejects derived bands for local datasets. |
| `_output_assembly.py` | Tabular-output post-processing. Turns per-point `(stats_dict, meta_dict)` results into named columns (including the per-class `class_count` / `class_fraction` expansion), rounds only these statistic columns (the user's own columns keep their values), restores the user's column names, records the bands each dataset actually used for the sidecar, and writes the CSV / Parquet file. |
| `_filenames.py` | Portable output names. `build_tile_filenames` and `sanitize_filename_component` turn sample IDs into names that use only `[A-Za-z0-9._-]`, adding a hash suffix to rewritten IDs. `build_tile_filenames` also adds the hash suffix to every ID whose name is equal to another ID's name when letter case is ignored (`Site1` and `site1`), because Windows and macOS treat such names as one file. Other IDs keep their names. `is_safe_path_component` rejects unsafe `batch_id` values and dataset keys. Both adapters use it, so tiles are named identically. |
| `catalog.py` | Loads and validates catalogs (built-in `ee_catalog.yml` and datasets registered through `update_catalog()`, including that each `data_source` has a registered adapter), caches `defaults.yml`, exposes `list_datasets()` / `reset_catalog()`, and, for catalog files (not dicts), auto-detects CRS, resolution, nodata, and band count for local rasters through rasterio (`resolution_m` in metres, from `geo.pixel_size_m()`). Earth Engine asset inspection happens in the GEE adapter, not here. |
| `catalog_docs.py` | Renders catalog entries as a Markdown document (theme sections, summary tables, per-dataset details). Exposed as `catalog_markdown()`. `scripts/generate_dataset_docs.py` uses it to regenerate `docs/datasets.md`. |
| `geo.py` | UTM helpers: `get_utm_crs()` (both adapters build metre-accurate windows with it), `get_utm_zone_label()` (the GEE `use_utm_zone` filter), and `build_tile_crs_zones()` (the sidecar `tile_crs`: the UTM zones of the points, for GEE tiles and for local tiles with `resample_m`; local tiles without `resample_m` keep the raster CRS). The tile grid of both adapters: `tile_pixel_count()` (pixels per side, also used by the GEE tile-size check) and `tile_top_left()` (tile corners on the pixel grid, with the point's pixel at the centre for odd sizes). `pixel_size_m()` converts a local raster's pixel width from CRS units to metres. |
| `progress.py` | `ProgressEvent` (public) and the low-level `(completed, total)` callback that adapters call through `emit_progress_step()`. Independent of the `quiet` progress bars. |
| `adapters/__init__.py` | Adapter registry. Maps `data_source` strings (`earth_engine`, `local`) to adapter classes through `register()` / `get_adapter()`. Imports the built-in adapters so they register themselves. |
| `adapters/base.py` | `BaseAdapter`, which declares the members that `extract()` uses and nothing else: `fetch_stats_batch`, `export_tiles`, and `build_dataset_meta` (with their shared signatures), and the context-manager lifecycle (`close()`, `__enter__` / `__exit__`). See "Adapter interface". |
| `adapters/local_adapter.py` | Local raster adapter. Reads GeoTIFFs through rasterio, crops a UTM-zone square around each point, supports per-band nodata (including NaN), uses the catalog `crs` and `nodata` only where the file declares none and rejects values that differ from the file (`_resolve_local_raster_settings`, which `extract()` also calls before the first adapter), widens a statistics window smaller than one native pixel to two pixels (as the GEE adapter does), and exports tiles at native resolution, or resampled onto the shared UTM tile grid of `geo.py` with `rasterio.warp.reproject` for parity with GEE. |
| `adapters/earth_engine/adapter.py` | `GeeRasterAdapter`. Detects the asset type, checks the native scale, runs per-point work in a `ThreadPoolExecutor`, composes the sibling helpers, and builds the per-dataset metadata. Caches collection-level state for the lifetime of the adapter instance (see "GEE image building pipeline"). |
| `adapters/earth_engine/_image.py` | GEE initialization on first use (calls `auth.init_gee()` when Earth Engine is not yet initialized) and the session-pool patch, the collection timestamp index, `nearest` / `contains` / `mosaic` image selection, derived-band registration (`KNOWN_DERIVED_BANDS`: slope, aspect), and the central `_build_image` pipeline, which also builds the companion image for circular aspect statistics (`circular_aspect=True`: the sine and cosine of the aspect, without flat pixels). |
| `adapters/earth_engine/_reducers.py` | Server-side reducer registry, `_build_combined_reducer` (so all statistics for a point resolve in one round-trip), `reduceRegion` result parsing (single-band, multi-band, per-class histogram), the circular mean, standard deviation, and variance of `aspect` (`_circular_aspect_statistics`), and the per-band coverage summary. |
| `adapters/earth_engine/_tiles.py` | Size guard for synchronous downloads (`_check_tile_size`) and the retrying tile downloader (`_download_tile_via_url`, which wraps `ee.Image.getDownloadURL` + `requests` with exponential backoff for 429 / 5xx responses and connection errors, raises any status other than 200 at once with Earth Engine's error message, writes each tile through a temporary file, and leaves no file (also no tile of an earlier run) when a download fails, and requests the exact tile grid with `crs_transform` and `dimensions`). |
| `reducers.py` | Python-side reducer registry (mean, median, min, max, sum, std, var, count, mode, class_count, class_fraction, q05..q95), used by the local adapter. Also `list_reducers()` (public) and `validate_reducers()`, which warns about reducer / `data_type` mismatches such as `mean` on a categorical raster. `point` is not in the registry; the adapters handle it. |
| `qc.py` | Builds per-dataset QC columns from the adapters' meta dicts (core flags, plus date, CRS, per-band coverage, and the error of failed points where available), warns when points fall below `min_coverage_pct`, and splits the merged DataFrame into the stats and QC tables. Also formats the error text of a failed point for both adapters (`format_point_error`), and counts failed points for the sidecar (`summarize_point_errors`). |
| `metadata.py` | Writes the sidecar JSON (`run` / `config` / `datasets` / optional `warnings`) and provides the summaries that adapters put into it (`summarize_date_info`, `summarize_tile_export`). |
| `auth.py` | `init_gee()`: initializes Earth Engine from a service-account JSON, found through `ENVOI_EE_CREDENTIALS`, the user config directory, or `./credentials/` (see `README.md`). |
| `_version.py` | `__version__`. Change it only as part of a release. |
| `envoi_webapp/app.py` | Streamlit interface, started with `envoi-webapp`. |
| `envoi_webapp/helpers.py` | Web-app logic without Streamlit: CSV validation, run-config building, temporary credential handling, and the call to `envoi.extract`. Reuses `_validate_sample_ids` and the reducer sets from `reducers.py`. |

## Adapter interface

`extract()` uses an adapter only through these members:

| Member | Used in | Returns |
|---|---|---|
| `with AdapterClass(spec) as adapter:` | both modes | The adapter. `close()` runs on exit, even when the batch raises. `spec` is the catalog entry merged with any per-run band override. |
| `fetch_stats_batch(lats, lons, window_m, reducer_names, *, dates=None, progress_desc=None, disable_progress=False, progress_callback=None)` | tabular | A list with one `(stats_dict, meta_dict)` pair for each point, in input order. |
| `export_tiles(lats, lons, window_m, output_dir, *, ids=None, dates=None, dataset_name="dataset", resample_m=None, filename_suffix=None, progress_desc=None, disable_progress=False, progress_callback=None)` | raster | `(paths, meta_list)`, one entry for each point. Tiles go to `output_dir/<dataset_name>/`. `paths[i]` is `None` when no tile was written for that point (failed export, point outside the extent, or no pixels). |
| `build_dataset_meta(spec, meta_list=None, exported_paths=None, quality=None, lats=None, lons=None)` | both modes | The dataset's entry in the sidecar's `datasets` section: source information, native CRS and resolution, band names, date-selection summary, and quality statistics. Local datasets also record the nodata value that envoi used (`nodata`: a number, `"nan"` for NaN, or `null`) and its source (`nodata_source`: `"file"`, `"catalog"`, or `null`). |

`BaseAdapter` declares these members, and nothing else. Both adapters implement them with identical signatures.
`tests/test_adapters.py` checks the parameter names, kinds, and defaults of each registered adapter against `BaseAdapter`. Notes:

- `LocalRasterAdapter.fetch_values` (raw pixels for one window) is not part of the interface.
  Only the local adapter has it, and it uses it internally.
- The `"point"` reducer samples the exact pixel at each `(lat, lon)` and merges the result into the same `stats_dict` as the window reducers.
  GEE resolves both in one `getInfo()` round-trip per point. The local adapter reads the window, then samples the pixel with `src.sample()` on the same open dataset.
- A failure for one point gives that point an empty result instead of stopping the batch. See "Failures for single points" in `docs/coding_guidelines.md`.
  The adapter failure contract for a failed point (its data request or read raised an exception that the adapter caught):
  - Tabular: the stats dict is empty (no statistic keys). The meta dict has the QC values of a point outside the extent
    (`in_extent=False`, `n_pixels=0`, `coverage_pct=0.0`, `had_nodata=False`).
  - Raster: `paths[i]` is `None`.
  - The meta dict (`meta_list[i]` in raster mode) has `error`, the error text from `qc.format_point_error`
    (`"<exception type>: <message>"` on one line, at most 500 characters), and `exception`, the exception object.
    `exception` stays in memory: it is never written to JSON or to an output table.
  - A point outside the extent, or with only nodata pixels, has no `error`.

  `extract.py`, not the adapters, applies the rules for a batch: `_raise_if_every_point_failed` raises `RuntimeError`
  (chained to the exception of the first failed point) when every point failed, and `_warn_about_failed_points` emits one warning
  and returns it for the sidecar's `warnings`. `qc.py` adds the `<dataset>_error_<N>m` QC column, `_write_tile_manifest`
  adds the manifest's `error` column, and both modes add `n_errors` and `ids_errors` to the sidecar's quality summaries.
- Each adapter module registers itself at the end with `register("<data_source>", AdapterClass)`.
  `update_catalog()` and catalog loading raise `CatalogError` for a `data_source` without a registered adapter.
  `get_adapter()` raises `ValueError` for an unknown `data_source`, as a last check.

## GEE image building pipeline

```
GeeRasterAdapter.__post_init__
    asset type      dataset_spec.image / dataset_spec.collection when set,
                    otherwise ee.data.getAsset(path) -> IMAGE or IMAGE_COLLECTION
    native scale    dataset_spec.native_scale_m when set, otherwise nominalScale(),
                    which raises if it is the ~111 km EPSG:4326 default

First fetch_stats_batch() / export_tiles() call (collections only)
    timestamp index system:time_start / system:time_end of the images that intersect
                    the batch's points, in one getInfo(). Skipped for the mosaic policy.
                    A failed request stops the run with RuntimeError.

For each point: _build_image(dataset_spec, date, geometry, ...)
    IMAGE:            ee.Image(path)
    IMAGE_COLLECTION:
        filterBounds(point)
        filter(UTM_ZONE == zone)          <- only when dataset_spec.use_utm_zone
        image selection, by collection_date_policy:
          mosaic          -> mosaic().setDefaultProjection(native projection);
                             sample dates are ignored
          sample date     -> fast path: filterDate(start, end).first(), with start/end taken
                             from the timestamp index: "nearest" (default) start, or the
                             interval that "contains" the date. Dates outside the collection
                             clamp to the first or last image. For "contains", end is at most
                             the centre day + 1 day (centre day: the date limited to the
                             index's first and latest start, at 00:00 UTC)
          no sample date  -> the most recent image in the timestamp index (as the date)
          no own image in -> per-point fallback (_build_fallback_image, ee.Algorithms.If in
          the fast path      the point's request): an own image (an image at the point after
                             the UTM-zone filter) near the centre day, first within ±32
                             days, then ±366 days. "nearest": fewest whole days
                             between start day and centre day, earlier start on ties.
                             "contains": the image that contains the centre day, else the
                             latest start before it, else the earliest start after it.
                             Null when no own image is in the windows
          no index        -> if no image intersects the batch's points: filterDate around the
                             sample date (±1 day for "nearest", [date, date + 1 day] for
                             "contains"), or a mosaic of the whole collection when there is
                             no date; no fallback
          no image at the -> when the collection has no image at the point on any date
          point              (limit(1).size() after filterBounds, before the UTM-zone and
                             date filters): the first image of the collection that has every
                             configured source band, with every pixel masked
                             (ee.Algorithms.If in the point's request), so the point has no
                             data and no error. When the point has images, but neither the
                             fast path nor the fallback selects an own image (for example a
                             wrong use_utm_zone), the image stays null, and the point is a
                             failed point
        image_dates       -> {start, end} of the selected image (before the masked image), or
                             null for a point without images. A branch of the statistics
                             request; one small getInfo() per tile in raster runs
    _apply_derived_bands                  <- slope, aspect (KNOWN_DERIVED_BANDS)
    select(bands + derived bands)         <- derived bands only when no bands are set;
                                             all bands when neither is set

    circular_aspect=True (in place of the two steps above):
        sin and cos of aspect, masked to slope > 0   <- for the circular mean, std, and var
                                                        of the derived aspect band
```

`_fetch_stats_single` sends up to three reductions in one `ee.Dictionary` request: the window statistics,
the circular aspect statistics (only when the derived `aspect` band is reduced with `mean`, `std`, or `var`),
and the point value (only for the `point` statistic).
For collections with per-point date selection, the same request also returns `image_dates`, the start and end of the selected image.
`_resolve_date_info` takes `image_time_start` and `image_time_end` from it, and `date_clamped` and `date_source` from the timestamp index.
In raster runs, `_export_single` requests `image_dates` once per tile, in its worker thread, before the download. A failed request makes a failed tile.
The circular aspect image is built for each point from the same image as the main image (`_build_image` with the same date and point), and it is never cached in `_static_image`.

This pipeline lives in `adapters/earth_engine/_image.py` (`_build_image`, `_build_fallback_image`, `_resolve_date_filter_range`, `_find_nearest_timestamp`, `_get_collection_time_bounds`).
`GeeRasterAdapter._get_image()` in `adapter.py` decides which branch applies to each point.

State cached on the adapter instance (`extract()` creates one instance per (dataset, window size) pair):

- Set in `__post_init__`: the resolved `dataset_spec` (`_dataset_spec`), the native projection (`_native_proj`),
  the prebuilt image for IMAGE assets and mosaic-policy collections (`_static_image`),
  and the native scale and CRS (`_cached_native_scale`, `_cached_native_crs`; one `getInfo()` when no `native_scale_m` override is set).
- Set on the first batch: the collection timestamp index (`_collection_time_starts`, `_collection_time_ends`)
  and the band names and count (`_cached_band_name(s)`, `_cached_band_count`).
  Both are loaded on the main thread before worker threads start.
  A failed band-name request is not cached. In tabular mode, each point then asks again.
  In raster mode, the tile-size check asks again and stops the run when that request also fails.

All per-point worker threads share this state. The number of workers comes from the catalog entry's
`max_workers`, or else from `defaults.yml`.
