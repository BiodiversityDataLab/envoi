# Architecture Overview

This document describes how envoi is put together: which module owns what, how data flows
through `extract()`, the adapter interface, how the Earth Engine adapter builds an image,
and how the web app runs extraction jobs.
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
           |  3. load_defaults() reads configs/defaults.yml (cached per process).
           |  4. _parse_and_validate_dates(): drop rows without a date (with a warning),
           |     accept ISO 8601 intervals, times, time zones, year-only and year-month
           |     values, and reduce each to YYYY-MM-DD.
           |     _validate_and_reproject_crs(): reproject from input_crs to WGS84 and
           |     reject coordinates outside ±90° / ±180°.
           |  5. load_catalogs() merges the built-in catalog with datasets registered
           |     through update_catalog(). User entries override built-in entries.
           |  6. _as_config_list() turns the config into a list of run configs.
           |  7. For each run config: _parse_run_config() builds a RunSettings.
           |     Raster mode then runs _validate_sample_ids(), before any adapter exists.
           |  8. For each (dataset, window_size) pair:
           |
           +---> [tabular] _process_dataset_tabular()
           |       adapter.fetch_stats_batch() -> (stats, meta) for each point
           |         GEE: server-side reducers; local: Python reducers (reducers.py)
           |         The "point" reducer adds an exact-pixel sample in the same call.
           |       _append_stat_columns() -> attach_quality_control()
           |       -> adapter.build_dataset_meta()
           |
           +---> [raster]  _process_dataset_raster()
           |       adapter.export_tiles() -> one GeoTIFF for each point
           |         GEE: getDownloadURL + requests; local: rasterio (native grid, or
           |         rasterio.warp.reproject onto a UTM grid when resample_m is set)
           |       adapter.build_dataset_meta() -> _write_tile_manifest()
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
    <dataset>/tiles_manifest.csv       maps each input row to its tile
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
| `extract.py` | Orchestrator. Renames user columns to canonical names, delegates input validation, parses each run config into `RunSettings`, loops over (dataset, window) pairs, dispatches to the mode helpers (`_process_dataset_tabular`, `_process_dataset_raster`), writes the tile manifest, and assembles the stats and QC tables and the metadata sidecar. Wraps adapter progress updates into public `ProgressEvent`s. |
| `_input_validation.py` | Validates the input DataFrame: required id/lat/lon columns, sample IDs for raster mode (`_validate_sample_ids`, also used by the web app), GBIF / ISO 8601 date parsing, and CRS reprojection to WGS84 with a coordinate range check. Returns warnings that the orchestrator records in the metadata sidecar. |
| `_config_parsing.py` | Run-config validation. Defines `RunSettings`, normalizes a dict / list / YAML path into a list of run configs (`_as_config_list`), parses one raw dict into `RunSettings` (`_parse_run_config`), and resolves the reducer list for a dataset's `data_type` (`_resolve_stats_for_dataset`). Expands the dataset-entry shorthand (string / `{name: [bands]}` / `{name: {bands: [...]}}`), rejects unsafe `batch_id` values, and rejects derived bands for local datasets. |
| `_output_assembly.py` | Tabular-output post-processing. Turns per-point `(stats_dict, meta_dict)` results into named columns (including the per-class `class_count` / `class_fraction` expansion), rounds stat columns, restores the user's column names, records the bands each dataset actually used for the sidecar, and writes the CSV / Parquet file. |
| `_filenames.py` | Portable output names. `build_tile_filenames` and `sanitize_filename_component` turn sample IDs into names that use only `[A-Za-z0-9._-]`, adding a hash suffix to rewritten IDs. `is_safe_path_component` rejects unsafe `batch_id` values and dataset keys. Both adapters use it, so tiles are named identically. |
| `catalog.py` | Loads and validates catalog YAMLs (built-in `ee_catalog.yml` and datasets registered through `update_catalog()`), caches `defaults.yml`, exposes `list_datasets()` / `reset_catalog()`, and auto-detects CRS, resolution, nodata, and band count for local rasters through rasterio. Earth Engine asset inspection happens in the GEE adapter, not here. |
| `catalog_docs.py` | Renders catalog entries as a Markdown document (theme sections, summary tables, per-dataset details). Exposed as `catalog_markdown()`. `scripts/generate_dataset_docs.py` uses it to regenerate `docs/datasets.md`. |
| `geo.py` | UTM helpers: `get_utm_crs()` (both adapters build metre-accurate windows with it), `get_utm_zone_label()` (the GEE `use_utm_zone` filter), and `build_tile_crs_zones()` (tile CRS metadata in the GEE adapter). |
| `progress.py` | `ProgressEvent` (public) and the low-level `(completed, total)` callback that adapters call through `emit_progress_step()`. Independent of the `quiet` progress bars. |
| `adapters/__init__.py` | Adapter registry. Maps `data_source` strings (`earth_engine`, `local`) to adapter classes through `register()` / `get_adapter()`. Imports the built-in adapters so they register themselves. |
| `adapters/base.py` | `BaseAdapter`: the context-manager lifecycle (`close()`, `__enter__` / `__exit__`) and the shared method signatures. See "Adapter interface". |
| `adapters/local_adapter.py` | Local raster adapter. Reads GeoTIFFs through rasterio, crops a UTM-zone square around each point, supports per-band nodata (including NaN), and exports tiles at native resolution, or resampled and snapped to a UTM grid with `rasterio.warp.reproject` for parity with GEE. |
| `adapters/earth_engine/adapter.py` | `GeeRasterAdapter`. Detects the asset type, checks the native scale, runs per-point work in a `ThreadPoolExecutor`, composes the sibling helpers, and builds the per-dataset metadata. Caches collection-level state for the lifetime of the adapter instance (see "GEE image building pipeline"). |
| `adapters/earth_engine/_image.py` | GEE initialization on first use (calls `auth.init_gee()` when Earth Engine is not yet initialized) and the session-pool patch, pixel-grid snapping, the collection timestamp index, `nearest` / `contains` / `mosaic` image selection, derived-band registration (`KNOWN_DERIVED_BANDS`: slope, aspect), and the central `_build_image` pipeline. |
| `adapters/earth_engine/_reducers.py` | Server-side reducer registry, `_build_combined_reducer` (so all statistics for a point resolve in one round-trip), `reduceRegion` result parsing (single-band, multi-band, per-class histogram), and the per-band coverage summary. |
| `adapters/earth_engine/_tiles.py` | Size guard for synchronous downloads (`_check_tile_size`) and the retrying tile downloader (`_download_tile_via_url`, which wraps `ee.Image.getDownloadURL` + `requests` with exponential backoff for 429 / 5xx responses). |
| `reducers.py` | Python-side reducer registry (mean, median, min, max, sum, std, var, count, mode, class_count, class_fraction, q05..q95), used by the local adapter. Also `list_reducers()` (public) and `validate_reducers()`, which warns about reducer / `data_type` mismatches such as `mean` on a categorical raster. `point` is not in the registry; the adapters handle it. |
| `qc.py` | Builds per-dataset QC columns from the adapters' meta dicts (core flags, plus date, CRS, and per-band coverage where available), warns when points fall below `min_coverage_pct`, and splits the merged DataFrame into the stats and QC tables. |
| `metadata.py` | Writes the sidecar JSON (`run` / `config` / `datasets` / optional `warnings`) and provides the summaries that adapters put into it (`summarize_date_info`, `summarize_tile_export`). |
| `auth.py` | `init_gee()`: initializes Earth Engine from a service-account JSON key. The key is a file, found through `ENVOI_EE_CREDENTIALS`, the user config directory, or `./credentials/` (see `README.md`), or the key content given as `credentials_json`. With `credentials_json`, there is no file lookup, and an invalid key gives a fixed `ValueError` without a chained exception, because the error of the key parser can contain the key. |
| `_version.py` | `__version__`. Change it only as part of a release. |
| `envoi_webapp/app.py` | Streamlit interface, started with `envoi-webapp` (local mode) or by `deploy/serve/app.py` in the container image (hosted mode). Validates the form, applies the hosted upload and form limits, submits jobs to the `JobManager`, and shows the progress, "Cancel", and the result. Never calls `ee.Initialize()`. See "Web-app execution model". |
| `envoi_webapp/helpers.py` | Web-app logic without Streamlit: CSV validation, run-config building (one run config for each data-product row, with that row's output type), service-account key validation, and `redact_credential_secrets()`. Reuses `_validate_sample_ids` and the reducer sets from `reducers.py`. |
| `envoi_webapp/settings.py` | Reads `ENVOI_WEBAPP_MODE`, `ENVOI_WEBAPP_WORKSPACE`, and `ENVOI_WEBAPP_MAX_JOBS` into `WebappSettings`. `HostedLimits` holds the hosted limits. `check_upload()` and `check_hosted_limits()` check an upload before the parse and a form before a job. |
| `envoi_webapp/jobs.py` | `JobManager`, one for each server process (`get_job_manager()`): job admission and limits, job workspaces, one worker process for each job, the message channel, "Cancel", the hosted housekeeping thread, and workspace deletion. Keeps only a hash of each key (`key_hash()`). |
| `envoi_webapp/worker.py` | The worker process (`python -m envoi_webapp.worker`): reads the job request from stdin, calls `init_gee(credentials_json=...)` and `extract()`, writes the run log, makes the ZIP archive in hosted mode, and sends JSON-line messages. Removes key material from stderr, the run log, and error messages. |
| `envoi_webapp/job_protocol.py` | The data that crosses the process boundary: `JobRequest`, the `progress` / `done` / `error` messages (`encode_message()`, `parse_message()`), `JobState`, and `JobSnapshot`. |

## Adapter interface

`extract()` uses an adapter only through these members:

| Member | Used in | Returns |
|---|---|---|
| `with AdapterClass(spec) as adapter:` | both modes | The adapter. `close()` runs on exit, even when the batch raises. `spec` is the catalog entry merged with any per-run band override. |
| `fetch_stats_batch(lats, lons, window_m, reducer_names, *, dates=None, progress_desc=None, disable_progress=False, progress_callback=None)` | tabular | A list with one `(stats_dict, meta_dict)` pair for each point, in input order. |
| `export_tiles(lats, lons, window_m, output_dir, *, ids=None, dates=None, dataset_name="dataset", resample_m=None, filename_suffix=None, progress_desc=None, disable_progress=False, progress_callback=None)` | raster | `(paths, meta_list)`, one entry for each point. Tiles go to `output_dir/<dataset_name>/`. `paths[i]` is `None` when that point's export failed. |
| `build_dataset_meta(spec, meta_list=None, exported_paths=None, quality=None, lats=None, lons=None)` | both modes | The dataset's entry in the sidecar's `datasets` section: source information, native CRS and resolution, band names, date-selection summary, and quality statistics. |

Both adapters implement these with identical signatures. Notes:

- `export_tiles` is not declared on `BaseAdapter`. Each adapter defines it.
- `BaseAdapter` also declares `fetch_values` (raw pixels for one window) and `fetch_batch` (a loop over `fetch_values`).
  Only `LocalRasterAdapter` implements `fetch_values`, and it uses it internally. The GEE adapter implements neither,
  and nothing in envoi calls `fetch_batch`.
- The `"point"` reducer samples the exact pixel at each `(lat, lon)` and merges the result into the same `stats_dict` as the window reducers.
  GEE resolves both in one `getInfo()` round-trip per point. The local adapter reads the window, then samples the pixel with `src.sample()` on the same open dataset.
- A failure for one point gives that point an empty result instead of stopping the batch. See "Failures for single points" in `docs/coding_guidelines.md`.
- Each adapter module registers itself at the end with `register("<data_source>", AdapterClass)`.
  `get_adapter()` raises `KeyError` for an unknown `data_source`.

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

For each point: _build_image(dataset_spec, date, geometry, ...)
    IMAGE:            ee.Image(path)
    IMAGE_COLLECTION:
        filterBounds(point)
        filter(UTM_ZONE == zone)          <- only when dataset_spec.use_utm_zone
        image selection, by collection_date_policy:
          mosaic          -> mosaic().setDefaultProjection(native projection);
                             sample dates are ignored
          sample date     -> filterDate(start, end).first(), with start/end taken from the
                             timestamp index: "nearest" (default) start, or the interval that
                             "contains" the date. Dates outside the collection clamp to the
                             first or last image.
          no sample date  -> the most recent image in the timestamp index
          no index        -> if the timestamp fetch failed: filterDate around the sample date
                             (±1 day for "nearest", [date, date + 1 day] for "contains"),
                             or a mosaic of the whole collection when there is no date
    _apply_derived_bands                  <- slope, aspect (KNOWN_DERIVED_BANDS)
    select(bands + derived bands)         <- derived bands only when no bands are set;
                                             all bands when neither is set
```

This pipeline lives in `adapters/earth_engine/_image.py` (`_build_image`, `_resolve_date_filter_range`, `_find_nearest_timestamp`, `_get_collection_time_bounds`).
`GeeRasterAdapter._get_image()` in `adapter.py` decides which branch applies to each point.

State cached on the adapter instance (`extract()` creates one instance per (dataset, window size) pair):

- Set in `__post_init__`: the resolved `dataset_spec` (`_dataset_spec`), the native projection (`_native_proj`),
  the prebuilt image for IMAGE assets and mosaic-policy collections (`_static_image`),
  and the native scale and CRS (`_cached_native_scale`, `_cached_native_crs`; one `getInfo()` when no `native_scale_m` override is set).
- Set on the first batch: the collection timestamp index (`_collection_time_starts`, `_collection_time_ends`)
  and the band names and count (`_cached_band_name(s)`, `_cached_band_count`).
  Both are loaded on the main thread before worker threads start.

All per-point worker threads share this state. The number of workers comes from the catalog entry's
`max_workers`, or else from `defaults.yml`.

## Web-app execution model

The web app (`src/envoi_webapp/`) never runs an extraction in the Streamlit server process.
`earthengine-api` keeps its credentials in one module-global state object, and `ee.Initialize()` replaces them for all threads of the process.
Only a process boundary keeps the keys of two users apart. The web app therefore runs each extraction job in its own worker process,
which initializes Earth Engine with the key of that job only. The server process never calls `ee.Initialize()`, in either mode.

```
Streamlit server process (one for each app; never calls ee.Initialize())
  session A --+                        +-- worker process A: init_gee(key A) -> extract() -> run log (-> ZIP)
  session B --+-- JobManager ----------+-- worker process B: init_gee(key B) -> extract() -> run log (-> ZIP)
              |   (one for each        |
              |    server process)     +-- stdin:  pickled JobRequest with the key, then closed
              |                            stdout: JSON lines (progress, then done or error)
              |                            stderr: redacted, at most 1 MB, in the job workspace
              +-- housekeeping thread (hosted mode, every 5 s): run time, workspace size,
                  abandon time-out, retention
```

- **Session.** `render_app()` runs top to bottom on each rerun. A session keeps only its job ID and a random session token in `st.session_state`.
  `jobs.get_job_manager()` gives every session the same `JobManager`, so a job outlives a rerun.
  It keeps the manager in a module-level variable under a lock, not in the Streamlit resource cache:
  any browser client can clear that cache for all users, and a new manager would start the hosted limits again from zero.
- **Start.** `JobManager.submit()` checks the limits and reserves a slot under one lock. Then it creates a job workspace with a random name (128 bits, mode `0o700`)
  and starts `python -m envoi_webapp.worker` with `subprocess.Popen`. The command line and the environment hold no key.
  The worker environment has no `ENVOI_EE_CREDENTIALS` or `GOOGLE_APPLICATION_CREDENTIALS`, and it has `PYTHONSAFEPATH=1`
  (`python -m` would otherwise import modules from the working folder). The working folder is the job workspace.
- **Message channel.** One channel thread for each job writes the pickled `JobRequest` to the worker's stdin, closes stdin, and drops the bytes.
  The manager then keeps only a SHA-256 hash of the key (`key_hash()`), for the limit of one job for each key and for `cancel_for_key()`.
  The worker answers on a copy of its original stdout, with one JSON object per line (`job_protocol.py`).
  It points file descriptor 1 and `sys.stdout` to stderr, so a `print()` cannot write to the channel.
- **Key material.** The worker removes key material with `redact_credential_secrets()` from stderr, the run log, and the `error` message.
  An uncaught exception writes one redacted line, never a traceback. The worker disables core files.
- **Status.** While a job runs, a Streamlit fragment calls `JobManager.snapshot()` every 2 seconds. Each call is the job's heartbeat for the abandon time-out.
  When the job has a final state, the fragment reruns the whole page, which then shows the result.
- **States.** A job is `running`, then gets one final state: `succeeded`, `failed`, `cancelled` (by the user), or `stopped` (by a limit). The first final state wins.
  "Cancel" and the housekeeping thread set the state first, then stop the process (`terminate()`, and `kill()` after 5 seconds).
- **Run log.** The worker writes `envoi-run-log-<UTC start time>.txt` into the output folder: the `WARNING` and higher records of the `envoi` logger,
  and the Python warnings from envoi source files. It is a web-app file. `extract()` does not write it, so the output contract of `extract()` does not change.

Both modes use the same worker and the same job manager. `ENVOI_WEBAPP_MODE` selects the mode.

| | Local mode (default, `envoi-webapp`) | Hosted mode (`ENVOI_WEBAPP_MODE=hosted`, Linux and macOS only) |
|---|---|---|
| Output folder | The folder that the user selects. | `<workspace>/outputs/`. The worker then moves the files into a ZIP archive in the workspace, which the user downloads. |
| Workspace root | A new folder `envoi-webapp-jobs-<random>` (mode `0o700`) for each server process, so that other users of the computer cannot change it. | The fixed folder `envoi-webapp-jobs`. On POSIX, the manager refuses it when it is a symbolic link, belongs to another user, or gives permissions to the group or to others. |
| Worker `HOME` and temporary folder (`TMPDIR`, `TEMP`, `TMP`, `CPL_TMPDIR`) | Unchanged, so that user installations of pip packages stay importable. | The job workspace. Temporary files of libraries then count toward the workspace size limit and are deleted with the workspace. |
| Limits | One running job for each session. | `HostedLimits` in `settings.py`: upload, form, run time, workspace size, concurrent jobs (`ENVOI_WEBAPP_MAX_JOBS`), one job for each key, disk budget, and abandon time-out. |
| Workspace deletion | When the job ends. The workspace holds only a marker file and the worker's stderr file. Files in the user's folder are never deleted. | Failed, cancelled, or stopped job: when the job ends (no partial results). Succeeded job: at the earliest of 10 minutes after the first download click, 30 minutes after the job ended, a new job of the same session, and "Clear results". When the job manager starts: the workspaces that an earlier server process left. |

`ENVOI_WEBAPP_WORKSPACE` sets the parent folder of the workspace root (default: the system temporary folder).
The manager deletes a workspace only after its worker process has exited.
Job state is in memory only. A restart of the server process ends all jobs and loses their results.
The container image in `deploy/serve/` starts the app in hosted mode (see `CONTRIBUTING.md`).
