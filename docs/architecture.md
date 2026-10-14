# Architecture Overview

This document describes how envoi is put together: which module owns what, how data flows
through `extract()`, the adapter interface, how to add a data provider, and how the Earth Engine
adapter builds an image.
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
 dates, CRS reprojection        reject unknown keys, validate        call the adapters' catalog
                                batch_id, bands, reducers, and       hooks, cache defaults.yml
                                output_type against the adapter's          |
                                capability declarations         configs/ee_catalog.yml
                                                                configs/defaults.yml
                                         |
            Pre-flight, for every dataset of every run config, before any adapter exists:
                     get_adapter(data_source).check_dataset(merged spec)
                                         |
          For each run config, for each dataset: build one adapter, and use it
          for every window_size. extract() checks each result of the adapter.
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
           Shared by the adapters: geo.py (UTM zones), _filenames.py (tile names),
           metadata.py (date summary), progress.py (progress events)
                                         |
             returns (stats, meta) for each point, or writes GeoTIFF tiles
                                         |
          extract.py: checks each result against the result contract,
          and computes the tile summary quality.tiles (metadata.py)
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
           |     The entries are copies, so the run cannot change the stored catalog.
           |  6. _as_config_list() turns the config into a list of run configs.
           |     _parse_run_configs() builds a RunSettings for every run config, and
           |     rejects duplicate batch_id values, before the first adapter exists.
           |     _parse_run_config() rejects unknown keys in a run config and in its
           |     settings. It checks the output_type, the reducers, and the derived
           |     bands of each dataset against the capability declarations of the
           |     dataset's adapter class, and stores the reducer list of each tabular
           |     dataset (RunSettings.stats_by_dataset). It builds no adapter.
           |     _validate_no_output_column_names() rejects input columns (original
           |     names) that a tabular run config would add to its output tables.
           |  7. For each RunSettings: raster mode runs _validate_sample_ids(), before
           |     the adapters of that run config exist.
           |     For every dataset of every run config (catalog entry plus band
           |     overrides): the class method check_dataset() of the dataset's adapter
           |     runs the adapter's own pre-flight check, before the first adapter
           |     exists. The local adapter checks the file, the bands, and the catalog
           |     crs and nodata against the file (_resolve_local_raster_settings()).
           |  8. For each dataset of a run config: build one adapter
           |     (with AdapterClass(merged_config) as adapter:), and use it for every
           |     window_size. close() runs once, after the last window.
           |     For each window_size:
           |
           +---> [tabular] _process_dataset_tabular()
           |       adapter.fetch_stats_batch() -> (stats, meta) for each point
           |         GEE: server-side reducers; local: Python reducers (reducers.py)
           |         The "point" reducer adds an exact-pixel sample in the same call.
           |       _check_adapter_result(): RuntimeError when the result breaks the
           |         result contract (a bug in the adapter)
           |       _raise_if_every_point_failed(): RuntimeError when every point failed
           |       _append_stat_columns() -> attach_quality_control()
           |       -> adapter.build_dataset_meta() -> _check_adapter_result()
           |       -> _warn_about_failed_points()
           |
           +---> [raster]  _process_dataset_raster()
           |       adapter.export_tiles() -> one GeoTIFF for each point
           |         GEE: one tile-size check for the batch (an oversized request
           |         raises ValueError before any download), then getDownloadURL +
           |         requests; local: rasterio (native grid, or
           |         rasterio.warp.reproject onto a UTM grid when resample_m is set)
           |       _check_adapter_result(): RuntimeError when the result breaks the
           |         result contract (a bug in the adapter)
           |       _raise_if_every_point_failed(): RuntimeError when every tile failed
           |       adapter.build_dataset_meta() -> _check_adapter_result()
           |       -> summarize_tile_export(): the tile summary quality.tiles
           |       -> _write_tile_manifest() -> _warn_about_failed_points()
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
| `extract.py` | Orchestrator. Renames user columns to canonical names, delegates input validation, parses every run config into `RunSettings` before the first adapter is built, runs the pre-flight check (`check_dataset`) of every dataset's adapter, builds one adapter for each run config and dataset and uses it for every window size, dispatches to the mode helpers (`_process_dataset_tabular`, `_process_dataset_raster`), checks each adapter result against the result contract (`_check_adapter_result`, `RuntimeError` for a broken rule), computes the tile summary `quality.tiles` in raster mode, writes the tile manifest, and assembles the stats and QC tables and the metadata sidecar. Wraps adapter progress updates into public `ProgressEvent`s. Names no built-in `data_source`. |
| `_input_validation.py` | Validates the input DataFrame: required id/lat/lon columns, temporary names for unmapped columns that have a working-column name (`_build_reserved_column_renames`), sample IDs for raster mode (`_validate_sample_ids`, also used by the web app), GBIF / ISO 8601 date parsing (only the listed forms; every other value raises), coordinates: missing, non-numeric, and infinite values are rejected, then CRS reprojection to WGS84 with a coordinate range check, and input columns with the name of an output column of the call (`_validate_no_output_column_names`). Returns warnings that the orchestrator records in the metadata sidecar. |
| `_config_parsing.py` | Run-config validation. Defines `RunSettings`. Normalizes a dict, a list, or a YAML path into a list of run configs (`_as_config_list`). Parses one raw dict into `RunSettings` (`_parse_run_config`). Parses all run configs of a call (`_parse_run_configs`), and rejects `batch_id` values whose output files would collide, also when letter case is ignored. Resolves the reducer list of each tabular dataset for its `data_type` (`_resolve_stats_for_dataset`), and stores it in `RunSettings.stats_by_dataset`. Rejects a run config or `settings` that is not a mapping, and a `datasets` value that is not a list. Rejects unknown keys in a run config and in `settings` (`_RUN_CONFIG_KEYS`, `_SETTINGS_KEYS`). The message suggests the closest allowed key. Expands the dataset-entry shorthand: a string, `{name: [bands]}`, or `{name: {bands: [...]}}`. A band list replaces the catalog's bands, also when it has only derived bands. It also marks the dataset for band-named columns with `band_naming: per_band`. Rejects unsafe `batch_id` values, and a dataset listed twice in one run config. In raster mode, also rejects dataset names that differ only in letter case. Checks each dataset against the capability declarations of its adapter class (`get_adapter(data_source)`): the `output_type`, the reducers, and the derived bands of a band list. Splits a band list on the derived bands of all registered adapters (`all_computable_derived_bands()`). Builds no adapter, and names no built-in `data_source`. |
| `_output_assembly.py` | Tabular-output post-processing. Turns per-point `(stats_dict, meta_dict)` results into named columns (including the per-class `class_count` / `class_fraction` expansion), rounds only these statistic columns (the user's own columns keep their values), restores the user's column names, records the bands each dataset actually used for the sidecar, and writes the CSV / Parquet file. |
| `_filenames.py` | Portable output names. `build_tile_filenames` and `sanitize_filename_component` turn sample IDs into names that use only `[A-Za-z0-9._-]`, adding a hash suffix to rewritten IDs. `build_tile_filenames` also adds the hash suffix to every ID whose name is equal to another ID's name when letter case is ignored (`Site1` and `site1`), because Windows and macOS treat such names as one file. Other IDs keep their names. `is_safe_path_component` rejects unsafe `batch_id` values and dataset keys. Both adapters use it, so tiles are named identically. |
| `catalog.py` | Loads and validates catalogs: the built-in `ee_catalog.yml`, and the datasets registered through `update_catalog()`. Caches `defaults.yml`, and exposes `list_datasets()` and `reset_catalog()`. One function, `_validate_catalog_entry`, checks each entry of a catalog file and of a catalog dict, in three steps. First, the generic rules. The entry is a mapping, and its `data_source` has a registered adapter. Each key is in `_GENERIC_CATALOG_KEYS` or in the adapter's `catalog_keys`. `path` is a non-empty text or an `os.PathLike`. `data_type` is absent, null, or in `_DATA_TYPES`. `derived_bands` and `supported_derived_bands` name only the adapter's `computable_derived_bands`. Second, the adapter's `validate_catalog_entry`. Third, for catalog files only (not dicts), the adapter's `inspect_catalog_entry`. A broken rule raises `CatalogError`, also a `ValueError` of a hook. The function stores `path` as text. envoi stores and returns copies of the entries. A catalog dict without a `datasets` mapping raises `CatalogError`. Names no built-in `data_source`: the Earth Engine rules are in `GeeRasterAdapter.validate_catalog_entry`, and the inspection of local rasters is in `LocalRasterAdapter.inspect_catalog_entry`. Earth Engine asset inspection happens in the GEE adapter's `__post_init__`. |
| `catalog_docs.py` | Renders catalog entries as a Markdown document (theme sections, summary tables, per-dataset details). Exposed as `catalog_markdown()`. `scripts/generate_dataset_docs.py` uses it to regenerate `docs/datasets.md`. |
| `geo.py` | UTM helpers: `get_utm_crs()` (both adapters build metre-accurate windows with it), `get_utm_zone_label()` (the GEE `use_utm_zone` filter), and `build_tile_crs_zones()` (the sidecar `tile_crs`: the UTM zones of the points, for GEE tiles and for local tiles with `resample_m`; local tiles without `resample_m` keep the raster CRS). The tile grid of both adapters: `tile_pixel_count()` (pixels per side, also used by the GEE tile-size check) and `tile_top_left()` (tile corners on the pixel grid, with the point's pixel at the centre for odd sizes). `pixel_size_m()` converts a local raster's pixel width from CRS units to metres. |
| `progress.py` | `ProgressEvent` (public) and the low-level `(completed, total)` callback that adapters call through `emit_progress_step()`. Independent of the `quiet` progress bars. |
| `adapters/__init__.py` | Adapter registry. Maps `data_source` strings to adapter classes through `register()` / `get_adapter()`. `register()` raises `ValueError` for a name that does not match `^[a-z][a-z0-9_]*$`, a class that is not a subclass of `BaseAdapter`, a name under which a different class is registered (a class of a reloaded module replaces the registered class), and a capability declaration that is not valid. It is internal: it is not in `envoi.__all__`. `all_computable_derived_bands()` returns the union of the registered adapters' `computable_derived_bands`. Imports the built-in adapters (`earth_engine`, `local`) so they register themselves. |
| `adapters/base.py` | `BaseAdapter`, which declares what `extract()`, `_config_parsing.py`, and `catalog.py` use of an adapter: the capability declarations (`output_types`, `supported_reducers`, `computable_derived_bands`, `catalog_keys`), the class-method hooks (`check_dataset`, `validate_catalog_entry`, `inspect_catalog_entry`), the calls `fetch_stats_batch`, `export_tiles`, and `build_dataset_meta` (with their shared signatures), and the context-manager lifecycle (`close()`, `__enter__` / `__exit__`). Its docstring holds the result contract and the lifetime of an adapter. `REQUIRED_POINT_META_KEYS` (defined in `qc.py`) holds the meta keys that every point has, and `_BAND_NAMING_KEY` / `_PER_BAND_NAMING` the mark of a per-call band list. See "Adapter interface". |
| `adapters/local_adapter.py` | Local raster adapter. Reads GeoTIFFs through rasterio, crops a UTM-zone square around each point, supports per-band nodata (including NaN), uses the catalog `crs` and `nodata` only where the file declares none and rejects values that differ from the file (`_resolve_local_raster_settings`, which its `check_dataset` also calls before the first adapter is built), widens a statistics window smaller than one native pixel to two pixels (as the GEE adapter does), and exports tiles at native resolution, or resampled onto the shared UTM tile grid of `geo.py` with `rasterio.warp.reproject` for parity with GEE. For catalog files, `inspect_catalog_entry` adds the CRS, resolution, nodata, and band count of the raster to the entry (`resolution_m` in metres, from `geo.pixel_size_m()`). |
| `adapters/earth_engine/adapter.py` | `GeeRasterAdapter`. Detects the asset type, checks the native scale, runs per-point work in a `ThreadPoolExecutor`, composes the sibling helpers, and builds the per-dataset metadata. Its `validate_catalog_entry` checks the Earth Engine catalog rules: `data_type` is required and not null, `dataset_spec` has only its allowed keys (`dataset_spec.bands` and `dataset_spec.derived_bands` are rejected), and `collection_date_policy` is `nearest`, `contains`, or `mosaic`. Caches collection-level state for the lifetime of the adapter instance (see "GEE image building pipeline"). |
| `adapters/earth_engine/_image.py` | GEE initialization on first use (calls `auth.init_gee()` when Earth Engine is not yet initialized) and the session-pool patch, the collection timestamp index, `nearest` / `contains` / `mosaic` image selection, derived-band registration (`KNOWN_DERIVED_BANDS`: slope, aspect), and the central `_build_image` pipeline, which also builds the companion image for circular aspect statistics (`circular_aspect=True`: the sine and cosine of the aspect, without flat pixels). |
| `adapters/earth_engine/_reducers.py` | Server-side reducer registry, `_build_combined_reducer` (so all statistics for a point resolve in one round-trip), `reduceRegion` result parsing (single-band, multi-band, per-class histogram), the circular mean, standard deviation, and variance of `aspect` (`_circular_aspect_statistics`), and the per-band coverage summary. |
| `adapters/earth_engine/_tiles.py` | Size guard for synchronous downloads (`_check_tile_size`) and the retrying tile downloader (`_download_tile_via_url`, which wraps `ee.Image.getDownloadURL` + `requests` with exponential backoff for 429 / 5xx responses and connection errors, raises any status other than 200 at once with Earth Engine's error message, writes each tile through a temporary file, and leaves no file (also no tile of an earlier run) when a download fails, and requests the exact tile grid with `crs_transform` and `dimensions`). |
| `reducers.py` | Python-side reducer registry (mean, median, min, max, sum, std, var, count, mode, class_count, class_fraction, q05..q95), used by the local adapter. `KNOWN_REDUCERS` is the closed set of reducer names that a run config can request: the registry plus `point`. Config parsing, `register()` (each adapter's `supported_reducers` must be a subset), and the adapters use it. Also `list_reducers()` (public) and `validate_reducers()`, which warns about reducer / `data_type` mismatches such as `mean` on a categorical raster. `point` is not in the registry; the adapters handle it. |
| `qc.py` | Defines `REQUIRED_POINT_META_KEYS`, the meta keys that every point of `fetch_stats_batch` has. Builds per-dataset QC columns from the adapters' meta dicts (core flags, plus date, CRS, per-band coverage, and the error of failed points where available), warns when points fall below `min_coverage_pct`, and splits the merged DataFrame into the stats and QC tables. Also formats the error text of a failed point for both adapters (`format_point_error`), and counts failed points for the sidecar (`summarize_point_errors`). |
| `metadata.py` | Writes the sidecar JSON (`run` / `config` / `datasets` / optional `warnings`) and provides summaries for it: `summarize_date_info`, which an adapter can call in `build_dataset_meta` (the GEE adapter does), and `summarize_tile_export`, which `extract.py` calls for the tile summary `quality.tiles` of every adapter. |
| `auth.py` | `init_gee()`: initializes Earth Engine from a service-account JSON, found through `ENVOI_EE_CREDENTIALS`, the user config directory, or `./credentials/` (see `README.md`). |
| `_version.py` | `__version__`. Change it only as part of a release. |
| `envoi_webapp/app.py` | Streamlit interface, started with `envoi-webapp`. |
| `envoi_webapp/helpers.py` | Web-app logic without Streamlit: CSV validation, run-config building, temporary credential handling, and the call to `envoi.extract`. Reuses `_validate_sample_ids` and the reducer sets from `reducers.py`. |

## Adapter interface

An adapter class has three parts:

- capability declarations: class attributes that state what the adapter supports,
- class-method hooks: the provider's own rules, which run before any adapter instance exists,
- the calls that `extract()` makes on an adapter instance during a run.

`BaseAdapter` (`adapters/base.py`) declares all three. Its docstring holds the result contract.

### Capability declarations

Each adapter sets these class attributes. Each one is a `frozenset` of `str`, and `register()` checks it.
The built-in adapters are dataclasses, so they set the declarations without a type annotation: in a dataclass, an annotation makes the attribute a field.

| Declaration | Meaning | `earth_engine` | `local` |
|---|---|---|---|
| `output_types` | The output types that the adapter supports. A non-empty subset of `{"tabular", "raster"}`. | `tabular`, `raster` | `tabular`, `raster` |
| `supported_reducers` | The reducers that `fetch_stats_batch` supports. A subset of `reducers.KNOWN_REDUCERS`. Empty only when `output_types` has no `"tabular"`. | all of `KNOWN_REDUCERS` | all of `KNOWN_REDUCERS` |
| `computable_derived_bands` | The derived bands that the adapter computes from a source band. | `slope`, `aspect` | none |
| `catalog_keys` | The top-level catalog keys that the adapter accepts in addition to the generic keys (`_GENERIC_CATALOG_KEYS` in `catalog.py`). It includes the keys that `inspect_catalog_entry` adds. | `dataset_spec`, `max_workers`, `crs` | `crs`, `nodata`, `resolution_m`, `type`, `band_count` |

The generic modules use the declarations before any adapter is built:

- `_parse_run_config()` raises `ValueError` for an `output_type`, a reducer, or a derived band of a band list that the dataset's adapter does not declare.
- `_parse_run_config()` splits a per-call band list on the derived bands of all registered adapters (`all_computable_derived_bands()`).
  So a band list cannot select a source band that has the name of a derived band, also for a dataset whose adapter computes no derived bands.
- `catalog.py` raises `CatalogError` for a key that is not a generic key and not in `catalog_keys`.
  It also raises it for a `derived_bands` or `supported_derived_bands` value that is not in `computable_derived_bands`.

### Class-method hooks

These class methods run before any adapter instance exists.
The defaults check nothing, and the default `inspect_catalog_entry` returns an unchanged copy of the entry.
No hook sends a network request or changes its input. The catalog hooks never import an optional SDK.

| Hook | Called by, and when | Errors |
|---|---|---|
| `validate_catalog_entry(name, entry)` | `catalog.py`, for each entry of the adapter's `data_source` in a catalog file or dict, after the generic checks. It checks the adapter's own catalog rules. | `CatalogError`, which `catalog.py` raises unchanged. Or `ValueError`, which `catalog.py` raises as a `CatalogError` with the same message after `datasets.<name>: `. |
| `inspect_catalog_entry(name, entry)` | `catalog.py`, for each entry of a catalog file (not of a dict), after `validate_catalog_entry`. It returns a new dict with the values that it reads from the data source. It adds only keys of `catalog_keys`, and replaces no value that the entry sets. When it cannot read the data source, it logs a warning and returns the entry without the missing values, so that the catalog still loads. | As for `validate_catalog_entry`. |
| `check_dataset(spec, *, context)` | `extract()`, for every dataset of every run config, after it parsed the run configs and before it builds the first adapter. `spec` is the catalog entry merged with the band overrides of the run config. The hook checks what it can check without a data request. Each message starts with `context`. | `ValueError` for a configuration that is not valid. `FileNotFoundError` or another `OSError` for a missing resource, for example a file. |

### Calls on an adapter instance

`extract()` uses an adapter instance only through these members:

| Member | Used in | Returns |
|---|---|---|
| `with AdapterClass(spec) as adapter:` | both modes | The adapter. `extract()` builds one instance for each (run config, dataset) pair, and uses it for every window size of that pair. `close()` runs once on exit, after the last window, also when a window raises. `spec` is the catalog entry merged with any per-run band override. |
| `fetch_stats_batch(lats, lons, window_m, reducer_names, *, dates=None, progress_desc=None, disable_progress=False, progress_callback=None)` | tabular | A list with one `(stats_dict, meta_dict)` pair for each point, in input order. |
| `export_tiles(lats, lons, window_m, output_dir, *, ids=None, dates=None, dataset_name="dataset", resample_m=None, filename_suffix=None, progress_desc=None, disable_progress=False, progress_callback=None)` | raster | `(paths, meta_list)`, one entry for each point. Tiles go to `output_dir/<dataset_name>/`. `paths[i]` is `None` when no tile was written for that point (failed export, point outside the extent, or no pixels). |
| `build_dataset_meta(spec, meta_list=None, quality=None, lats=None, lons=None)` | both modes | The dataset's entry in the sidecar's `datasets` section: source information, native CRS and resolution, band names, date-selection summary, and, in tabular mode, the coverage summary that `extract()` passes as `quality`, as the last key. In raster mode, `extract()` adds the tile summary `quality.tiles` to the returned dict itself. Local datasets also record the nodata value that envoi used (`nodata`: a number, `"nan"`, `"inf"`, or `"-inf"` for a value that is not finite, or `null`) and its source (`nodata_source`: `"file"`, `"catalog"`, or `null`). |

An adapter overrides `build_dataset_meta`, `fetch_stats_batch` when it declares `"tabular"`, and `export_tiles` when it declares `"raster"`, with the parameters of `BaseAdapter`.
`tests/test_adapters.py` checks each registered adapter: the parameter names, kinds, and defaults of each call and each class-method hook against `BaseAdapter`, that each hook is still a class method, and that the adapter overrides the calls of its output types.

**Result checks.** `extract.py` checks the result of each call before it uses the result (`_check_adapter_result`).
A broken rule raises `RuntimeError`. The message names the dataset, the adapter class, the call, and the rule, and says that the problem is a bug in the adapter.
The rules (the `BaseAdapter` docstring states them in full):

- `fetch_stats_batch` returns a list with one `(stats, meta)` pair of dicts for each point, in input order.
  Each meta dict has every key of `REQUIRED_POINT_META_KEYS`: `in_extent`, `n_pixels`, `had_nodata`, and `coverage_pct`.
- `export_tiles` returns `(paths, meta_list)`, two lists with one entry for each point. Each path is a `pathlib.Path` or `None`, and each meta is a dict.
- `build_dataset_meta` returns a dict.

**Lifetime.** One adapter instance serves every window size of its (run config, dataset) pair.
So the setup of the adapter (for example the Earth Engine asset lookup, or the open file of a local raster) runs once for each run config and dataset.
An adapter must not cache a value that depends on the window size.
`extract()` makes a deep copy of the dict of each `build_dataset_meta` call before it adds keys to it and merges the windows.

Notes:

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
  and returns it for the sidecar's `warnings`. `_warn_about_far_images` does the same for points whose own fallback image
  (see `image_dates` below) is more than 32 days from their sample date. `qc.py` adds the `<dataset>_error_<N>m` QC column, `_write_tile_manifest`
  adds the manifest's `error` column, and both modes add `n_errors` and `ids_errors` to the sidecar's quality summaries.
- Each adapter module registers itself at the end with `register("<data_source>", AdapterClass)`.
  `update_catalog()` and catalog loading raise `CatalogError` for a `data_source` without a registered adapter, and for an entry that breaks a generic rule or a rule of its adapter.
  `get_adapter()` raises `ValueError` for an unknown `data_source`, as a last check.

## Adding a data provider

A data provider is a source of environmental data with its own access method, for example Earth Engine or local GeoTIFF files.
In code, a provider is one `data_source` value with one registered adapter class.
A new provider is one new adapter module. The generic modules do not change: `extract.py`, `_config_parsing.py`, `catalog.py`,
`_input_validation.py`, `_output_assembly.py`, `qc.py`, `metadata.py`, `reducers.py`, `geo.py`, and `_filenames.py`.
`tests/test_provider_contract.py` shows this with a test-only fake provider (`FakeStatsAdapter`), which runs through `envoi.extract()`.
Use it as a model, together with "Adapters" in `docs/coding_guidelines.md`.

1. **Module.** Put the adapter in `src/envoi/adapters/<provider>.py`.
   When it needs helper modules, use a subpackage `src/envoi/adapters/<provider>/` (model: `adapters/earth_engine/`).
   New providers live in the envoi source tree: `register()` is internal, and envoi has no plugin API.
2. **Class.** Subclass `BaseAdapter`. The constructor takes one argument, `spec`: the catalog entry merged with the band overrides of the run config.
   Read the settings from `spec`, and store them as attributes.
3. **Capability declarations.** Set `output_types`, `supported_reducers`, `computable_derived_bands`, and `catalog_keys` (see "Capability declarations").
   Declare only what the adapter supports. The generic checks then reject the rest before any adapter is built.
   Options of the provider go into the catalog entry, through `catalog_keys`. A run config has no provider-specific settings.
4. **Registration.** At the end of the module, call `register("<data_source>", AdapterClass)`. The built-in modules import it as `_register`.
   Import the module at the end of `adapters/__init__.py`, so that `import envoi` registers the adapter.
   The `data_source` name must match `^[a-z][a-z0-9_]*$`. Users type it in catalog entries.
5. **Calls.** Override `build_dataset_meta`, `fetch_stats_batch` for `"tabular"`, and `export_tiles` for `"raster"`, with the parameters of `BaseAdapter`.
   Override `close()` when the adapter holds a resource, and make it safe to call more than once.
   - `fetch_stats_batch` and `export_tiles`: report progress with `progress.emit_progress_step(progress_callback, completed, total)`.
     Call it with `completed=0` before the first point, and again after each finished point. `total` is the number of points.
     Show a progress bar with the text `progress_desc`, unless `disable_progress` is true. The built-in adapters use `tqdm.auto.tqdm`.
   - `export_tiles`: write the tiles to `output_dir/<dataset_name>/`, with the file names from `_filenames.build_tile_filenames(ids, dataset_name, filename_suffix)`.
     With `resample_m`, put each tile on the shared tile grid in the UTM zone of the point (`geo.get_utm_crs(lon, lat)`).
     The tile has `geo.tile_pixel_count(window_m, resample_m)` pixels per side, and its corner comes from `geo.tile_top_left(x_m, y_m, n_pixels, resample_m)`.
     The tiles of all providers then align. Without `resample_m`, the Earth Engine adapter uses the same grid at the native scale, and the local adapter keeps the grid of the file.
   - `build_dataset_meta`: return the sidecar fields that both built-in adapters write: `data_source`, `path`, `data_type` (when the entry sets it), `asset_type`, `native_crs`, `native_spatial_resolution_m`, `tile_crs` (in raster mode), and `dataset_information` (when the entry sets it).
     Add `quality`, when given, as the last key. Return a new dict for each call.
6. **Hooks.** Override a class-method hook when the provider has a rule for it (see "Class-method hooks"):
   - `validate_catalog_entry`: the provider's catalog rules, for example a required key, or the allowed keys and values of a nested option.
   - `inspect_catalog_entry`: values that the adapter reads from the data source of an entry in a catalog file, for example the CRS of a file.
   - `check_dataset`: each check that needs no data request, for example that a file exists or that a band is valid.

   Each hook raises the errors of the hook table, with a message that says how to fix the problem.
7. **Result contract.** Return the results that "Result contract" in the `BaseAdapter` docstring describes.
   The docstring gives the statistic-key forms, the optional meta keys that generic code reads, the result of a failed point, and the spec keys of a per-call band list. In summary:
   - Each statistic key becomes an output column: `<dataset>_<key>_<N>m`, or `<dataset>_<key>` for `point` and `{band}_point`.
   - Each meta dict of `fetch_stats_batch` has every key of `REQUIRED_POINT_META_KEYS`. The meta dicts of `export_tiles` need no key.

   `extract()` checks the shape of each result (see "Result checks").
8. **Optional SDKs and credentials.**
   - Import an optional SDK only inside the functions that use it. `import envoi`, catalog loading, and the catalog hooks never import it.
   - In `check_dataset`, check that the SDK can be imported and that the required credentials exist, without a network request.
     The message says what to install or what to set.
     The error type for a missing SDK or missing credentials in `check_dataset` is not decided yet.
     The specification of the first provider that needs an SDK or credentials decides it, and updates "Error contract" in `docs/coding_guidelines.md`.
   - Read the credentials on first use, as the Earth Engine adapter initializes Earth Engine on first use.
   - Never put credentials in the catalog, the run config, or the metadata sidecar.
   - A new SDK is a new dependency. Discuss it before you add it (see "Tool preferences" in `docs/coding_guidelines.md`).

   `earthengine-api` and `rasterio` are required dependencies, so the built-in adapters import them at module level.
9. **Lifetime.** `extract()` builds one instance for each (run config, dataset) pair, and uses it for every window size.
   Do not cache a value that depends on the window size.
10. **Tests.** Add tests that use no network, as `tests/test_provider_contract.py` does:
    - Registration and signatures: `tests/test_adapters.py` checks every registered adapter automatically.
      Add the adapter, with the expected values of its four capability declarations, to the parameters of `TestBuiltInCapabilityDeclarations`.
    - Catalog: an entry with the provider's own keys loads. An unknown key and each rule of `validate_catalog_entry` raise `CatalogError` that names the key.
      `inspect_catalog_entry` returns a new dict and does not change its input.
    - Pre-flight: each failure of `check_dataset`, also a missing SDK or missing credentials, raises before any adapter is built.
      Count the constructor calls, as `TestPreFlightChecks` does.
    - Extraction: a run through `envoi.extract()` with stubbed data requests gives the exact statistic columns, QC columns, and sidecar entry.
      Also test a failed point and a run with several window sizes.
    - Register test variants of an adapter in a copy of the registry (the `isolated_registry` fixture), so that no other test sees them.
    - Tests that use the network need a pytest marker and a skip condition. Register the marker in `pyproject.toml` (`[tool.pytest.ini_options] markers`).
      `pytest -m "not gee"` and CI do not deselect a new marker, so the skip condition must also apply in CI and in offline runs.
      The specification of the first provider with live tests decides the marker and the skip condition, and then updates `AGENTS.md`, `CONTRIBUTING.md`, and the CI commands.
      Tests without a marker must not use the network.
11. **Documentation.** Update this document: the module map, "Module responsibilities", and the declaration table.
    Describe in `docs/advanced_usage.md` how users register a dataset of the provider: its catalog keys and its credentials.
    Update `README.md` when the setup changes. Add a `CHANGELOG.md` entry under `Added`. Record lasting decisions in `docs/decision_log.md`.

Known limitations:

- Only `src/envoi/configs/ee_catalog.yml` is a built-in catalog (`catalog.BUILTIN_EE_CATALOG`).
  Users register the datasets of a new provider with `update_catalog()`.
  The specification of the first provider of another kind decides where its built-in entries go.
- The web app (`src/envoi_webapp/`) lists every catalog dataset and always uses an Earth Engine key.
  It offers every reducer and both output types for every dataset. So a request that the provider does not support fails when the run starts, not in the form.
- A per-call band list cannot select a source band that has the name of a derived band of any registered adapter.
- `update_catalog()` with a dict does not call `inspect_catalog_entry`.
- The Earth Engine adapter does not check its credentials in `check_dataset` yet. A call without credentials stops when it sets up the first Earth Engine dataset: after the datasets before it used time, and after the outputs of the earlier run configs were written.
  The check needs the credential search of `auth.py`, which PR #75 changes. It is a follow-up after PR #75 merges.
- The contract has no per-point status other than `error`, no retry policy, no job context, and no cancellation.
  The decision log entry "2026-10-13 — Provider contract" gives the reasons.

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
                             "contains": the image that contains the centre day, at any
                             distance (no window), else the latest start before it, else
                             the earliest start after it, in the windows.
                             Null when no own image is in the windows (and, for "contains",
                             none contains the centre day)
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
        image_dates       -> {start, end, fallback} of the selected image (before the masked
                             image), or null for a point without images; fallback is true when
                             the fast path had no own image. A branch of the statistics
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
For a point that used its own fallback image, has a sample date, and is not clamped, it also adds the internal meta key `fallback_image_gap_days` (whole days between the sample day and the image, by the day rule). `extract.py` uses it for the far-image warning; no output writes it.
In raster runs, `_export_single` requests `image_dates` once per tile, in its worker thread, before the download. A failed request makes a failed tile.
The circular aspect image is built for each point from the same image as the main image (`_build_image` with the same date and point), and it is never cached in `_static_image`.

This pipeline lives in `adapters/earth_engine/_image.py` (`_build_image`, `_build_fallback_image`, `_resolve_date_filter_range`, `_find_nearest_timestamp`, `_get_collection_time_bounds`).
`GeeRasterAdapter._get_image()` in `adapter.py` decides which branch applies to each point.

State cached on the adapter instance (`extract()` creates one instance for each (run config, dataset) pair, and uses it for every window size, so none of this state depends on the window size):

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
