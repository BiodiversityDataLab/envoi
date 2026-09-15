# Envoi architecture overview

**Status:** working review document  
**Reviewed:** 2026-09-14  
**Scope:** `src/envoi`, `src/envoi_webapp`, configuration, outputs, and tests

## Executive overview

Envoi is a catalog-driven, batch-oriented environmental data extraction package. Its public centre is `envoi.extract()`: callers provide a pandas `DataFrame` of sample locations and a run configuration, and Envoi selects datasets from its catalog, dispatches each dataset to a source adapter, and produces tabular statistics or raster tiles together with quality-control and provenance metadata.

The package currently has two data backends:

- `LocalRasterAdapter`, which reads local raster files with Rasterio and performs reductions with NumPy-backed reducer functions.
- `GeeRasterAdapter`, which constructs and reduces Google Earth Engine images remotely and downloads raster tiles through Earth Engine download URLs.

The adapters are separated at the source-access level, while validation, output assembly, quality control, naming, metadata, and progress reporting are shared. This is the main architectural seam for adding future providers.

The current implementation is not a plugin architecture yet. Adapter registration is internal and convention-based, catalog entries are loosely structured dictionaries, and generic configuration code contains some Earth Engine-specific rules. These limitations are analysed in [future-development-review.md](future-development-review.md).

## System context

```text
User Python code                         Streamlit application
       |                                         |
       |                                  builds the same run
       |                                  config and calls extract()
       +--------------------+--------------------+
                            |
                            v
                  envoi.extract(df, config)
                            |
             +--------------+--------------+
             |                             |
      input validation                config parsing
      and reprojection                and normalization
             |                             |
             +--------------+--------------+
                            |
                      dataset catalog
                            |
                 data_source -> adapter class
                            |
              +-------------+--------------+
              |                            |
      Google Earth Engine             local raster
              |                            |
              +-------------+--------------+
                            |
             statistics / point metadata / tiles
                            |
          +-----------------+------------------+
          |                 |                  |
     tabular assembly    quality control    metadata
          |                 |                  |
          +-----------------+------------------+
                            |
                CSV, Parquet, DataFrame,
                GeoTIFFs, manifests, JSON
```

## Public API

`src/envoi/__init__.py` exports:

| Symbol | Purpose |
|---|---|
| `extract` | Stable user-facing extraction entry point. |
| `init_gee` | Initialize the Earth Engine Python client from a service-account JSON file. |
| `list_datasets` | Inspect the built-in catalog plus process-registered datasets. |
| `update_catalog` | Add or replace datasets in the process-global user catalog. |
| `reset_catalog` | Clear datasets added through `update_catalog`. |
| `catalog_markdown` | Render catalog documentation. |
| `list_reducers` | List NumPy-backed reducer names. It does not include the adapter-level `point` operation. |
| `ProgressEvent` | Structured progress notification passed to callbacks. |

`extract()` accepts a `DataFrame`, a configuration dictionary/list/YAML path, an output directory, input CRS and column-name options, metadata and quiet-mode switches, and an optional progress callback. It usually returns a mapping from output identifiers to paths or DataFrames. A single tabular `dataframe` run returns the DataFrame directly as a convenience.

## End-to-end execution flow

### 1. Input normalization

`extract.py` copies the input `DataFrame`, validates the requested identifier and coordinate columns, and renames them internally to `id`, `lat`, `lon`, and `date`. User-facing names are restored before tabular output is returned or written.

`_input_validation.py` then:

1. Parses GBIF-style and ISO 8601 date values.
2. Reduces intervals and timestamps to day precision when necessary.
3. Drops rows that cannot supply usable dates where applicable.
4. Reprojects non-WGS84 coordinates to EPSG:4326.
5. Returns warnings for inclusion in output metadata.

Raster output also validates that sample identifiers are present and unique because they become tile filenames.

### 2. Defaults and catalogs

`catalog.py` loads:

- `configs/defaults.yml`, containing runtime defaults such as window size, coverage threshold, output format, decimal precision, and Earth Engine worker count.
- `configs/ee_catalog.yml`, containing the built-in datasets.
- Any datasets previously registered through `update_catalog()`.

Later catalog layers replace earlier entries by dataset name. Local raster entries are inspected with Rasterio when possible to infer CRS, resolution, nodata, raster type, and band count.

The catalog's `data_source` value is the dispatch key into the adapter registry.

### 3. Configuration parsing

`_config_parsing.py` normalizes one configuration or a list of configurations into `RunSettings` instances. A `RunSettings` object holds:

- `batch_id`
- selected datasets and per-call band overrides
- `output_type`
- `output_file_format`
- one or more window sizes
- minimum coverage threshold
- normalized reducer selections
- raster resampling resolution
- original user forms needed for metadata round-tripping

Dataset selections can be written as a plain name or as a single-key dictionary containing band overrides. Reducer lists can be flat or split between continuous and categorical data.

### 4. Orchestration and dispatch

For each run configuration, `extract()` loops sequentially through:

```text
run configuration
    -> selected dataset
        -> requested window size
```

For every dataset/window pair it:

1. Shallow-merges the catalog specification with per-call band overrides.
2. Looks up the adapter class using the specification's `data_source`.
3. Constructs the adapter inside a context manager.
4. Invokes the tabular or raster method.
5. Builds dataset metadata.
6. Closes the adapter before processing the next pair.

An important lifecycle detail is that the adapter is instantiated **once per dataset/window pair**, not once for the whole `extract()` call. Adapter caches therefore do not survive across multiple window sizes for the same dataset.

### 5. Tabular mode

In tabular mode the orchestrator calls:

```python
adapter.fetch_stats_batch(
    lats,
    lons,
    window_size,
    reducer_names,
    dates=dates,
    progress_callback=callback,
)
```

The effective internal return format is:

```python
list[tuple[dict, dict]]
# one (statistics, point_metadata) pair per input row
```

`_output_assembly.py` discovers result columns from the statistic dictionary keys and adds dataset and window suffixes. `qc.py` expects common metadata fields including `in_extent`, `n_pixels`, `had_nodata`, and `coverage_pct`; it adds optional date, region CRS, and per-band coverage fields where present.

The main table and QC table are split after all datasets have been processed. Depending on configuration, the main table is returned as a DataFrame or written as CSV/Parquet. The QC table is written when metadata output is enabled.

### 6. Raster mode

In raster mode the orchestrator calls `adapter.export_tiles()`. Each adapter writes one GeoTIFF per sample where possible and returns parallel lists of output paths and point metadata. Envoi also writes a manifest that maps original sample identifiers to the sanitized tile filenames.

Raster results are organized under:

```text
<output_dir>/<batch_id>/<dataset>/
```

When several windows are requested, window suffixes prevent collisions within the dataset directory.

### 7. Metadata and return values

`metadata.py` writes a JSON sidecar containing:

- package version and run timestamp
- resolved run configuration
- resolved dataset selections and bands
- adapter-produced dataset information
- quality summaries
- date-selection and tile-export summaries where applicable
- selected warnings from input/configuration processing

The return mapping contains the primary stats file/DataFrame in tabular mode or dataset tile directories in raster mode. QC and metadata paths are not returned as first-class result objects; callers infer their locations from naming conventions.

## Module map

### Core orchestration and shared services

| Module | Responsibility |
|---|---|
| `extract.py` | Public orchestration: input normalization, catalog/config resolution, dataset/window loops, adapter dispatch, and output coordination. |
| `_input_validation.py` | Required columns, date parsing, coordinate validation, CRS reprojection, and sample-ID validation. |
| `_config_parsing.py` | Configuration loading, normalization, validation, reducer selection, and `RunSettings`. |
| `catalog.py` | Catalog loading/merging, local raster inspection, defaults caching, and process-wide catalog updates. |
| `catalog_docs.py` | Markdown rendering for dataset catalog documentation. |
| `_output_assembly.py` | Statistics-column construction, rounding, column-name restoration, tabular file writing, and resolved dataset metadata. |
| `qc.py` | Per-point QC columns, coverage summaries, warnings, and stats/QC table separation. |
| `metadata.py` | JSON sidecar generation and shared dataset/tile summary helpers. |
| `reducers.py` | NumPy reducer registry and reducer/data-type compatibility warnings. |
| `progress.py` | `ProgressEvent`, callback types, and progress emission helpers. |
| `geo.py` | WGS84/UTM selection and tile CRS-zone summaries. |
| `_filenames.py` | Portable path-component validation, sanitization, length limits, and tile manifest naming. |
| `auth.py` | Earth Engine initialization and credential-file discovery. |

### Adapter modules

| Module | Responsibility |
|---|---|
| `adapters/__init__.py` | Internal mapping from `data_source` names to adapter classes. Eagerly imports built-ins so they register themselves. |
| `adapters/base.py` | Context-manager lifecycle and partial method conventions shared by adapters. It is not an abstract or runtime-validated interface. |
| `adapters/local_adapter.py` | Rasterio-backed reads, point/window extraction, NumPy reductions, nodata handling, reprojection/resampling, tile writing, and local metadata. |
| `adapters/earth_engine/adapter.py` | Earth Engine adapter, dataset initialization, per-point concurrency, statistics orchestration, tile export, caches, and metadata. |
| `adapters/earth_engine/_image.py` | Earth Engine initialization checks, shared HTTP-pool patching, image/collection selection, spatial/date filtering, grid snapping, and derived bands. |
| `adapters/earth_engine/_reducers.py` | Earth Engine reducer translation, combined reducers, server-result parsing, histograms, point sampling, and coverage summaries. |
| `adapters/earth_engine/_tiles.py` | Synchronous tile-size guard, download URL generation, HTTP timeouts, and transient-error retry/backoff. |

### Web application

| Module | Responsibility |
|---|---|
| `envoi_webapp/app.py` | Streamlit page composition, widgets, dataset-selection state, form validation display, progress rendering, local directory chooser, and CLI launcher. |
| `envoi_webapp/helpers.py` | CSV/CRS validation, reducer filtering, run-config construction, service-account validation/redaction, temporary credential files, output-directory validation, and the direct call into `extract()`. |

The Streamlit application is presently a local frontend, not a separate service tier. It executes `extract()` synchronously in the Streamlit process and writes directly to its filesystem.

## Adapter boundary as implemented

The intended common operations are:

| Operation | Used by the orchestrator | Declared on `BaseAdapter` |
|---|---:|---:|
| context-manager lifecycle | yes | yes |
| `fetch_stats_batch` | tabular mode | yes |
| `export_tiles` | raster mode | **no** |
| `build_dataset_meta` | both modes | yes |
| `fetch_values` / `fetch_batch` | not used by `extract()` | yes |

This means the effective contract differs from the documented base class. In particular, `GeeRasterAdapter` does not supply the base class's raw `fetch_values()` behaviour, while both production adapters implement the undeclared `export_tiles()` method needed by `extract()`.

Adapters receive only the merged dataset specification. There is no job-scoped context object carrying credentials, network clients, artifact storage, cancellation, logging, or concurrency policy.

## Backend-specific execution

### Local raster path

The local adapter opens the configured raster and caches transforms and band information on the adapter instance. For each point it converts WGS84 coordinates to the source grid, reads a point or window, masks nodata, and runs Python reducer functions. Raster export uses Rasterio reprojection when a target resolution is requested and writes tiles locally.

Local multiband outputs use generic identifiers such as `b1`, `b2`, and `b3` unless higher-level metadata supplies additional meaning.

### Earth Engine path

The Earth Engine adapter initializes an image or collection from the catalog path. Collection processing may filter spatially, select imagery by sample date, mosaic tiled assets, derive bands such as slope/aspect, and select final bands.

Tabular reductions are built server-side. Several reducers are combined so a point normally requires one evaluated Earth Engine expression. The adapter then resolves points concurrently with a `ThreadPoolExecutor`.

Raster mode constructs a UTM-region tile for each point, obtains an Earth Engine download URL, and downloads it with retry/backoff. Each tile uses the UTM zone associated with its centre point.

The adapter caches projection, scale, band, and collection timestamp information on its instance. Because instances are scoped to one dataset/window pair, those caches are shared across that pair's worker threads but not across other window sizes.

## State and concurrency

The following state is process-global:

- registered adapter classes in `adapters._REG`
- datasets added through `catalog.update_catalog()`
- cached defaults and built-in dataset names
- Earth Engine Python SDK state
- `_gee_initialized` in the Earth Engine image helper
- the Earth Engine HTTP session modified by the connection-pool patch

Streamlit widget/session state is per browser connection, but the process globals and server filesystem above are shared between sessions running in the same process.

The outer extraction loops are sequential. Earth Engine parallelizes points within a dataset/window pair; the local adapter primarily performs local sequential/batch work. There is no package-level scheduler, global worker budget, cancellation token, or request deadline.

## Important dependency directions

The intended dependency direction is:

```text
public API / web UI
        -> orchestration
        -> shared validation and catalog services
        -> adapter registry
        -> concrete adapters
        -> external systems
```

There are currently two notable reverse or cross-layer dependencies:

1. Generic `_config_parsing.py` imports Earth Engine-derived-band knowledge.
2. The webapp imports private core validation helpers and internal reducer constants.

These couplings are central subjects of the future-development review.

## Existing strengths to preserve

- One recognizable public extraction entry point.
- Catalog-driven dataset selection rather than provider details in user code.
- Source access is already separated into local and Earth Engine implementations.
- The orchestrator calls batch methods, allowing providers to optimize internally.
- Context-manager cleanup exists for adapter-held resources.
- Input validation, quality control, metadata, naming, and output assembly are substantially shared.
- Progress callbacks are structured and independent of terminal progress bars.
- Metadata records resolved datasets, configuration, quality summaries, and package version.
- Local and Earth Engine behaviour have substantial automated test coverage.

## Test organization

The suite covers configuration parsing, input validation, local extraction, reducers, QC, filenames, progress, catalog behaviour, metadata, webapp helpers, and Streamlit helper behaviour. Credentialed Earth Engine integration tests are marked separately and skip when credentials are unavailable.

At the review point, 314 non-live tests passed and 46 credentialed Earth Engine tests were deselected. See the companion review for identified contract, concurrency, deployment, and test gaps.
