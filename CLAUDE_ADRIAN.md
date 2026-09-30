# CLAUDE.md — envoi

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

## Git conventions

- **Do not add co-authorship or attribution trailers.** No `Co-Authored-By:` line in commit
  messages, and no "Generated with Claude Code" footer in pull request bodies or descriptions.
  This overrides the default Claude Code behaviour of appending them.

---

## What this project is

**envoi** is a Python package that enriches geographic point data
with environmental datasets. The input is a table of sample points
(`gbifID`, `decimalLatitude`, `decimalLongitude`, optionally `eventDate` —
following the GBIF / Darwin Core convention); the output is either that same
table with appended environmental columns or images of environmental datasets,
ready for spatial ecological modeling or similar analyses. The input column
names can be overridden via the `*_column` parameters on `extract()`.

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
    "datasets": ["dem_copernicus_glo30"],
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
from envoi import update_catalog
update_catalog("my_catalog.yml")          # from a YAML file
update_catalog({"datasets": {...}})       # or a dict
```

## Catalog design

The built-in catalog (`src/envoi/configs/ee_catalog.yml`) is bundled with the
package and loaded automatically. Only `data_source` and `path` are required —
everything else is auto-detected or optional.

```yaml
datasets:
  dem_copernicus_glo30:
    data_source: earth_engine
    path: COPERNICUS/DEM/GLO30   # asset type auto-detected via ee.data.getAsset()
```

**`display_name`** (optional): human-readable label for a dataset, following its
title in the GEE catalog (`dem_copernicus_glo30` → `"Copernicus DEM GLO-30"`).
Used as the heading in `docs/datasets.md` and intended as the label in the web
app's dataset dropdown. `list_datasets("info")` always returns it, falling back
to the dataset key when an entry omits it.

**`category`** (optional): the theme a dataset is filed under in the generated
reference `docs/datasets.md` (e.g. `Terrain`, `Climate`).

Both are presentation metadata — neither affects extraction.

**GEE auto-detection:** asset type (IMAGE vs IMAGE_COLLECTION) is resolved via
`ee.data.getAsset()` — no `asset_type` key needed in catalog.

**Local auto-detection:** CRS, resolution, nodata, and band count are read from the
file via rasterio (`_inspect_raster()`).

**`dataset_spec` block** (optional, GEE only): per-dataset overrides that
the GEE adapter consults when the default behaviour is wrong. The keys
currently honoured are:

- `native_scale_m` — manual override when GEE's `nominalScale()` returns
  the EPSG:4326 default (~111 km) for assets whose native projection
  metadata is missing (e.g. some Landsat composites).
- `use_utm_zone` — when `True`, the adapter filters the collection by
  `UTM_ZONE` so a per-point query picks the tile covering that point.
  Only set this for collections whose images actually carry a `UTM_ZONE`
  property — `aef_satellite_embeddings` is the only built-in that does.
  On a collection without it the filter empties the collection, so check
  first (`ee.Image(collection.first()).getInfo()["properties"]`).
  Tiled collections that are already disambiguated by `filterBounds`
  (e.g. the GLO-30 DEM) do not need it.
- `collection_date_policy` — `"nearest"` (default), `"contains"`, or
  `"mosaic"`. Controls how an ImageCollection's per-point image is
  selected from a sample date.
  - `"nearest"` / `"contains"` — genuine time series. Pick the image with
    the closest timestamp, or the one whose interval covers the date.
  - `"mosaic"` — the collection tiles **space, not time**: it is one
    static product cut into geographic tiles, and each image's timestamp
    records when that tile was acquired. Dates are ignored entirely; the
    tiles covering each point are spatially filtered and mosaicked.
    Set this whenever a "static" product turns out to carry many distinct
    `system:time_start` values (check with
    `collection.aggregate_array("system:time_start").distinct()`) — read
    as a time axis, "most recent image" selects a tile on the other side
    of the world and every statistic comes back null. `dem_copernicus_glo30`
    is the built-in example. The mosaic gets the collection's native
    projection stamped back on via `setDefaultProjection`, because
    `mosaic()` drops projection metadata and `ee.Terrain.slope`/`aspect`
    silently return null without it.

```yaml
aef_satellite_embeddings:
  data_source: earth_engine
  path: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
  data_type: continuous
  dataset_spec:
    use_utm_zone: true
```

**Automatic date handling for ImageCollections:** when the input DataFrame
has an `eventDate` column, the adapter fetches the collection's available
timestamps and selects the single nearest image to each point's date.
Out-of-range dates are clamped to the closest boundary. When no `eventDate`
column is provided, the most recent image is used. Date decisions are
recorded in the output metadata.

---

## Files overview

```
src/envoi/
    extract.py               ← main entry point; orchestrates per-dataset/window loop
    catalog.py               ← catalog loading, update_catalog(), reset_catalog(), list_datasets()
    catalog_docs.py          ← renders catalog entries as Markdown; catalog_markdown() + docs/datasets.md generator
    _config_parsing.py       ← run config validation and normalization (RunSettings dataclass)
    _input_validation.py     ← DataFrame column validation, sample-ID checks, CRS reprojection, date parsing
    _filenames.py            ← portable filename sanitization shared by both adapters
    _output_assembly.py      ← tabular post-processing: column appending, rounding, file writing
    metadata.py              ← per-dataset metadata assembly and JSON sidecar writer
    auth.py                  ← GEE authentication from service account JSON
    reducers.py              ← Python-side reducer registry (mean, std, quantiles, ...)
    geo.py                   ← CRS / geometry helpers (UTM zone resolution) shared by adapters
    qc.py                    ← coverage QC flags
    _version.py              ← package version string
    configs/
        ee_catalog.yml       ← built-in GEE dataset registry (bundled with package)
        defaults.yml         ← project-wide setting defaults
    adapters/
        base.py              ← BaseAdapter abstract class
        earth_engine/
            adapter.py       ← GeeRasterAdapter (fetch_stats_batch, export_tiles)
            _image.py        ← ee.Image construction (asset resolution, date selection)
            _reducers.py     ← server-side reduceRegion assembly and result parsing
            _tiles.py        ← GeoTIFF tile export (size guards, download, file writing)
        local_adapter.py     ← LocalRasterAdapter (rasterio-based, dynamic UTM per point)

examples/
    run.yml                  ← example run config
    walkthrough.ipynb        ← end-to-end notebook walkthrough

docs/
    datasets.md              ← GENERATED dataset reference (rebuild, don't hand-edit)

scripts/
    generate_dataset_docs.py ← regenerates docs/datasets.md from ee_catalog.yml

credentials/
    ee_credentials.json      ← GEE service account key (gitignored)
```

---

## Output naming rules

envoi must run on Windows, macOS and Linux, so anything that becomes a file or
folder name is held to the strictest common denominator: `[A-Za-z0-9._-]` only.
`_filenames.py` owns this, and there are two different policies depending on
where the value came from:

- **Sample IDs** come from the user's data file, so they are **sanitized**, never
  rejected. Real GBIF occurrenceIDs contain `/` (URL-style IDs) and `:`
  (`urn:catalog:...`), both of which break file writing. A short hash of the
  original is appended whenever a rewrite happened, so two IDs can never
  collapse onto one filename. An ID that is already portable is passed through
  untouched — existing users' tile filenames must stay stable.
- **`batch_id` and dataset catalog keys** are hand-authored config, so an
  unusable one is **rejected** with an error telling the author to fix it. This
  also blocks a `..` batch_id from escaping the output directory.

Because a sanitized filename is not always reconstructible from the ID, every
raster export writes a `tiles_manifest.csv` next to the tiles mapping each input
row to its tile file. That manifest is the supported way to join tiles back to
input rows — don't rebuild filenames by hand.

Raster mode also requires every sample ID to be present and unique, since it
names one file per point. Blank or duplicated IDs raise before any adapter is
built, rather than silently costing the user tiles.

---
