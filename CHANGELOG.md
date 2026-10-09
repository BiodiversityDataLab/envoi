# Changelog

## [Unreleased]

### Added
- `init_gee()` accepts the content of a service-account key with the new keyword argument `credentials_json`: the JSON text (`str` or `bytes`) or a `dict`. Use it when the key comes from a secret store or an upload instead of a file. With `credentials_json`, `init_gee()` does not look for a key file. Pass `credentials_path` or `credentials_json`, not both. If the key is not valid, the error message contains no part of the key.
- The web app has a hosted mode for a shared public server, selected with the environment variable `ENVOI_WEBAPP_MODE=hosted`. A public instance on SciLifeLab Serve is in preparation. In hosted mode:
  - Each user uploads their own Earth Engine key. The key stays in the server's memory for the browser session and is never written to disk. Each extraction runs in its own process with that key only.
  - The results come as one ZIP download with the outputs and the run log. There is no output directory on the server to choose.
  - Limits apply to the upload, the number of rows and data products, the window sizes, the run time, the size of the results, and the number of extractions at the same time. The page shows the limits.
  - The server deletes the results 10 minutes after the first download, or at the latest 30 minutes after the extraction ends. A failed, cancelled, or stopped extraction gives no results.

  See "Hosted web app" in the README.
- Web app: a **Cancel** button stops a running extraction.
- Web app: the progress and the results stay on the page when you change other settings, until you click **Clear results** or start a new extraction.
- Web app: each extraction writes a run log (`envoi-run-log-<UTC start time>.txt`) next to the outputs. It lists the warnings and errors that envoi reported, for example points for which Earth Engine returned an error. The page shows the number of warnings, and the last lines of the run log when an extraction fails.
- Web app: each data-product row has its own **Output type** (tabular or raster), so one run can mix tabular statistics and raster tiles.
- A container image definition for the hosted web app on SciLifeLab Serve (`deploy/serve/`), and a GitHub workflow that builds, checks, and publishes the image as `ghcr.io/biodiversitydatalab/envoi-webapp`.

### Changed
- The web app runs each extraction in a separate background process instead of inside the page. In the local web app, the outputs still go to the output directory that you choose, with the same file names, plus the run log.
- Web app: the global **Output type** selector is gone. Choose the output type in each data-product row.
- Web app: the data-product filter **Type** is now called **Category**.
- Web app: a service-account key without `token_uri` is now rejected when you start an extraction, with a message that names the missing field. Keys that Google issues contain this field, so use the unchanged key file from the Google Cloud Console.
- The `webapp` extra now needs Streamlit 1.52 or later. Run `pip install --upgrade "envoi-geospatial[webapp]"` to update an existing installation.
- Web app: **Cancel** in the local web app leaves the files that the extraction wrote before the cancel in the output directory. They can be incomplete, so check or delete them before you use them.

## [0.2.2] — 2026-08-28

### Added
- `catalog_markdown()` renders the dataset catalog as a formatted Markdown document — readable output for notebooks and docs instead of raw dicts.
- Catalog entries accept two new optional keys, both surfaced by `list_datasets("info")`: `display_name`, a human-readable label following the dataset's title in the Earth Engine catalog (e.g. `dem_copernicus_glo30` → "Copernicus DEM GLO-30"), and `category`, the theme the dataset is grouped under. Every built-in dataset now sets both. `display_name` falls back to the dataset key when an entry omits it, so it is always safe to display.
- New `dataset_spec` option `collection_date_policy: mosaic`, for Earth Engine collections that tile space rather than time — one static product cut into geographic tiles, where each image's timestamp records when that tile was acquired. Dates are ignored and the tiles covering each point are mosaicked. Without it, such timestamps are read as a time axis and "most recent image" selects a tile elsewhere on the planet, nulling every statistic.

### Changed
- `dem_copernicus_glo30` now points at `COPERNICUS/DEM/GLO30_2024_1`. Earth Engine has deprecated the original `COPERNICUS/DEM/GLO30`, and using it printed a deprecation warning on every run. Elevation values are unchanged (verified identical to 5 decimal places on sample points); the new release carries real per-tile acquisition dates, so the entry declares `collection_date_policy: mosaic`. **Derived `slope` and `aspect` values shift slightly** (up to ~6° of aspect on tested points) because they are now computed across a mosaic of neighbouring tiles instead of a single tile, which removes tile-edge artefacts. All other datasets were audited against Earth Engine's deprecation registry and are current.

- Raster output now writes a `tiles_manifest.csv` into each dataset's tile folder, mapping every input row to the tile file it produced (`<your id column>`, `tile_filename`, `exported`). Join it onto your input table to find a point's tile, including for points whose tile failed to export. Multi-window runs write one suffixed manifest per window (`tiles_manifest-200m.csv`).

### Fixed
- **Tile filenames are now portable across Windows, macOS and Linux.** Tile names are built from your ID column, which was previously interpolated into the path unchanged. URL-style occurrenceIDs (`http://arctos.database.museum/guid/MSB:Mamm:1`) contain `/`, which scattered tiles into nested directories on the Earth Engine path and raised an error on the local-raster path; `urn:catalog:...` IDs contain `:`, which is illegal on Windows. IDs are now reduced to `[A-Za-z0-9._-]`, with a short hash appended when a rewrite was needed so two IDs can never collide. **IDs that were already portable are untouched, so existing tile filenames are unchanged.**
- Raster mode now rejects blank and duplicated sample IDs up front, with a message naming the offending rows. Previously duplicated IDs silently overwrote each other's tiles and blank IDs produced a `nan-<dataset>.tif`, so a run could return fewer tiles than points with no error. The web app applies the same check when a CSV is uploaded. Tabular mode is unaffected — duplicates are harmless in a column.
- `batch_id` values and `update_catalog()` dataset names are validated as path components, since both become folder names. An unusable one (`my/batch`, `..`, a name with spaces) now raises instead of producing a broken or misplaced output folder.

### Documentation
- New generated dataset reference at `docs/datasets.md`: every built-in dataset grouped by theme, with a summary table plus resolution, temporal coverage, bands, licence, citation, and links per entry. Regenerate it with `python scripts/generate_dataset_docs.py` after editing `ee_catalog.yml`; a test fails if the committed file drifts out of sync.

## [0.2.1] — 2026-07-16

### Changed
- Usability refinements in the graphical user interface.

### Fixed
- Bugs related to dataset selection in the streamlit graphical user interface.

## [0.2.0] — 2026-07-16

### Changed
- **Breaking:** the default `id_column` is now `occurrenceID` (Darwin Core) instead of `gbifID`. Pass `id_column="gbifID"` to keep the old behaviour.
- `extract()` now returns the stats DataFrame directly (instead of a `{batch_id: df}` dict) when a single run config is passed with `output_file_format="dataframe"`. List configs are unchanged.
- A missing date column no longer raises a `UserWarning`; it prints a one-line notice and is still recorded in the metadata sidecar.
- The ability to run envoi from a graphical user interface was added. For the moment, this runs as a local streamlit app on localhost. See the instructions for installing and running it in `Streamlit web app` section of the `README.md`.

### Fixed
- When `input_crs` is set, the output table now keeps the original input-CRS coordinates in the latitude/longitude columns and adds reprojected `<lat>_wgs84`/`<lon>_wgs84` columns, instead of overwriting the coordinates with their WGS84 reprojection.

### Documentation
- `docs/advanced_usage.md`: added a table of contents and an Earth Engine `update_catalog()` example.

## [0.1.1] — 2026-06-09

### Changed
- `list_datasets()` no longer prints to stdout; it only returns its value. Wrap in `print()` if you need printed output.

### Documentation
- Walkthrough notebook: added venv/conda setup and `matplotlib` install instructions, and a QC output cell for the date-aware extraction section.
- README: switched to absolute URLs so links render correctly on PyPI.

### Internal
- Added gitleaks pre-commit hook for secret scanning.

[0.1.1]: https://github.com/BiodiversityDataLab/envoi/compare/v0.1.0...v0.1.1

## [0.1.0] — 2026-05-27

First public release of **envoi** on PyPI.

envoi enriches geographic point data with environmental variables from
Google Earth Engine and/or local rasters through a single, unified interface.
Input tables follow the GBIF / Darwin Core convention (`gbifID`,
`decimalLatitude`, `decimalLongitude`, optional `eventDate`).

[0.1.0]: https://github.com/BiodiversityDataLab/envoi/releases/tag/v0.1.0
