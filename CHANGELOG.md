# Changelog

## [Unreleased]

### Changed
- With `input_crs` set, the reprojected `<lat>_wgs84` and `<lon>_wgs84` columns now come directly after the ID, latitude, longitude and (when present) date columns in both the statistics table and the QC table. Before, they were the last columns. The statistics table also no longer contains the `lat_original` and `lon_original` columns, which were internal copies of your latitude and longitude columns. If a script reads those two columns, read your own latitude and longitude columns instead: they hold the same input-CRS coordinates. If a script selects columns by position, select them by name.
- A column named `date` is no longer used as the sample date when your date column has another name. Before, when the input had no `eventDate` column (or no column with the name you passed as `date_column=`), envoi used a `date` column as the sample date without a notice, and renamed it to `eventDate` in the output. Now the `date` column keeps its name and values, and the run behaves as a run without dates. If your dates are in a column named `date`, pass `date_column="date"`.

### Fixed
- Per-class columns of categorical datasets with named bands, such as `lulc_naturallands_2020_classification_class_10_count_200m`, are now `0` (count) or `0.0` (fraction) for points whose window did not contain that class, as other class columns already were. Before, these cells were empty, so an absent class looked like missing data. This affected Earth Engine runs that extract more than one band of a categorical dataset; single-band runs and local rasters were already correct.
- A missing, non-numeric, or infinite latitude or longitude now stops the run before any extraction, with a `ValueError` that names your coordinate columns and lists the IDs of the affected rows. Before, text gave an unclear `TypeError`, infinity gave an error that said the coordinates were not in WGS84, and a missing value failed later without naming the row, or gave empty statistics. Coordinates stored as text that holds a number, such as `"59.85"`, are now accepted.
- An extra input column named `id`, `lat`, `lon`, `date`, `lat_original`, or `lon_original` no longer breaks the run. For example, Darwin Core files often have an `id` column next to `occurrenceID`. Before, an extra `id` column gave two `occurrenceID` columns in the statistics table and an unclear error in raster mode, and the other names gave unclear errors. Now such a column keeps its name and values in the statistics table, and envoi does not use it. This also fixes web app uploads that have such a column.
- Two run configs in one list with the same `batch_id` now raise a `ValueError` before any output is written. This includes a run config without `batch_id`, which gets the name `output<N>` (N is its position in the list, starting at 1). Before, the later output overwrote the files of the earlier one, and the returned dict held only one of them. envoi now also checks every run config in the list before the first one runs, so an invalid setting in a later run config stops the call before any output is written or any Earth Engine quota is used.
- `resample_m` and `min_coverage_pct` must be finite numbers. Text (such as `"30"`), `true`/`false`, `null`, NaN, and infinity now raise a `ValueError` that names the output. Before, some of these values gave an unclear `TypeError`, and others were accepted with a wrong result: `true` was read as `1`, and `min_coverage_pct: nan` never flagged low coverage. NumPy numbers such as `np.float32(30)` are accepted, and the metadata sidecar records them as plain numbers.

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
