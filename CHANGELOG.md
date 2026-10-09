# Changelog

## [Unreleased]

### Added
- The `date_info` section of the metadata sidecar has a new count, `n_contains_sample_date`: the number of points whose date is inside the time range of the image that envoi used. This is the usual case for Earth Engine datasets with `collection_date_policy: contains`, which had no count in `date_info` before.
- envoi now reports each point whose data request or read failed, for example because of a temporary Earth Engine error or a damaged block in a local raster. The run continues with the other points, and:
  - the QC table gets a column `<dataset>_error_<N>m` (`<dataset>_error_point` for point-only statistics) with the error of each failed point, and an empty value for the other points;
  - in raster runs, the tile manifest gets an `error` column with the same content;
  - each `quality` summary of the metadata sidecar, including `quality.tiles`, has `n_errors` (the number of failed points) and `ids_errors` (their IDs);
  - envoi gives one warning for each dataset and window with failed points, and the sidecar records it in `warnings`.

  The error columns exist only when at least one point of that dataset and window failed. `n_errors` and `ids_errors` are always present (`0` and `[]` when nothing failed). A failed point is still listed in `ids_no_data`, and `quality.tiles.n_failed` still counts every tile that was not exported, for any reason. Before, a failed point looked like a point without data. To repeat only the failed points, run `extract()` again for the IDs in `ids_errors`.

### Changed
- With `input_crs` set, the reprojected `<lat>_wgs84` and `<lon>_wgs84` columns now come directly after the ID, latitude, longitude and (when present) date columns in both the statistics table and the QC table. Before, they were the last columns. The statistics table also no longer contains the `lat_original` and `lon_original` columns, which were internal copies of your latitude and longitude columns. If a script reads those two columns, read your own latitude and longitude columns instead: they hold the same input-CRS coordinates. If a script selects columns by position, select them by name.
- A column named `date` is no longer used as the sample date when your date column has another name. Before, when the input had no `eventDate` column (or no column with the name you passed as `date_column=`), envoi used a `date` column as the sample date without a notice, and renamed it to `eventDate` in the output. Now the `date` column keeps its name and values, and the run behaves as a run without dates. If your dates are in a column named `date`, pass `date_column="date"`.
- When the data request or read of every point of one dataset and window fails, the run now stops with a `RuntimeError` that names the dataset and the window and shows the error of the first point. Before, the run finished, and the outputs had empty values for every point without an error. Outputs that were written before the error, such as the files of an earlier run config in a list, stay on disk. Correct the problem (for example the network connection or your access to the dataset) and run the extraction again.

### Fixed
- Per-class columns of categorical datasets with named bands, such as `lulc_naturallands_2020_classification_class_10_count_200m`, are now `0` (count) or `0.0` (fraction) for points whose window did not contain that class, as other class columns already were. Before, these cells were empty, so an absent class looked like missing data. This affected Earth Engine runs that extract more than one band of a categorical dataset; single-band runs and local rasters were already correct.
- A missing, non-numeric, or infinite latitude or longitude now stops the run before any extraction, with a `ValueError` that names your coordinate columns and lists the IDs of the affected rows. Before, text gave an unclear `TypeError`, infinity gave an error that said the coordinates were not in WGS84, and a missing value failed later without naming the row, or gave empty statistics. Coordinates stored as text that holds a number, such as `"59.85"`, are now accepted.
- An extra input column named `id`, `lat`, `lon`, `date`, `lat_original`, or `lon_original` no longer breaks the run. For example, Darwin Core files often have an `id` column next to `occurrenceID`. Before, an extra `id` column gave two `occurrenceID` columns in the statistics table and an unclear error in raster mode, and the other names gave unclear errors. Now such a column keeps its name and values in the statistics table, and envoi does not use it. This also fixes web app uploads that have such a column.
- Two run configs in one list with the same `batch_id` now raise a `ValueError` before any output is written. This includes a run config without `batch_id`, which gets the name `output<N>` (N is its position in the list, starting at 1). Before, the later output overwrote the files of the earlier one, and the returned dict held only one of them. envoi now also checks every run config in the list before the first one runs, so an invalid setting in a later run config stops the call before any output is written or any Earth Engine quota is used.
- `resample_m` and `min_coverage_pct` must be finite numbers. Text (such as `"30"`), `true`/`false`, `null`, NaN, and infinity now raise a `ValueError` that names the output. Before, some of these values gave an unclear `TypeError`, and others were accepted with a wrong result: `true` was read as `1`, and `min_coverage_pct: nan` never flagged low coverage. NumPy numbers such as `np.float32(30)` are accepted, and the metadata sidecar records them as plain numbers.
- Dates written `YYYY/MM/DD`, such as `2021/06/15`, are now read as that date. Before, envoi kept only the year and used January 1 of that year, with a warning such as `Date '2021' interpreted as 2021-01-01`.
- Other date values with `/`, such as `15/06/2021`, `06/15/2021`, `06/2021`, or `2021/6/5`, now stop the run before any extraction, with a `ValueError` that names your date column and lists the values. Before, envoi kept only the part before the first `/`, which gave a wrong date (for example, `2021/6/5` became January 1, 2021) or an unclear error. Write these dates as `YYYY-MM-DD`. A value with exactly one `/` whose start is an ISO 8601 date or date-time, such as `2021-06-15/2021-06-20`, is still read as an interval (`start/end`), and envoi uses its start day.
- For Earth Engine datasets with `collection_date_policy: contains`, a sample date on the start of the first image, or inside the time range of the last image, is no longer marked as clamped. Before, the QC columns `<dataset>_date_clamped_<N>m` and `<dataset>_date_source_<N>m` showed `True` and `clamped_to_nearest` for these dates, and the sidecar `date_info` counted them as clamped, although envoi used the correct image. The statistics do not change. A date is now clamped only when it is before the start of the first image, or on or after the end of the last image's time range. This time range always includes the whole start day of the last image.
- The `date_info` section of the metadata sidecar is no longer missing when the request for the first point of an Earth Engine dataset failed. Before, envoi looked for date information only in the first point.
- Custom Earth Engine datasets with `collection_date_policy: contains` whose images start and end on the same day (for example, images whose `system:time_end` is equal to `system:time_start`) now get statistics and tiles. Before, envoi searched for such an image in an empty date range, so the statistics were empty and no tile was exported. The built-in `contains` datasets were not affected, because their images cover 8 days or more.
- Earth Engine collections in which two images have the same start time and different end times no longer break date selection. envoi now keeps one time range per start time, with the latest end time. Before, with `collection_date_policy: nearest` (the default) and a date column, every point of such a dataset failed: the statistics were empty, and raster runs stopped with an unclear error.
- Earth Engine raster runs whose tiles are too large for Earth Engine's download limit (about 32 MB for each tile) now stop with a `ValueError` before any tile is downloaded. The error tells you to reduce `window_size_m`, increase `resample_m`, or select fewer bands. Before, every tile failed with a warning, and the run ended without tiles and without an error.
- When Earth Engine cannot return the image dates of a collection (for example, because of a network error or a timeout), the run now stops with a `RuntimeError` that names the collection. Run the extraction again. Before, envoi logged a warning and continued without the image dates, so it could not select the image for each sample date: points without a date got a mosaic of the most recent images, points with a date got an image from within one day of the sample date (or empty statistics when there was none), and the outputs did not show this.
- A temporary error in the request for the band names of an Earth Engine dataset no longer makes every later point of that dataset fail. This request is made only for datasets whose bands are not set in the catalog or the run config. Before, envoi kept the placeholder band name `value` after one failed request, and every later point of the dataset got empty statistics. Now envoi asks again for the next point. In raster runs, envoi needs the band count before the first download, so a band-name request that fails twice now stops the run.
- envoi no longer replaces an Earth Engine session that is already initialized, for example a session that you started with `ee.Initialize()` or `envoi.init_gee()`. Before, envoi sent a test request when it set up the first Earth Engine dataset. When that request failed, for example because of a temporary network error, envoi initialized Earth Engine again with the credentials file that `init_gee()` finds, which could replace your credentials and Cloud project, or stop with "credentials not found" and hide the real error. Now envoi initializes Earth Engine only when it is not initialized, and sends no test request.
- A read error in a local raster, for example a damaged block in the file, no longer stops the whole run. The point is reported as a failed point (see "Added"), and the other points get their values. A failed pixel read for the `point` statistic no longer looks like a point outside the raster. When a point needs both a window read and a pixel read (window statistics together with `point`), an error in either read now makes every statistic of that point empty.
- In Earth Engine runs that extract more than one band, a point whose request failed no longer adds statistic columns without a band name, such as `<dataset>_mean_<N>m`, which were empty for every point. The failed point now has empty values in the usual `<dataset>_<band>_<statistic>_<N>m` columns. Also, the per-class columns (`class_count`, `class_fraction`) of a failed point are now empty instead of `0`, because the classes at that point are unknown.

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
