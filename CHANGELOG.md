# Changelog

## [Unreleased]

### Added
- `catalog_markdown()` renders the dataset catalog as a formatted Markdown document — readable output for notebooks and docs instead of raw dicts.
- Catalog entries accept two new optional keys, both surfaced by `list_datasets("info")`: `display_name`, a human-readable label following the dataset's title in the Earth Engine catalog (e.g. `dem_copernicus_glo30` → "Copernicus DEM GLO-30"), and `category`, the theme the dataset is grouped under. Every built-in dataset now sets both. `display_name` falls back to the dataset key when an entry omits it, so it is always safe to display.

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
