# Changelog

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
