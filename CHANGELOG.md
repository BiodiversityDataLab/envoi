# Changelog

All notable changes to this project are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking)
- Default input column names now follow the GBIF / Darwin Core convention:
  `id` → `gbifID`, `lat` → `decimalLatitude`, `lon` → `decimalLongitude`,
  `date` → `eventDate`. The output tables preserve the user's chosen names, so
  callers that relied on the old defaults can restore them by passing
  `id_column="id"`, `latitude_column="lat"`, `longitude_column="lon"`,
  `date_column="date"` to `extract()`.

## [0.1.0] — Unreleased

First public release on PyPI.

### Added
- `extract(df, config)` — unified entry point for extracting environmental
  data at sample points, supporting both Google Earth Engine and local rasters
  through a single interface.
- Built-in GEE dataset catalog (`ee_catalog.yml`) with auto-detection of asset
  type, native scale, and projection.
- Local raster adapter with dynamic UTM reprojection and rasterio-backed
  auto-detection of CRS, resolution, nodata, and band count.
- Per-dataset reducers: `mean`, `std`, `min`, `max`, `quantile`, `class_count`,
  `class_fraction`.
- Per-band and per-point coverage QC with sidecar JSON metadata.
- `update_catalog()` for registering custom datasets at runtime (from YAML or
  dict).
- `list_datasets()` and `list_reducers()` helpers.
- Tabular and raster output modes (`parquet`, `csv`, GeoTIFF tiles).
- Automatic date handling for GEE ImageCollections (`nearest` and `contains`
  policies, with date clamping for out-of-range points).
- Service-account authentication helper (`init_gee`) reading from
  `credentials/ee_credentials.json`.

[0.1.0]: https://github.com/BiodiversityDataLab/envoi/releases/tag/v0.1.0
