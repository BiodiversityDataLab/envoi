# Requirements

## Goal (one sentence)
A Python lib/CLI that enriches points (`lat, lon, [date]`) with environmental predictors from local rasters and one remote source, returning a model-ready table with QC and provenance.

## In scope
- LocalRaster + one Remote adapter (MVP)
- Predictors v1: elevation/slope, land-cover majority, one climate variable
- Optional country-based grid generation
- QC + provenance
- Parquet/CSV output

## Out of scope
- Heavy ML; large UI; many remote sources

## Inputs
- Table: `id, lat, lon[, date]` (WGS84)
- Catalog (YAML): datasets, reducers, resolution, temporal rule

## Outputs
- Gold table (Parquet/CSV) with feature columns + provenance
- Optional tiles for QA
- QC summary

## Constraints
- CRS: normalize to EPSG:4326
- Carry per-predictor license/citation
