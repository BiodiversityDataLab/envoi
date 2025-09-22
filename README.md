# biodata-enricher

A Python library and CLI to enrich point observations (`lat`, `lon`, optional `date`) with environmental predictors from **local rasters** and **remote sources** (e.g., GEE/STAC), returning a **model-ready** table (Parquet/CSV) with **QC** and **provenance**.

> MVP scope: 2 sources (LocalRaster + one remote), 3–4 predictor families (elevation/slope, land-cover majority, one climate var), basic reducers (mean/std/majority), Parquet output, QC.

## Quick start

```bash
# 1) Create & activate venv, install
make install

# 2) Try the sample end-to-end (uses a stub enrich for now)
make sample-run

# 3) Run tests & linters
make test
```

### CLI example
```bash
biodata enrich --in data/points_sample.csv   --out out/gold.parquet   --catalog configs/catalog.yml   --predictors dem_elev,cop_lc_2021,era5_temp_month   --window_m 500   --temporal nearest_month
```

## Project layout
```
src/biodata/            # library code
  adapters/             # data source adapters (LocalRaster, GEE/STAC)
  enrich.py             # main API
  reducers.py           # numeric/categorical/temporal reducers
  geometry.py           # CRS normalization, projections
  grid.py               # country-based grid generation
  qc.py                 # schema/value checks, coverage
  writers.py            # Parquet/CSV writers + metadata
  config.py             # load/validate catalog + config
  cli.py                # command-line interface
configs/                # catalog.yml and other configs
docs/                   # requirements, plan, predictors, QC spec, minutes
tests/                  # unit tests
data/                   # sample data (small only; big data is NOT committed)
.github/workflows/      # CI config
```

## MVP tasks
- [ ] Confirm predictor list v1 & output spec
- [ ] Implement LocalRasterAdapter (DEM + land cover)
- [ ] CRS normalize + optional grid builder
- [ ] Reducers (mean/std/majority) + temporal (nearest_month/monthly_mean)
- [ ] GEEAdapter for one remote dataset
- [ ] QC + provenance + Parquet writer
- [ ] Docs + architecture diagram

## License
MIT for code. Data inherits original sources' licenses; carry per-predictor license/citation.
