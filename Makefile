SHELL := /bin/bash

.PHONY: install test sample-run fmt lint

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

test:
	. .venv/bin/activate && pytest -q

fmt:
	. .venv/bin/activate && black src tests

lint:
	. .venv/bin/activate && ruff check src tests

sample-run:
	mkdir -p out
	. .venv/bin/activate && biodata enrich --in data/points_sample.csv --out out/gold.parquet --catalog configs/catalog.yml --predictors dem_elev,cop_lc_2021 --window_m 500 --temporal nearest_month
	@echo "Sample run completed. (Stub predictors for now)"
