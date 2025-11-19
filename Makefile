SHELL := /bin/bash

.PHONY: install test sample-run sample-groups fmt lint

PREDICTORS ?= dem_elev

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

test:
	. .venv/bin/activate && pytest -q

fmt:
	. .venv/bin/activate && black src tests

lint:
	. .venv/bin/activate && ruff check src tests

sample-run:
	mkdir -p out
	. .venv/bin/activate && biodata enrich --in data/points_sample.csv --out out/gold.parquet --catalog configs/catalog.yml --predictors $(PREDICTORS) --window_m 500 --temporal nearest_month
	@echo "Sample flat run completed."

sample-groups:
	mkdir -p out
	. .venv/bin/activate && biodata enrich --in data/points_sample.csv --out out --catalog configs/catalog.yml --groups configs/run.yml --window_m 100 --temporal nearest_month
	@echo "Sample groups run completed."
