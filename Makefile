SHELL := /bin/bash

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: install ensure-venv test fmt lint sample-run sample-groups rerun clean

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PIP) install -e ".[dev]"

ensure-venv:
	@test -x $(PY) || (echo "Creating venv..."; python3 -m venv $(VENV); $(PY) -m pip install -U pip; $(PIP) install -e ".[dev]")

test: ensure-venv
	$(PY) -m pytest -q

fmt: ensure-venv
	$(PY) -m black src tests

lint: ensure-venv
	$(PY) -m ruff check src tests

# Flat (legacy)
PREDICTORS ?= dem_elev
sample-run: ensure-venv
	mkdir -p out
	$(PY) -m biodata.cli enrich \
	  --in data/points_sample.csv \
	  --out out/gold.parquet \
	  --catalog configs/catalog.yml \
	  --predictors $(PREDICTORS) \
	  --window_m 500 --temporal nearest_month
	@echo "Sample flat run completed."

# Groups (recommended)
sample-groups: ensure-venv
	mkdir -p out
	$(PY) -m biodata.cli enrich \
	  --in data/points_sample.csv \
	  --out out \
	  --catalog configs/catalog.yml \
	  --groups configs/run.yml \
	  --window_m 100 --temporal nearest_month
	@echo "Sample groups run completed."

rerun: ensure-venv
	$(PY) -m biodata.cli rerun --from out/last_run.json

clean:
	rm -rf out
