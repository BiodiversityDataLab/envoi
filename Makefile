SHELL := /bin/bash

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: install ensure-venv test fmt lint clean

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

clean:
	rm -rf out
