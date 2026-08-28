# Contributing to envoi

Thanks for your interest in contributing! This guide covers how to set up a development environment, the conventions used in the project, and how to get your changes reviewed.

---

## Ways to contribute

- **Report bugs** by opening an issue on [GitHub](https://github.com/BiodiversityDataLab/envoi/issues). Include a minimal example, the full traceback, and your envoi/Python versions.
- **Request features or new datasets** through an issue. For new built-in catalog entries, please include the GEE asset ID (or local raster source), a citation, and whether the data is continuous or categorical.
- **Submit a pull request** for bug fixes, documentation improvements, or new features. For larger changes, please open an issue first so we can discuss the approach.

---

## Development setup

```bash
git clone https://github.com/BiodiversityDataLab/envoi.git
cd envoi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install                 # one-time, sets up git hooks
```

This installs envoi in editable mode along with the development dependencies (`pytest`, `ruff`, `black`, `build`, `twine`). The `pre-commit install` step wires up the formatting and lint hooks defined in `.pre-commit-config.yaml` so they run automatically on each commit. To run them manually on specific files:

```bash
pre-commit run --files <path1> <path2>
pre-commit run --all-files          # run on the whole repo
```

### Earth Engine credentials

Tests marked `gee` need a live Earth Engine service account. Drop the JSON key at `credentials/ee_credentials.json` or set `ENVOI_EE_CREDENTIALS` to its path. See [README.md](README.md#earth-engine-setup) for the full setup. Without credentials you can still run the non-GEE tests.

---

## Running tests

```bash
pytest                       # all tests
pytest -m "not gee"          # skip live Earth Engine tests
pytest -m gee                # run only the Earth Engine tests
pytest tests/test_extract.py # a single file
```

Live GEE tests are marked with `@pytest.mark.gee` and need network access plus credentials. Please add new GEE-dependent tests behind this marker so the default suite stays runnable offline.

---

## Code style

- **Formatting** — `black` with a 100-character line length. Run `black .` before committing.
- **Linting** — `ruff` with the project config. Run `ruff check .` and fix or justify any new warnings.
- **Comments** — write inline comments liberally. Explain *what* non-trivial blocks do, not only *why* — many users and contributors are not professional programmers, so err on the side of more comments rather than fewer.
- **Variable names** — prefer full, descriptive names (`run_config`, `output_dir`, `coverage_values`) over short abbreviations. `df` for a pandas DataFrame is fine; `cfg`, `cov`, `col` are not.

---

## Repository map

- `src/envoi/` — package source (the orchestrator, adapters, catalog, reducers, QC, output assembly, metadata).
- `src/envoi/configs/` — bundled catalog (`ee_catalog.yml`) and project defaults (`defaults.yml`).
- `src/envoi/adapters/` — adapter registry, `BaseAdapter`, `LocalRasterAdapter`, and the `earth_engine/` subpackage.
- `tests/` — pytest suite, including the `gee`-marked live Earth Engine tests and shared fixtures in `conftest.py`.
- `examples/` — minimal example `run.yml` and `catalog.yml` showing the config schema.
- `demo/` — `getting_started.ipynb`, an interactive walkthrough of the main features.
- `docs/` — design notes (`architecture.md`) and extended usage (`advanced_usage.md`).
- `.github/workflows/` — CI (`ci.yml`) and PyPI release (`release.yml`) pipelines.

---

## Architecture overview

```
extract(df, config)              ← orchestrator (src/envoi/extract.py)
    ↓
_input_validation.py             ← required columns, date parsing, CRS reprojection
_config_parsing.py               ← normalize dict / list / YAML → list of RunSettings
catalog.py                       ← load + merge built-in + user catalogs
    ↓
adapters/__init__.py             ← adapter registry (data_source → adapter class)
    ├── adapters/earth_engine/   ← GeeRasterAdapter + _image / _reducers / _tiles helpers
    └── adapters/local_adapter   ← LocalRasterAdapter (rasterio + geo.py for UTM)
    ↓
reducers.py                      ← python-side reducer registry (local adapter)
qc.py + _output_assembly.py      ← QC flags, column naming, CSV/Parquet write
metadata.py                      ← sidecar JSON (run / config / datasets / warnings)
```

See [docs/architecture.md](docs/architecture.md) for the full module map, data flow, and adapter interface contract.

---

## Adding a new built-in dataset

Built-in Earth Engine datasets live in [src/envoi/configs/ee_catalog.yml](src/envoi/configs/ee_catalog.yml). To add one:

1. Pick a stable, descriptive ID (e.g. `ndvi_landsat_annual`, `lulc_worldcover_2021`). The convention is `<theme>_<source>_<additonal_information>`.
2. Add an entry with at least `data_source: earth_engine` and `path: <GEE asset ID>`. Most other fields are auto-detected; only override them when the default is wrong (see the commented reference block at the top of the catalog file).
3. Include a short `description`, a `citation`, and the `data_type` (`continuous` or `categorical`).
4. Add a smoke test in `tests/test_gee_features.py` marked `@pytest.mark.gee`.

---

## Submitting a pull request

1. Fork the repository and create a feature branch (`git checkout -b feature/my-change`).
2. Make your changes with appropriate tests.
3. Run `black .`, `ruff check .`, and `pytest -m "not gee"` locally.
4. Push your branch and open a pull request against `main`. Describe the change, link any related issues, and note whether the change requires Earth Engine credentials to test.
5. A maintainer will review. Small, focused PRs are easier to review and merge than large multi-purpose ones.

---

## Continuous integration

Two GitHub Actions workflows run automatically:

- **`ci.yml`** runs on every push and pull request. It installs envoi with the `dev` extras across Python 3.10–3.13, runs `ruff check src tests` and `black --check src tests`, then `pytest -q`. The live `gee`-marked tests are skipped in CI (no service account is provisioned), so they should pass deterministically based on the non-GEE suite.

If CI fails on your PR, the formatter/lint output is the first thing to check — running `pre-commit run --all-files` locally reproduces those steps.

The `package` job additionally validates `CITATION.cff` and checks that its `version` field matches `__version__` in `src/envoi/_version.py`. If that step fails, one of the two was bumped without the other — see the release checklist below.

---

## Releasing

Releases publish to PyPI **and** are archived on Zenodo, which mints a DOI for each one. The order of the last two steps matters: `release.yml` fires on a *tag push*, but Zenodo only reacts to a *published GitHub Release*. Pushing a tag alone gets the version onto PyPI with no DOI.

1. **Bump the version in three places, in a single PR:**
   - `src/envoi/_version.py` — `__version__`
   - `CITATION.cff` — both `version` and `date-released` (CI fails if the version doesn't match `_version.py`)
   - `CHANGELOG.md` — a new section for the release
2. **Merge to `main`** once CI is green.
3. **Push the tag:** `git tag vX.Y.Z && git push origin vX.Y.Z`. This triggers `release.yml`, which builds the distribution and publishes it to PyPI via Trusted Publishing.
4. **Publish a GitHub Release** for that tag (Releases → Draft a new release → pick the existing tag → Publish). Zenodo archives the repository at that tag and mints a version DOI for it, plus updating the permanent concept DOI that always points at the latest version.

The release title should match the tag — it becomes the title of the Zenodo archive record.

---

## Questions

Open an issue or start a [discussion](https://github.com/BiodiversityDataLab/envoi/discussions).