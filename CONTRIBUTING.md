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
source .venv/bin/activate          # Windows: .venv\Scripts\activate (PowerShell: .venv\Scripts\Activate.ps1)
pip install -e ".[dev]"
pip install pre-commit
pre-commit install                 # one-time, sets up git hooks
```

To work on the web app, install its extra with `pip install -e ".[dev,webapp]"` and start it with `envoi-webapp`.

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

See [docs/coding_guidelines.md](docs/coding_guidelines.md) for the full code, documentation, and test conventions. The main points:

- **Formatting** — `black` with a 100-character line length. Run `black .` before committing.
- **Linting** — `ruff` with the project config. Run `ruff check .` and fix or justify any new warnings.
- **Comments** — write inline comments liberally. Explain *what* non-trivial blocks do, not only *why* — many users and contributors are not professional programmers, so err on the side of more comments rather than fewer.
- **Variable names** — prefer full, descriptive names (`run_config`, `output_dir`, `coverage_values`) over short abbreviations. `df` for a pandas DataFrame is fine; `cfg`, `cov`, `col` are not.

---

## AI coding agents

The repository includes shared instructions and configuration for AI coding agents (Claude Code and Codex). You don't need them to contribute, but if you use an agent:

- **Instructions** — [AGENTS.md](AGENTS.md) holds the rules every agent follows. Claude Code reads it through [CLAUDE.md](CLAUDE.md). Larger tasks follow [.agents/WORKFLOW.md](.agents/WORKFLOW.md).
- **Specialist roles** — the role instructions live in `.codex/agents/*.toml`. Codex uses them directly. Claude Code uses the short wrappers in `.claude/agents/`, which read the same files. Change a role's instructions only in its `.toml` file.
- **Skills** — shared skills live in `.agents/skills/`. Claude Code finds them through the link in `.claude/skills/`. On Windows without symlink support, that link is checked out as a plain file, and Claude Code then finds the `asd-ste100` skill only through its path in `AGENTS.md`.
- **Models** — the Codex role files pin model names, and the Claude Code wrappers use model aliases (`opus`, `sonnet`). Update both by hand when a model is replaced.
- **Permissions** — [.claude/settings.json](.claude/settings.json) lets Claude Code run the routine checks without asking, and blocks it from reading `credentials/`. The block works only when you start Claude Code in the repository root. Put personal permissions in `.claude/settings.local.json`, which Git ignores.
- **Before you start an agent**, activate the development environment (see [Development setup](#development-setup)), so the agent can run `pytest` and `ruff` directly.
- **Codex** loads `.codex/` only for trusted projects. A role file cannot make a role read-only, so start read-only reviews with `codex --sandbox read-only`.
- **Local files** — `.agents/work/` (task plans and notes) and `.claude/worktrees/` (temporary agent worktrees) are ignored by Git.

---

## Repository map

- `src/envoi/` — package source (the orchestrator, adapters, catalog, reducers, QC, output assembly, metadata).
- `src/envoi/configs/` — bundled catalog (`ee_catalog.yml`) and project defaults (`defaults.yml`).
- `src/envoi/adapters/` — adapter registry, `BaseAdapter`, `LocalRasterAdapter`, and the `earth_engine/` subpackage.
- `tests/` — pytest suite, including the `gee`-marked live Earth Engine tests and shared fixtures in `conftest.py`.
- `examples/` — minimal example `run.yml` and `catalog.yml` showing the config schema.
- `examples/walkthrough.ipynb` — an interactive walkthrough of the main features.
- `src/envoi_webapp/` — the optional Streamlit web app, and its job runner (`jobs.py`, `worker.py`) that runs each extraction in a separate worker process.
- `deploy/serve/` — the container image of the hosted web app for SciLifeLab Serve (see [Web-app container image](#web-app-container-image)).
- `docs/` — design notes (`architecture.md`), extended usage (`advanced_usage.md`), coding conventions (`coding_guidelines.md`), and the generated dataset reference (`datasets.md`).
- `scripts/` — repository tooling, currently `generate_dataset_docs.py` (regenerates `docs/datasets.md` from the catalog).
- `.github/workflows/` — CI (`ci.yml`), PyPI release (`release.yml`), and web-app image (`webapp-image.yml`) pipelines.

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
4. Set `display_name` to the dataset's title as it appears in the Earth Engine catalog (e.g. `"Copernicus DEM GLO-30"`), shortening it if the official title is a full sentence. This is the label shown in the documentation and in the web app's dataset menu, so it must be unique across the catalog.
5. Set `category` to the theme the dataset belongs to (`Terrain`, `Climate`, `Land cover / land use`, `Satellite imagery`, `Vegetation & productivity`, `Human impact`, `Other`). This is the heading the dataset is filed under in the generated documentation.
6. Regenerate the dataset reference: `python scripts/generate_dataset_docs.py`, and commit the updated `docs/datasets.md`. A test fails if the two drift apart; check with `pytest tests/test_catalog_docs.py`.
7. Add a smoke test in `tests/test_gee_features.py` marked `@pytest.mark.gee`.

---

## Submitting a pull request

1. Fork the repository and create a feature branch (`git checkout -b feature/my-change`).
2. Make your changes with appropriate tests.
   For user-visible changes, add an entry to `CHANGELOG.md` under `## [Unreleased]`, in `Added`, `Changed`, or `Fixed`.
   Create the `## [Unreleased]` heading at the top if it does not exist.
   Write the entry for users: state what changed and what they must do differently.
3. Run `black .`, `ruff check .`, and `pytest -m "not gee"` locally.
4. Push your branch and open a pull request against `main`. Describe the change, link any related issues, and note whether the change requires Earth Engine credentials to test.
5. A maintainer will review. Small, focused PRs are easier to review and merge than large multi-purpose ones.

---

## Continuous integration

These GitHub Actions workflows check your changes automatically:

- **`ci.yml`** runs on every push and pull request. It installs envoi with the `dev` and `webapp` extras across Python 3.10–3.13, runs `ruff check src tests deploy` and `black --check src tests deploy`, then `pytest -q`. The live `gee`-marked tests are skipped in CI (no service account is provisioned), so they should pass deterministically based on the non-GEE suite.
  Its `webapp-portability` job runs the web-app tests (`tests/test_webapp_job_protocol.py`, `test_webapp_worker.py`, `test_webapp_jobs.py`, `test_webapp_settings.py`, and `test_webapp_apptest.py`) on Windows and macOS with Python 3.12. The web app starts a worker process for each extraction, and process start, pipes, and process stop differ between operating systems.
- **`webapp-image.yml`** builds and checks the container image of the hosted web app on pull requests that change it. See [Web-app container image](#web-app-container-image).

If CI fails on your PR, the formatter/lint output is the first thing to check — running `pre-commit run --all-files` locally reproduces those steps.

The `package` job additionally validates `CITATION.cff` and checks that its `version` field matches `__version__` in `src/envoi/_version.py`. If that step fails, one of the two was bumped without the other — see the release checklist below.

---

## Web-app container image

The hosted web app runs from a container image on [SciLifeLab Serve](https://serve.scilifelab.se/). The image files are in `deploy/serve/`:

- `Dockerfile` — the image: the Python 3.12 slim base image (pinned by digest), the pinned dependencies, envoi from the source, a user with UID 1000, hosted mode (`ENVOI_WEBAPP_MODE=hosted`), and Streamlit on port 8501.
- `app.py` — the page script that the container runs. It disables core files, then starts the web app.
- `.streamlit/config.toml` — the Streamlit settings for hosted mode, for example the 50 MB upload limit. Keep the upload limit equal to `HostedLimits.max_upload_bytes` in `src/envoi_webapp/settings.py`.
- `requirements.txt` — the pinned versions of all runtime dependencies. `freeze_requirements.py` generates it.
- `smoke_check.py` — a check that needs no network: the page script runs in hosted mode, and a real worker process rejects a generated, invalid key with an error that contains no key material.

The image copies only `pyproject.toml`, `README.md`, `LICENSE`, `src/`, and `requirements.txt`, `app.py`, `smoke_check.py`, and `.streamlit/config.toml` from `deploy/serve/`. The `.dockerignore` file in the repository root keeps credentials, tests, data, and outputs out of the build context as a second barrier.

### Build and check the image locally

You need Docker. Run these commands from the repository root:

```bash
docker build --file deploy/serve/Dockerfile --tag envoi-webapp .
docker run --rm --detach --name envoi-webapp --publish 8501:8501 envoi-webapp
curl --fail http://localhost:8501/_stcore/health   # HTTP 200 when the app is ready (allow up to a minute)
docker exec envoi-webapp id -u                      # must print 1000
docker stop envoi-webapp
docker run --rm --network none --entrypoint python envoi-webapp smoke_check.py
```

While the container runs, the app is at `http://localhost:8501`. To try hosted mode without Docker, on Linux or macOS, start the web app with `ENVOI_WEBAPP_MODE=hosted envoi-webapp`, or run `ENVOI_WEBAPP_MODE=hosted python smoke_check.py` in `deploy/serve/`.

### Update the pinned dependencies

The image installs exactly the versions in `deploy/serve/requirements.txt` (`pip install --no-deps`), and the build fails when `pip check` finds a missing dependency. Regenerate the file when you change the dependencies in `pyproject.toml`, when you change the base image digest in the `Dockerfile`, or to update the versions:

1. On Linux x86_64 with Python 3.12 (the platform of the image), create a new virtual environment and install envoi in it with `pip install -e ".[dev,webapp]"`.
2. Run `pytest -m "not gee"`. Continue only when it passes, so that the image gets the versions that the tests used.
3. From the repository root, run the command in the header of `requirements.txt`, with the Python of that environment:

   ```bash
   .venv/bin/python deploy/serve/freeze_requirements.py --output deploy/serve/requirements.txt
   ```

   The script stops with an error on another platform or Python version. It replaces the file only when all its checks pass, so a failed run leaves the file unchanged.
4. Build and check the image (see above).

### The image workflow

`.github/workflows/webapp-image.yml` builds, checks, and publishes the image:

- **When it runs:** on pull requests that change `deploy/`, `src/envoi_webapp/`, `src/envoi/`, `pyproject.toml`, `.dockerignore`, or the workflow file, on pushed `v*` tags, and on a manual run (Actions → "webapp image" → "Run workflow").
- **Checks** (job `build-and-check`): it builds the image, starts it, and waits up to 60 seconds for HTTP 200 from `/_stcore/health`. Then it checks that the app runs as UID 1000, runs `smoke_check.py` without network, and searches the whole image for a folder named `credentials` and for a JSON file with a service-account key.
- **Publication** (job `publish`): only for a pushed `v*` tag, or for a manual run with `publish` set to true. Pull requests never publish. The job pushes the checked image, not a new build, to `ghcr.io/biodiversitydatalab/envoi-webapp:<version>-<short commit SHA>`. `<version>` comes from `src/envoi/_version.py`, and the short commit SHA has 7 characters.
- **Tags never change.** The job fails when the tag exists already. Serve needs a new tag for each version, and a rollback on Serve selects an earlier tag.

---

## Releasing

Releases publish to PyPI **and** are archived on Zenodo, which mints a DOI for each one. A release also publishes the container image of the hosted web app (step 5). The order of steps 3 and 4 matters: `release.yml` fires on a *tag push*, but Zenodo only reacts to a *published GitHub Release*. Pushing a tag alone gets the version onto PyPI with no DOI.

1. **Bump the version in three places, in a single PR:**
   - `src/envoi/_version.py` — `__version__`
   - `CITATION.cff` — both `version` and `date-released` (CI fails if the version doesn't match `_version.py`)
   - `CHANGELOG.md` — a new section for the release
2. **Merge to `main`** once CI is green.
3. **Push the tag:** `git tag vX.Y.Z && git push origin vX.Y.Z`. This triggers `release.yml`, which builds the distribution and publishes it to PyPI via Trusted Publishing.
4. **Publish a GitHub Release** for that tag (Releases → Draft a new release → pick the existing tag → Publish). Zenodo archives the repository at that tag and mints a version DOI for it, plus updating the permanent concept DOI that always points at the latest version.
5. **Publish and deploy the web-app image.** The tag push in step 3 also starts `webapp-image.yml`. When its checks pass, it publishes `ghcr.io/biodiversitydatalab/envoi-webapp:<version>-<short commit SHA>` (see [The image workflow](#the-image-workflow)). Check that the workflow passed. Then a maintainer with access to the envoi project on SciLifeLab Serve changes the app to the new image tag. To roll back, select an earlier tag on Serve.

   No image is published and no app is deployed on Serve yet. These first steps need a maintainer:
   - **First publication.** A maintainer approves the first publication: a `v*` tag, or a manual run of `webapp-image.yml` with `publish` set to true. After it, the maintainer makes sure that the `envoi-webapp` package on GitHub is public, because Serve needs an image in a public registry.
   - **First deployment on Serve.** Before it, the Serve team must confirm the open hosting questions, for example outbound access to `*.googleapis.com` and the size of the container disk. Then a maintainer creates the app on Serve with the image tag, port 8501, and the permission "Link", so that only people with the link can open it. The maintainer tests the app through the link, and then changes the permission to "Public".

The release title should match the tag — it becomes the title of the Zenodo archive record.

---

## Questions

Open an issue or start a [discussion](https://github.com/BiodiversityDataLab/envoi/discussions).