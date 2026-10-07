# AGENTS.md — envoi

Shared repository instructions for Claude Code, Codex, and other coding agents.

## Scope and repository guidance

- These instructions apply to the entire repository.
- Read applicable subdirectory-specific `AGENTS.md` files before changing files in their scope.
- More specific repository instructions take precedence within their scope.
- Prefer repository conventions over generic agent preferences.
- `CONTRIBUTING.md` is the full contributor guide. The setup and commands below are the subset agents need.
- Follow `pyproject.toml` and `.pre-commit-config.yaml` for formatting and linting settings.
- Before you write or review code in `src/` or `tests/`, read the checklist in `docs/coding_guidelines.md`.
  Then read the sections of that guide for the code you touch.
- Read `docs/architecture.md` when changing architecture or adapter behavior.
- Read `docs/advanced_usage.md` when changing extraction configuration or public behavior.
- Keep these instructions accurate as the project evolves.

## Project purpose and priorities

envoi is a Python package that enriches geographic sample points with environmental data.
It supports Google Earth Engine (GEE) and local rasters through a unified interface.
Inputs are tables of sample points. Default columns are `occurrenceID`, `decimalLatitude`, `decimalLongitude`, and optional `eventDate`.
Callers can override these names through the public `*_column` parameters on `extract()`.
Coordinates are WGS84 (EPSG:4326) by default. `input_crs` declares another CRS, and envoi reprojects to WGS84 internally.
Outputs include tables with environmental columns and raster tiles for ecological research and spatial modeling.

Priorities:
- Accuracy: validate inputs, calculations, and outputs so researchers can trust the results.
- Flexibility: extend the adapter architecture without duplicating shared behavior.
- Reproducibility: preserve run settings, dataset information, and relevant decisions in output metadata.
- GEE access: support efficient server-side queries without requiring data downloads in advance.
- Local raster parity: keep equivalent processing and output contracts across data sources.
- Usability: provide clear configuration, errors, and documentation for users with limited programming experience.
- Global coverage: handle geographic locations, projections, and raster extraction correctly across the globe.
- Portability: support Windows, macOS, and Linux.

## Environment and commands

Use an existing suitable environment when available.
Otherwise, follow the development setup in `CONTRIBUTING.md`, which includes the web app extra.

Common checks:

```bash
pytest tests/test_extract.py
pytest -m "not gee"
pytest -m gee                  # full live suite: only when the user asks
ruff check src tests
black --check src tests
pre-commit run --files <changed paths>
```

Use these exact command forms. Other forms, such as `python3 -m pytest` or `PYTHONPATH=src pytest`, can cause extra permission prompts.

Live GEE tests require network access and service-account credentials.
Use `credentials/ee_credentials.json` or the `ENVOI_EE_CREDENTIALS` environment variable as documented in `README.md`.
Live GEE tests and extractions are slow and consume Earth Engine quota.
Run only the targeted `gee` tests relevant to the change, e.g. `pytest tests/test_gee_features.py -k <name>`.
Do not run large extractions unless the user asks.

## Repository map

- `src/envoi/extract.py`: public extraction entrypoint and orchestration.
- `src/envoi/catalog.py`: dataset registration, inspection, and catalog loading.
- `src/envoi/_config_parsing.py`: extraction configuration normalization and validation.
- `src/envoi/_input_validation.py`: input columns, sample IDs, coordinates, and dates.
- `src/envoi/adapters/base.py`: shared adapter contract.
- `src/envoi/adapters/earth_engine/`: GEE adapter, image selection, reducers, and tile export.
- `src/envoi/adapters/local_adapter.py`: local raster extraction.
- `src/envoi/geo.py`, `reducers.py`, and `qc.py`: geographic helpers, local reducers, and quality checks.
- `src/envoi/_filenames.py`: portable output filename construction.
- `src/envoi/_output_assembly.py` and `metadata.py`: tabular outputs and metadata.
- `src/envoi/configs/`: bundled dataset catalog and default settings.
- `src/envoi_webapp/`: Streamlit interface.
- `tests/`: automated tests and shared fixtures.
- `examples/`: example configurations and notebook walkthrough.
- `docs/`: architecture, advanced usage, coding guidelines, and generated dataset reference.
- `scripts/`: repository tooling, including dataset documentation generation.
- `.agents/`: shared engineering workflow, plan requirements, artifact templates, and skills (`.agents/skills/`).
- `.codex/agents/`: specialist role instructions, used by Codex directly and by Claude Code through `.claude/agents/`.
- `.claude/`: Claude Code role wrappers (`agents/`), the skill link (`skills/`), and shared settings.

## Core working principles

Adapted from
[andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills/tree/main).

### Think before coding

- Inspect relevant entrypoints, configuration, tests, and input/output contracts before editing.
- State material assumptions and tradeoffs explicitly.
- Ask the user about unresolved consequential requirements, scope, compatibility, or scientific decisions.
- Resolve routine implementation questions from repository evidence and established conventions.

### Simplicity first

- Implement only the requested behavior.
- Choose the simplest approach that satisfies the requirements.
- Reuse existing components before introducing new ones.
- Avoid speculative configuration, unnecessary abstractions, and helpers that add only indirection.

### Surgical changes

- Keep each changed line relevant to the requested work.
- Match existing repository conventions.
- Avoid unrelated refactoring, formatting, and cleanup.
- Report unrelated problems separately.
- Remove code made obsolete by your changes.
- Do not revert or overwrite uncommitted changes you did not make.
- If a file changed since you read it, re-read it before editing.

### Goal-driven execution

- Define observable acceptance criteria before substantial implementation.
- Check changed behavior with relevant tests, commands, or generated outputs.
- Continue until the requested outcome is complete or a concrete blocker prevents progress.
- Report incomplete checks and unresolved risks explicitly.

## Code style and documentation

- Follow `docs/coding_guidelines.md` for naming, comments, type hints, docstrings, and code structure.
- Update related documentation, examples, and configuration when their behavior changes.
- Treat `docs/datasets.md` as generated content. Regenerate it instead of editing it manually.
- For user-visible changes, add a `CHANGELOG.md` entry as "Submitting a pull request" in `CONTRIBUTING.md` describes.

## Extraction, catalog, and output contracts

- Preserve the unified `extract(df, config)` interface across GEE and local raster adapters.
- Treat output column names, output folder and file layout, and the metadata sidecar schema as public contracts.
  Researchers' downstream scripts depend on them. Change them only with user approval and a `CHANGELOG.md` entry.
- When you change user-facing behavior in one adapter, make the same change in the other, or document the difference.
- The web app in `src/envoi_webapp/` repeats some package validation and options.
  When changing input validation or user-facing settings, check whether the web app needs the same change.
- Support configurable input column names through the public `*_column` parameters.
- Register custom datasets through `update_catalog()` and the existing catalog machinery.
- When adding built-in datasets, follow the [dataset checklist in CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-built-in-dataset).
- Keep shared behavior outside source-specific adapters where practical.
- Preserve automatic GEE asset inspection and local raster metadata inspection.
- Keep presentation fields such as `display_name` and `category` separate from extraction behavior.
- Inspect dataset metadata before introducing `dataset_spec` overrides. See "GEE dataset pitfalls" below.
- Preserve `collection_date_policy` semantics: `nearest`, `contains`, and spatial `mosaic`.
- Preserve date-selection metadata and documented handling of missing or out-of-range sample dates.
- Use `_filenames.py` for output path components across both adapters.
- Output file and folder names may contain only `[A-Za-z0-9._-]`.
- Sanitize sample IDs to that character set. Append a stable hash suffix when an ID was rewritten.
- Preserve filenames for IDs that already satisfy portability and length limits.
- Reject unsafe configuration-supplied `batch_id` values and dataset keys instead of silently rewriting them.
- Preserve raster-mode validation that requires present, nonblank, unique sample IDs before adapter construction.
- Preserve tile manifest CSVs that map input rows to exported files. Multi-window runs use suffixed manifest filenames.
- Do not reconstruct filenames independently.

### GEE dataset pitfalls

These mistakes do not raise errors. They silently produce null or wrong statistics.
Check for them when adding or changing a GEE catalog entry or `dataset_spec`.

- **Space-tiled "static" collections need `collection_date_policy: mosaic`.**
  Some static products are cut into geographic tiles, and each tile's `system:time_start` records its acquisition date.
  Confirm whether timestamps represent acquisition dates of spatial tiles rather than successive observations.
  For one static product split across tiles, use `collection_date_policy: mosaic`.
  Treating tile acquisition dates as a time series can exclude tiles covering the sample point and produce empty results.
  Check dataset documentation first. Inspect timestamps and properties on a small spatially filtered subset when needed.
  Avoid aggregating timestamps across an entire large collection. Timestamp variation alone does not establish that `mosaic` is appropriate.
  `dem_copernicus_glo30` is the built-in example.
- **Mosaics lose their projection.** `mosaic()` drops projection metadata.
  Without it, `ee.Terrain.slope` and `ee.Terrain.aspect` silently return null.
  The adapter restores the collection's native projection with `setDefaultProjection`. Keep that step.
- **`use_utm_zone` empties collections without a `UTM_ZONE` property.**
  Set it only after checking `ee.Image(collection.first()).getInfo()["properties"]`.
  `aef_satellite_embeddings` is a built-in example.
  Collections that `filterBounds` already disambiguates (e.g. the GLO-30 DEM) do not need it.
- **`nominalScale()` can return about 111 km.** This is the EPSG:4326 default for assets with missing native projection metadata.
  Statistics are then computed at the wrong resolution. Set `dataset_spec.native_scale_m` to the true resolution.

## Communication and writing

- Use plain, direct language and consistent terms.
- Define unfamiliar project-specific terms before using them.
- State conclusions in your own words after assessing subagent findings.
- Summarize behavior changed, checks performed, and remaining limitations.
- For durable specifications, plans, handoffs, and review findings, read `.agents/skills/asd-ste100/SKILL.md`.
- Apply that skill's clarity rules without removing technical precision or scope qualifiers.

## Engineering workflow and agent coordination

- Handle small, well-defined changes directly.
- For non-trivial changes, state the intended approach and material assumptions before editing.
- For substantial work, read and follow `.agents/WORKFLOW.md`.
  It covers specialist roles, delegation, plans, task artifacts, code review, and the decision log.
- Record lasting architectural, workflow, and scientific decisions in `docs/decision_log.md`.

## Validation and code review

- Start with the smallest check that directly exercises the changed behavior.
- Add or update tests for changes to core behavior where appropriate.
- Write tests as the "Tests" section of `docs/coding_guidelines.md` describes, including the `gee` marker and credential skip.
- The `gee` marker labels tests. It does not exclude them from default pytest runs.
- Run relevant tests and `pre-commit run --files <changed paths>` on modified files.
- For broad code changes, also run the non-GEE suite and repository lint/format checks.
- Inspect the final diff for accidental changes.
- Do not weaken tests, validation, linting, or safety checks to make a change pass.
- If a required check cannot run, state the reason and the command the user must run.
- When you review code, follow the "Code review" section of `.agents/WORKFLOW.md`.

## Safety and Git conventions

- Keep credentials, secrets, large datasets, and runtime outputs outside version control.
- Do not read, print, or copy the contents of `credentials/` or the file named by `ENVOI_EE_CREDENTIALS`.
  To debug authentication, check only whether the file exists, and call `envoi.auth.init_gee()`.
- Do not change `src/envoi/_version.py`, `CITATION.cff`, or published release sections in `CHANGELOG.md` unless the user asks.
  Turning `[Unreleased]` into a release section is part of the release, not routine work.
  Releases follow the checklist in `CONTRIBUTING.md`.
- Avoid destructive Git commands unless the user explicitly requests them.
- Wait for user approval before creating commits or pull requests unless that authority is explicitly delegated.
- Do not add agent co-authorship trailers or generated-by attribution to commits or pull request descriptions.
