# Coding guidelines

## Purpose

This guide records the coding and documentation patterns of the envoi codebase.
Many envoi users and contributors are researchers, not professional programmers.
Write code that such a reader can follow from top to bottom.

This guide covers only envoi-specific decisions. Otherwise, follow PEP 8 and `pyproject.toml` (`black`, `ruff`, 100-character lines).
`AGENTS.md` holds the working rules: checks to run, credentials, Git, workflow, the output and filename contracts,
and the GEE dataset pitfalls.
`CONTRIBUTING.md` holds setup and release steps. `docs/architecture.md` holds the module map, data flow, and adapter method contract.

When this guide, the documentation, and the code disagree, check the implementation and the tests.
Report a disagreement that affects a public contract before you change anything.

### How to use this guide

Read the checklist below for every change. Then read only the sections for the code you touch.
Where a rule names a function, read that function as the model to follow.

### Rule labels

- **[Contract]**: Behavior that users, downstream scripts, tests, or other modules depend on.
  Change it only with user approval and a `CHANGELOG.md` entry, as `AGENTS.md` describes.
- **[Proposed]**: A new requirement that existing code does not follow everywhere yet.
  The team must agree to it before it becomes binding.

A rule without a label is the preferred style. Do not reformat unrelated code only to apply it.

## Checklist for every change

- Prefer clarity over cleverness or brevity. Make the smallest change that fits the current architecture.
- Never return a silently wrong or silently incomplete result. Raise an error, or record a warning in the metadata sidecar.
- Validate input and config before any adapter is built or any Earth Engine quota is used.
- Raise `ValueError` for invalid user input or config, including wrong types. Write messages a non-programmer can act on.
- Make a user-facing change in both adapters, or reject it in the other one and document the difference.
- Never change the caller's dataframe or a catalog dict in place.
- Check the coordinate order at every call: `geo.py` and `pyproj` take longitude first, adapter methods take latitude first.
- Build output names only through `_filenames.py`.
- Use full, descriptive names (`run_config`, not `cfg`).
- Put a short comment at the start of each processing stage. Explain why when the code depends on a non-obvious fact.
- Add type hints, and a docstring that states the contract.
- Assert values in tests, not only column presence.

## Design patterns

### Orchestration in `extract()`

`extract()` in `src/envoi/extract.py` is the single public entry point. **[Contract]**
It reads as a sequence of stages, each calling a helper that owns the details.
The "Data flow" section of `docs/architecture.md` shows the stages.

- Keep the stage sequence visible in the orchestrator. Do not hide it behind a single `_run_everything()` call.
- Give each output mode its own helper (`_process_dataset_tabular`, `_process_dataset_raster`), and dispatch with a plain `if`/`elif`.
- Each helper owns one unit of work, for example one `(dataset, window_size)` pair. The caller owns the loops and the merging of results.
- Move pure logic into the private modules (`_input_validation.py`, `_config_parsing.py`, `_output_assembly.py`).
  The orchestrator shows the stages of a run, not every branch of every stage.

A reader can then understand a full run by reading one function.

### Explicit conditional control flow

Keep optional behavior visible at the layer that owns the decision.
The helper does the work for the enabled case. It does not hide whether the behavior is active.

```python
# Prefer
if write_metadata:
    write_metadata_sidecar(...)

# Avoid
_maybe_write_metadata(write_metadata, ...)
```

Run options (`output_type`, `write_metadata`) belong in `extract()`. Source-specific choices belong in the adapter.

### Module layout and public API

A leading underscore marks code that is not part of the public API.
The public API is the list in `src/envoi/__init__.py` (`__all__`). **[Contract]**
Add a name there only on purpose, and add a `CHANGELOG.md` entry.

The underscore means "internal to envoi", not "internal to this file".
Package modules, tests, and `src/envoi_webapp/` may import private helpers to reuse a rule instead of copying it.
For example, the web app calls `_validate_sample_ids`.

- Give each module one responsibility, stated in its module docstring.
- Split out a new private module when a group of helpers has a shared purpose and the parent grows too long to read.
  `_output_assembly.py` was split out of `extract.py` for this reason.
- Add a helper when it isolates a meaningful operation, supports reuse, or makes a contract easier to test.
- Do not add helpers only to make functions shorter, one-function modules used in one place,
  or helpers that only call another function or hide one non-reused line.

### Adapters

An adapter reads one kind of data source: `GeeRasterAdapter` (Earth Engine) or `LocalRasterAdapter` (GeoTIFF).
Each subclasses `BaseAdapter`. `docs/architecture.md` lists the method contract.
`LocalRasterAdapter.__post_init__` is the model for setup and cleanup.

Interface and registration:
- Keep method signatures identical across adapters. **[Contract]**
  To add a keyword, add it to `BaseAdapter` and both adapters in the same change. Check every caller first.
- Return one `(stats_dict, meta_dict)` pair per input point, in input order. **[Contract]**
  The GEE adapter finishes requests out of order and writes each result back to its input index.
- Register the adapter at the end of its module with `_register("<data_source>", AdapterClass)`, and import the module in `adapters/__init__.py`.
- If only one source supports a feature, reject it for the other source with a clear `ValueError`.
  Example: `_config_parsing.py` rejects the GEE-only derived bands (`slope`, `aspect`) for local datasets.

Setup, caching, and cleanup:
- Read settings from `spec` (the merged catalog entry) in `__post_init__`, and store derived values as attributes.
- Do cheap setup, and setup every call needs, in `__post_init__`.
  Defer expensive inspection that becomes cheaper with batch information.
  Example: the GEE adapter loads collection timestamps on the first batch, filtered by that batch's area and dates.
  Without the filter, large collections exceed Earth Engine's compute limits.
- Cache values that are expensive to fetch and used across points. A cache lives as long as the adapter instance,
  and `extract()` builds one instance per `(dataset, window_size)` pair.
- If setup opens a resource and then fails, release it before re-raising.
  The orchestrator's `with` block cannot do this, because `__exit__` never runs when the constructor fails.
- If the adapter holds a resource, override `close()`, and make it safe to call more than once.

Failures for single points (one bad point must not cost the whole batch; a programming error must still stop the run):
- Continue the batch when the point is outside the raster, its read fails, or its Earth Engine request fails.
  Return an empty result for that point: null statistics, or `None` instead of a tile path.
  The QC columns, tile manifest, and metadata sidecar then report it as failed. **[Contract]**
- Catch the specific exception types, as `LocalRasterAdapter` does (`ValueError`, `WindowError`, `RasterioIOError`).
  Let programming errors (`TypeError`, `AttributeError`, `KeyError`) propagate.
- The GEE adapter catches `Exception` per worker result, because Earth Engine reports unrelated failures with one exception type.
  It logs each failure with the point index. Do not copy this broad catch where specific types exist.
- Stop the run with a clear error for failures that affect every point,
  such as a missing GEE asset, missing credentials, or a native scale that falls back to about 111 km.

### Shared behavior outside adapters

Put logic that does not depend on the data source in a shared module. Do not copy it into both adapters.

| Concern | Module |
|---|---|
| UTM zone and CRS helpers | `geo.py` |
| Output filenames and path-component checks | `_filenames.py` |
| QC columns and coverage flags | `qc.py` |
| Metadata summaries and the sidecar file | `metadata.py` |
| Python-side reducers | `reducers.py` |
| Progress events | `progress.py` |

Source-specific code and decisions stay inside the adapter.
Examples are Earth Engine reducers, rasterio windows, image selection by date, band reading, and nodata handling.

### Settings, defaults, and the catalog

| Kind of value | Location | Example |
|---|---|---|
| Project-wide default that users can change | `src/envoi/configs/defaults.yml` | `window_size_m`, `max_workers` |
| Built-in dataset definition and dataset-specific behavior | `src/envoi/configs/ee_catalog.yml` | `path`, `bands`, `dataset_spec` |
| Custom dataset definition | A catalog registered through `update_catalog()` | A local GeoTIFF |
| Per-run choice | The run config passed to `extract()` | `batch_id`, `statistics`, `output_type` |
| Per-call option that is not part of a run | A keyword argument of `extract()` | `input_crs`, `id_column`, `quiet` |
| Fixed rule or closed set of values | Module-level constant in code | `_ALL_KNOWN_REDUCERS`, `_ALLOWED_CHARACTERS_PATTERN` |

Do not add a setting for every implementation detail.

- User-registered catalog entries override built-in entries with the same name. **[Contract]**
- GEE asset type detection and local raster metadata detection live in `catalog.py`.
- GEE datasets identify bands by name (`DEM`, `B4`). Local rasters use integer band indices that start at 1.

Parsed settings (model: `RunSettings` in `_config_parsing.py`):
- Parse raw config once, at the boundary, into a typed object. Downstream code reads that object, not the raw dict.
- `_parse_run_config()` raises `ValueError` for every missing or invalid setting. Code that receives a `RunSettings` does not validate it again.
- When the sidecar must record the user's original form of a setting, store it in a separate field (`user_stats`). Do not reconstruct it later.
- Name fields after their config keys, except where the shape changes: `window_size_m` (one value or a list) becomes `window_sizes`.

Module-level constants (model: `_ALL_KNOWN_REDUCERS` in `_config_parsing.py`):
- Use upper case, with a leading underscore when private. Use `frozenset` for closed sets and compiled `re` patterns.
- When the purpose or value is not obvious, put a comment above it that says what it controls and why.
- Avoid module-level mutable state, except documented caches and registries.
  Provide a way to reset that state, as `reset_catalog()` does, so tests stay isolated.

YAML files (`defaults.yml`, `ee_catalog.yml`, `examples/`):
- Use snake case keys, 2-space indentation, and brackets for short lists (`bands: [DEM]`).
- Start each group of entries with a heading comment (`# ── Terrain ───...`).
- Use real catalog dataset names in runnable examples.
- Explain every `dataset_spec` override in a comment that says what goes wrong without it.
  The `dem_copernicus_glo30` catalog entry is the model.

### Dataframe flow

- Pass dataframes into functions and return them. Do not keep them as mutable state on an object.
- Copy the caller's dataframe once at the public entry point. Never change the user's object. **[Contract]**
- Inside the pipeline, use the canonical column names `id`, `lat`, `lon`, and `date`.
  Rename right after validation, and rename back just before output (`_restore_user_column_names`).
- Never change a catalog dict in place. Build a merged copy: `merged_config = {**dataset_config, **(band_overrides or {})}`.
- Keep IDs, coordinates, dates, and per-point results aligned. When a step drops rows, filter every parallel list in the same step.
- Keep the input row order and index through the whole pipeline. `pd.concat` and `.loc` joins depend on them.
- Name boolean masks (`null_date_mask`) instead of writing long inline filter expressions.

Researchers often call `extract()` repeatedly in a notebook. Hidden changes to their dataframe or the shared catalog leak between calls.

### Geographic and statistical correctness

Mistakes here usually do not raise an error. They produce plausible numbers that are wrong.

Coordinates and windows:
- `_validate_and_reproject_crs` reprojects coordinates from `input_crs` to WGS84.
  The output keeps the user's original coordinates and adds `_wgs84` columns. **[Contract]**
- Check the coordinate order at every call. `get_utm_crs(lon, lat)` and `pyproj` transformers with `always_xy=True` take longitude first.
  Adapter methods (`fetch_values`, `fetch_stats_batch`, `export_tiles`) take latitude first.
- `window_size_m` is the side length of a square window in metres, not a radius. **[Contract]**
- Build metric windows in each point's UTM zone through `geo.py`. Do not compute distances in degrees.
- For a geographic change, also check both hemispheres, longitudes near ±180°,
  latitudes near the projection limits (UTM is not defined near the poles), and points at a raster edge.

Reducers and missing data:
- Handle nodata, NaN, and empty windows explicitly. A NaN nodata value needs `np.isnan`, because `nan == nan` is `False`.
- A reducer on an empty window returns NaN, not an error and not zero.
- The `point` reducer samples the exact pixel at the point. Its columns carry no window suffix. **[Contract]**
- `class_count` and `class_fraction` are 0 for an unobserved class, even when the window had no valid pixels.
  The QC columns (`in_extent`, `coverage_pct`) distinguish "absent class" from "no data". Keep both. **[Contract]**
- When you change a reducer, check its definition, empty input, masks, the denominator (`ddof=1` for `std`), and the output type.
- Compare the local and GEE implementations before you claim they give the same result. Document material differences.

Earth Engine requests (in addition to the GEE dataset pitfalls in `AGENTS.md`):
- Keep statistics on the server. Combine reducers so one request returns all statistics for a point (`_build_combined_reducer`).
- Avoid `getInfo()` inside per-point loops unless the result is that point's final value.
- Keep the tile-size check (`_check_tile_size`) and the retry and backoff in `_download_tile_via_url`.

### Validation, errors, and warnings

Validation lives in `_input_validation.py` (input dataframe), `_config_parsing.py` (run config),
`catalog.py` (catalog entries), and `_filenames.py` (path components).
Reuse these helpers. Do not write a second version in an adapter or the web app.

Raster mode's sample-ID check does not apply to tabular mode, where duplicate IDs are harmless.
Do not extend it without an approved behavior change.

Error contract:
- Raise `ValueError` for invalid user input or config, including wrong types. **[Contract]**
  `CatalogError` subclasses `ValueError`. The tests assert this (see `TRY004` in `pyproject.toml`).
- Raise `FileNotFoundError` for a missing local raster, and `RuntimeError` for an environment problem such as missing credentials.
- Chain converted third-party exceptions: `raise ValueError(...) from e`.

An error message must let a user with limited programming experience fix the problem without reading the source.
It says where the problem is (prefix config errors with `Output '{batch_id}': `), what is wrong in the user's own column and setting names,
what envoi found and what it allows, and, when not obvious, what to do.

```python
# Good
raise ValueError(
    f"Output '{batch_id}': unknown reducer(s) {unknown} in '{context}'. "
    f"Valid reducers: {sorted(_ALL_KNOWN_REDUCERS)}."
)

# Bad
raise ValueError("bad reducer")
```

Warnings tell the user that results are valid but affected by something, such as dropped rows or a reprojection.
Send each warning through two channels, as `_parse_and_validate_dates` does:
`warnings.warn(message, stacklevel=2)` for the user now, and a returned list that `extract()` records in the sidecar's `warnings`.
Emit one summary warning per kind of problem, not one per row.

Use the module logger for diagnostics that do not change results (`logger.debug`).
Avoid `print()` in package code. **[Proposed]** (One `print()` remains in `_parse_and_validate_dates`.)

Broad exceptions: `except Exception` is allowed only when the handler re-raises a clear `ValueError`,
downgrades a harmless best-effort failure to a logged warning (for example raster inspection in `catalog.py`),
or replaces one point's failed result with an empty result (see "Adapters").
Put a comment above the `except` that says which case applies.

Runtime code in `src/envoi/` never imports from `tests/`.

### Reproducibility and metadata

`AGENTS.md` lists the output contracts.

- Record what a run actually used: requested settings, resolved datasets and bands, source information, date decisions, coverage, and export results.
- Write metadata only through `metadata.py` and the adapters' `build_dataset_meta()`. Do not create a second sidecar format.
- Keep structured progress events (`progress_callback`) independent of the terminal progress bars (`quiet`). **[Contract]**

### Paths and output filenames

- Use `pathlib.Path`. Accept `str | Path` at public boundaries and convert once.
- Do not assume the user runs Python from the repository root. Use the caller's output path.
  Read the bundled YAML only through `load_defaults()` and `load_catalogs()`, which use `importlib.resources`.
- Build every output name component through `_filenames.py` (rules in `AGENTS.md`). Never use an f-string in an adapter or the web app.
  Sanitize data-supplied values (sample IDs) with `sanitize_filename_component` or `build_tile_filenames`.
  Reject unsafe config-supplied values (`batch_id`, dataset keys) with `is_safe_path_component`.
- Create output folders with `mkdir(parents=True, exist_ok=True)` just before the write.
- Write text files with `encoding="utf-8"`, so output is the same on every platform. **[Proposed]**

### Tool preferences

- Use pandas, numpy, rasterio, pyproj, shapely, geopandas, and `earthengine-api`.
  Do not introduce polars, or conversions between dataframe libraries, without a design decision.
- Keep Streamlit code in the optional `src/envoi_webapp/` package.
- Keep code compatible with Python 3.10 and the dependency versions in `pyproject.toml`.
- Discuss a new runtime dependency before adding it, because it affects every user's installation.

## Readability

### Naming

- Use full names: `run_config` not `cfg`, `output_dir` not `out_dir`, `coverage_values` not `cov`, `column` not `col`.
  `df`, `lat`, and `lon` are acceptable. Use full names in comprehensions too: `[column for column in df.columns]`.
- Name the specific thing: `null_date_mask` not `mask`, `exported_paths` not `paths`, `dataset_config` not `ds`.
- Put the unit at the end of a name when it could affect a calculation: `window_size_m`, `min_coverage_pct`.
- Name every action a function takes: `_validate_and_reproject_crs`, not `_check_crs`.
- Older code still has short names (`e`, `m`, `idx`). Do not copy them. Rename them only when you change the surrounding lines.

Usual meaning of function-name prefixes (the docstring states the exact contract):

| Prefix | Usual meaning |
|---|---|
| `_validate_` | Check input and raise when it is invalid. Can also return a cleaned dataframe and warnings. |
| `_parse_` | Turn raw user input into a typed value. |
| `_normalize_` | Turn one of several accepted shapes into one shape. |
| `_resolve_` | Choose a value from several sources or rules. |
| `fetch_` | Read data from a source (adapter methods). |
| `build_` | Construct and return a value. Can read from the source, but does not write files. |
| `summarize_` | Reduce per-point results to a metadata summary. |
| `write_` | Write to disk. |

### Layout

- Separate the logical stages of a function (setup, validation, processing, output) with blank lines.
  When a comment introduces the next block, put the blank line before the comment.
- In a long module or class, group related functions under a banner comment (`# ---...` / `# Band metadata helpers` / `# ---...`).
  Inside a long function, mark distinct branches with a short banner (`# ---- shape 1: plain string ----`).
  Do not use banners in short modules.

### Inline comments

envoi uses more comments than many Python projects, because many readers are researchers.
A comment that explains *what* a non-trivial block does is welcome, and so is one that explains *why*. When in doubt, add the comment.

- Put a short comment at the start of each processing stage, so a reader can follow a function by reading only its comments.
- When the code depends on a non-obvious fact, say what goes wrong without it.
  Examples are a GEE behavior, a numerical edge case, a cross-platform limit, or a scientific decision.
  Several GEE pitfalls produce no error, so a comment is the only protection against "simplifying" the code.
- Do not narrate trivial assignments.
- Do not refer to history ("the previous behaviour") unless a compatibility promise depends on it. If one does, name the promise.

```python
# Good: records a reason
# NaN nodata: we have to use isnan because nan == nan is False.
if isinstance(nodata, float) and np.isnan(nodata):
    ...

# Bad: repeats the code
# Set the batch id
batch_id = run_settings.batch_id
```

## Code documentation

### Module docstrings

Start each module with a docstring. **[Proposed]** `qc.py`, `metadata.py`, `catalog.py`, and `reducers.py` do not have one yet.
Add one when you make a substantial change to such a module. The model is `_input_validation.py`.
It states the module's responsibility, its main functions, and its boundaries (for example "no GEE calls").

### Type hints

Envoi-specific choices (otherwise use standard modern hints with `from __future__ import annotations`):
- Use `Any` for raw user config whose shape is not yet validated, and for Earth Engine objects.
- Show `None` explicitly when it is a valid result: `-> dict | None`.
- Return a tuple when a function produces several outputs, such as `-> tuple[pd.DataFrame, list[str]]` for `(df, crs_warnings)`.
  Name the elements in the docstring.

### Docstrings

- Use a one-line docstring only for simple helpers with obvious arguments. Otherwise start with a one-line summary, a blank line, then the detail.
- Explain purpose, contract, inputs, and outputs. Do not narrate every line. Use double backticks for code names.
- Document, when they apply: units and coordinate order; what `None`, NaN, or an empty result means;
  whether the function changes its input, writes files, or makes a network request; which resources it opens and who closes them;
  the exceptions it raises and the failures it recovers from; and scientific assumptions.

Use Google-style `Args`, `Returns`, and `Raises` sections in new and substantially rewritten docstrings. **[Proposed]**
The model is `_validate_sample_ids` in `_input_validation.py`.
Do not convert docstrings your change does not touch.
The public `extract()` docstring uses NumPy style. Keep it until the project chooses one style for public functions.

Class docstrings explain the component's purpose and main assumptions in three to six lines.
For an adapter, also state what it caches (model: `GeeRasterAdapter`).
Document non-obvious dataclass fields with a comment next to the field, as `RunSettings` does.
Use an `Attributes` section only for small result containers, such as `QualityControlBuildResult` in `qc.py`.

When a method depends on important cached attributes from `self`, list them in a `Uses attributes` section after `Args`:

```python
    Uses attributes:
        scale: Catalog override from ``dataset_spec.native_scale_m``, or None.
        _native_proj: Cached native projection, used when no override is set.
```

## Tests

`AGENTS.md` says which tests and checks to run. Write tests like this:

- Put tests in `tests/test_<module_or_feature>.py`, grouped in `Test<Behavior>` classes (`TestErrors`, `TestTileManifest`).
- Give each test a one-line docstring that states the expected behavior.
- Assert observable results, not implementation details. For numerical behavior, compare against an independent calculation.
- Use the synthetic fixtures in `tests/conftest.py` (`sample_df`, `dem_tif`). Do not add binary fixture files.
  Generate data with a seeded random generator.
- Write outputs to pytest's `tmp_path`.
- For errors, assert the type and a stable fragment of the message: `pytest.raises(ValueError, match="missing required column")`.
- Cover the relevant edge cases: nodata and NaN, empty windows, band selection, filename collisions, date boundaries, and multi-window outputs.
- Mark tests that need Earth Engine with `@pytest.mark.gee`.
  Skip them without credentials using `gee_credentials_available` from `tests/_gee_helpers.py`, as `tests/test_gee_features.py` does.
  Tests without the marker must not use the network.
- Rely on the `_reset_user_catalog` fixture for catalog isolation. Do not call `update_catalog()` without it.
