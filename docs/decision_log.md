# Decision log

This log records lasting architectural, workflow, and scientific decisions for envoi, as `AGENTS.md` requires.
Each entry gives the date, the decision, the reason, and the alternatives that were rejected.
Add new entries at the end. Do not edit old entries. When a decision changes, add a new entry that names the old one.

## 2026-10-07 — Agent setup for Codex and Claude Code

These decisions came from the review of the shared agent configuration (`AGENTS.md`, `.agents/`, `.codex/`, `.claude/`).

### One copy of the role instructions

- **Decision:** the `developer_instructions` in `.codex/agents/<role>.toml` are the only copy of each specialist role's instructions.
  The Claude Code wrappers in `.claude/agents/<role>.md` tell the agent to read the matching `.toml` file, and to stop if it cannot read it.
- **Reason:** two copies would drift. Codex gets its instructions injected directly, so its behavior does not change.
- **Rejected:** shared `.agents/roles/<role>.md` files (Codex would then also depend on a file read), and full copies in both tools.

### Five Claude Code roles with fixed tools

- **Decision:** Claude Code has all five roles: `explorer`, `researcher`, `plan_reviewer`, `implementer`, and `code_reviewer`.
  `explorer`, `plan_reviewer`, and `code_reviewer` get only Read, Grep, and Glob. `researcher` also gets WebSearch and WebFetch. `implementer` gets all tools.
- **Reason:** the built-in Claude Code agents do not follow the repository's role rules. The tool lists make the review roles read-only.
- **Consequences:** the read-only roles cannot run commands, so the primary agent gives `code_reviewer` the diff and the check results.
  The Claude Code `researcher` cannot write files, so the primary agent writes research files from its answer.
- **Rejected:** use the built-in `Explore` and `general-purpose` agents for `explorer` and `researcher`. Give `code_reviewer` Bash (it could then change files).

### Models

- **Decision:** the Codex role files keep their pinned model names and reasoning efforts. The Claude Code wrappers use the aliases `opus` (`plan_reviewer`, `code_reviewer`, `implementer`) and `sonnet` (`explorer`, `researcher`), which mirror the Codex tiers.
  The team updates both by hand when a model is replaced.
- **Reason:** the team wants explicit control of the models. Aliases name no version, so they need fewer updates than pinned IDs.
- **Rejected:** `model: inherit` for the Claude Code roles. Review quality would then depend on the model of the session that starts the review.

### Required return headings

- **Decision:** each role's "Return" list starts with "Use these items as section headings, in this order. Write 'none' for an empty item."
- **Reason:** in tests, one Claude Code role and one Codex role each left out part of their output contract. With the line, all roles passed.
- **Rejected:** accept answers with the right content but no headings. One answer had left out content, not only headings.

### Location of the shared skills

- **Decision:** shared skills live in `.agents/skills/`, where Codex finds them. `.claude/skills/<skill>` is a symlink to the same folder, for Claude Code. `AGENTS.md` also gives the skill path.
- **Reason:** each tool discovers skills in a different folder. A symlink keeps one copy.
- **Consequence:** on Windows without symlink support, Claude Code finds the skill only through the path in `AGENTS.md`. This limit was accepted without a Windows test.
- **Rejected:** the real folder in `.claude/skills/`, a second copy of the skill, and no automatic discovery (the earlier location, a top-level .skills folder, which neither tool scans).

### Where implementers work

- **Decision:** one implementer at a time works in the normal checkout, on the current branch.
  For parallel implementers, the primary agent commits first, then creates one worktree per implementer from the current branch with `git worktree add .claude/worktrees/<name> <branch>`.
  Implementers do not use the worktree isolation of the Claude Code Agent tool.
- **Reason:** that isolation starts from `origin/main`, so it does not contain the current branch or uncommitted work.
- **Rejected:** push work to `main` before each implementer, and drop parallel implementation.

### Shared Claude Code permissions

- **Decision:** `.claude/settings.json` allows these exact commands without a prompt: `pytest -m "not gee"`, `ruff check src tests`, `black --check src tests`, `pre-commit run --files ...`, `git status`, `git diff`, `git diff --stat`, `git diff --cached`, and `git log --oneline`.
  It denies reads of `credentials/`, `.env`, `~/.config/envoi/`, and `~/AppData/Roaming/envoi/`.
- **Reason:** fewer permission prompts for routine checks, and a second barrier for the rule "do not read credentials" in `AGENTS.md`.
- **Limits:**
  - The deny rules worked in sessions started in the repository root. In a session started in `src/`, reads of `../credentials/` asked for permission instead of being denied. The cause was not found.
  - The deny rules do not cover a file named by `ENVOI_EE_CREDENTIALS` in another folder, or every possible way to read a file.
  - `pre-commit run --files ... -c <file>` can run any hook of another configuration file. An agent must first write that file.
  - Codex ignores `sandbox_mode` in role files. A Codex role is read-only only when its session starts read-only (`codex --sandbox read-only`).
- **Rejected:** wildcard rules such as `pytest -m "not gee" *`, because a second `-m gee` option selects the live Earth Engine tests.
  A hook that checks `pytest` options, because it adds a script to maintain.

## 2026-10-09 — Pipeline bug fixes

These decisions came from the review of the extraction pipeline (branch `pipeline-bugs`).

### Failed points

- **Decision:** a point whose data request or read raised an exception that the adapter caught (a failed point) does not stop the run.
  - The adapters record the error text (`"<exception type>: <message>"` on one line, at most 500 characters, from `qc.format_point_error`) and the exception in the point's meta dict.
    A failed point has null statistics and adds no statistic column. A point outside the extent is not a failed point.
  - For an Earth Engine collection that selects one image for each point (`nearest` or `contains`), a point where the collection has no image at its location, on any date, gets a fully masked image: it is a point without data and has no error.
    A point whose location has images, but none after the selection filters (the selected dates, or the UTM zone of `use_utm_zone`), stays a failed point, so problems of the image selection stay visible.
    The check counts the images after `filterBounds` only, and stops at the first image (`limit(1)`).
  - `extract()` reports failed points in the QC column `{dataset}_error_{N}m` (or `{dataset}_error_point`), in an `error` column of the tile manifest, and in the sidecar fields `n_errors` and `ids_errors` of each quality summary.
    Each column exists only when a point of that batch failed. The two sidecar fields are always present.
    `quality.tiles.n_failed` keeps its meaning: the number of tiles that were not exported for any reason.
  - `extract()` gives one warning for each dataset and window with failed points, and records it in the sidecar `warnings`.
  - When every point of one dataset and window fails, `extract()` raises `RuntimeError` with the error of the first failed point (the lowest input index), chained to its exception.
    `extract.py` applies this rule and the warning after the adapter call. The adapters do not.
- **Reason:** before, a failed Earth Engine point looked like a point outside the extent, a local read error stopped the whole run,
  and a batch where every point failed finished with empty values and no error. Users could not tell missing data from a failed request.
  One bad point must not cost the whole batch, but a cause that affects every point must stop the run with the real error.
- **Rejected:**
  - Stop the run on any failed point: one temporary Earth Engine error would cost a long batch.
  - Report failed points only in a warning and the sidecar: users could not find the failed rows in the output tables.
  - The "every point failed" rule in each adapter: one rule would exist in two copies.
  - The full traceback as the error text: it is long and shows internal paths.
  - Count failed tiles in the existing `quality.tiles.n_failed`: that field already counts tiles that were not exported for any reason.
  - A fully masked image for every empty date selection of a collection: problems of the date selection would look like missing data.

### Population statistics

- **Decision:** `std` and `var` are population statistics (divide by the number of pixels n, `ddof=0`) in both adapters.
  The Earth Engine values do not change. The local values become smaller by the factor sqrt((n − 1)/n) for `std` and (n − 1)/n for `var`. A window with one valid pixel gives `std = 0` and `var = 0`.
- **Reason:** the adapters gave different values for the same request. The Earth Engine adapter uses `ee.Reducer.stdDev` and `ee.Reducer.variance`.
  `earthengine-api` also has `ee.Reducer.sampleStdDev` and `ee.Reducer.sampleVariance` ("the sample standard deviation", "the sample variance"), so `stdDev` and `variance` are the population statistics.
  With population statistics, only the local adapter changes, and Earth Engine results that users already have stay valid.
- **Rejected:**
  - Sample statistics in both adapters (`sampleStdDev` and `sampleVariance` on Earth Engine): every existing Earth Engine result would change.
  - Keep the difference and document it only: results from the two data sources would not be comparable.

### Circular aspect statistics

- **Decision:** for the Earth Engine derived band `aspect`, the window statistics `mean`, `std`, and `var` are circular statistics, in degrees.
  - `mean` is the circular mean direction in [0, 360). `std` is the circular standard deviation sqrt(−2 ln R), where R is the mean resultant length, clamped to at most 1. `var` is the square of `std`.
  - Pixels with slope 0 are not used for these three statistics. When the window has no pixel with slope above 0, or R is below 1e-12, the three values are null.
  - The adapter reduces the sine and cosine of the aspect (a companion image, masked to slope above 0) in the same `ee.Dictionary` request as the other statistics, with the same region, scale, CRS, and `bestEffort`. Python computes the circular values from the means and the count (`_circular_aspect_statistics`). No request is added, and column names do not change.
  - The other statistics of aspect (`median`, `min`, `max`, quantiles, `sum`, `mode`, `point`) stay linear, and the documentation says so. QC values come from the linear window branch.
- **Reason:** aspect is a direction. The linear mean of 350° and 10° is 180°, the opposite direction, so the linear mean, standard deviation, and variance were wrong for slopes that face north.
  Flat pixels: a live check on 2026-10-09 showed that `ee.Terrain.aspect` gives flat pixels the aspect 0 and does not mask them. Without the exclusion, flat pixels would pull the circular mean towards north.
- **Rejected:**
  - Linear statistics only, with a documentation note: the values are wrong where users need them most, on slopes that face north.
  - The circular mean only (with linear `std` and `var`): `std` and `var` of directions that cross north are also wrong.
  - New derived bands for northness and eastness (cos and sin of aspect): useful, but a new feature with new columns. It does not correct the existing `aspect` statistics.
  - Hidden extra bands in the main image: multi-band reductions and tile exports would include them.

### Tile size

- **Decision:** both adapters build a raster tile with n = max(1, floor(`window_size_m` / pixel size + 0.5)) pixels per side: the rule that the README states, round(window / pixel), with ties rounded up.
  - The pixel size is `resample_m` when it is set. Without it, Earth Engine tiles use the native scale. Local tiles without `resample_m` keep the source raster's own grid.
  - The tile corners lie on multiples of the pixel size in the point's UTM zone. When n is odd, the pixel that contains the point is the centre pixel. When n is even, the tile centre is the grid line nearest to the point.
  - One pair of helpers in `geo.py` (`tile_pixel_count`, `tile_top_left`) computes n and the corners for both adapters, and the Earth Engine tile-size check uses the same n.
  - The Earth Engine adapter requests the exact pixel grid from `getDownloadURL` (`crs`, `crs_transform`, and `dimensions`), not a region and a scale. With a region, Earth Engine adds every pixel that the region touches: a live check on 2026-10-09 gave tiles of 7 × 7 pixels and of 7 rows of 6 pixels instead of 6 × 6 at the 30.92 m native scale of the GLO-30 DEM. On the same grid, the pixel values are the same with both requests.
- **Reason:** the two adapters used different rules. Earth Engine tiles always had an even size, 2 · round(window / (2 · pixel)) (6 × 6 for 200 m at 30 m). Local tiles had round(window / pixel) pixels (7 × 7), and for odd sizes the point was in the centre pixel for only about one point in four.
  The README rule gives the size closest to the requested window, and a centre pixel that contains the point. It keeps a fixed size for each window and pixel size, and the alignment on one pixel grid, which CNN inputs and comparisons between data sources need.
  Ties round up: Python's `round()` rounds a tie to the even number (`round(2.5)` is 2), so a tie would round down or up depending on the number.
- **Rejected:**
  - The even rule 2 · round(window / (2 · pixel)) in both adapters: Earth Engine tiles would not change, but the tile side would differ from the window by up to one pixel, and the point would be at a pixel corner, not inside a centre pixel.
  - Document the difference between the adapters only: tiles from the two data sources would not be comparable.

### Pixel size in metres

- **Decision:** envoi converts the pixel width of a local raster to metres with one rule (`geo.pixel_size_m`).
  - Geographic CRS: the pixel width in degrees multiplied by 111 319.49 m per degree, the length of one degree at the equator.
    Earth Engine uses this value as the nominal scale of assets in degrees.
  - Projected CRS: the pixel width multiplied by the CRS's linear unit factor. Other CRS: no value (`null`).
- **Consequences:**
  - Away from the equator, the value is larger than the east–west width of a pixel on the ground, by the factor 1/cos(latitude) (2 at 60°).
  - The rule gives the sidecar field `native_spatial_resolution_m` of local datasets and the catalog field `resolution_m` that `_inspect_raster` fills in.
  - The same value is the threshold below which the local adapter widens a statistics window to two pixels.
    Away from the equator, a window a little wider than the true east–west pixel width can therefore also be widened.
- **Reason:** before, both fields held the pixel width in the units of the CRS (for example 0.001 for pixels of 0.001°), although their names say metres.
  One value for each raster, by the rule that Earth Engine uses, keeps the local and Earth Engine values comparable for data in degrees.
- **Rejected:**
  - A pixel width that depends on the latitude (at the raster centre, or at each point): the sidecar and catalog fields hold one value for each raster,
    and the value would differ from Earth Engine's nominal scale for the same data.

## 2026-10-12 — Pipeline follow-ups

These decisions came from the follow-ups of the 2026-10-09 pipeline bug fixes (branch `pipeline-bugs`).
The specification of this work is in a task folder that is not committed, so each decision is stated in full here.

### Tile names that differ only in letter case

- **Decision:** `build_tile_filenames` in `_filenames.py`, which both adapters use, gives the stable hash suffix to every sample ID whose tile name is equal to the tile name of another ID when letter case is ignored (a case collision, compared with `str.casefold()`).
  - The suffix is the one that envoi already adds to IDs that it rewrites: `-` and the first 8 hexadecimal characters of the SHA-1 of the ID as written. `Site1` and `site1` become `Site1-<hash>` and `site1-<hash>`, with two different hashes.
  - Every ID of the group gets the suffix, not only the second one, so the name of an ID does not depend on the order of the input rows.
  - An ID that envoi already rewrote (for example because it has a `/`) has its suffix and does not get a second one.
  - IDs without a case collision keep their names.
  - A last check compares the final names with letter case ignored and raises `ValueError` if two are still equal.
- **Reason:** Windows and macOS ignore letter case in file names by default. Two tiles named `Site1-<dataset>.tif` and `site1-<dataset>.tif` are one file there, so the second tile overwrote the first without an error, and the tile manifest listed both tiles as exported.
  Sample IDs come from the user's data, and data sets with IDs that differ only in letter case are valid. The hash suffix keeps both tiles, and the tile manifest gives the file name of each ID.
- **Rejected:**
  - Raise an error for such IDs: it would block data sets whose IDs are case-sensitive.
  - Add the suffix only to the second ID of a group: the name of an ID would depend on the order of the input rows.

### Catalog nodata and crs for local rasters

- **Decision:** the catalog keys `nodata` and `crs` of a local raster fill a value that the file does not declare. A catalog value that differs from the file's value raises `ValueError`.
  - `nodata` is one number. A list raises `ValueError`. A GeoTIFF has one nodata value for all its bands, so one value is enough, and the value that `_inspect_raster` copies from the file always agrees with the file.
  - When the file declares no nodata value, envoi uses the catalog `nodata` for every selected band in every read path: the point sample, the window statistics (the adapter masks the value after `rio_mask`), the native tile, and the `resample_m` tile. The value must fit the data type of every selected band: an integer band needs a whole number (an `int`, or a `float` without a fractional part such as `-9999.0`) in the range of the type; a float band accepts any number or NaN.
  - When the file declares a nodata value, envoi uses it. A catalog value that differs raises. Equal values do not raise, also NaN and NaN, and `-32768` and `-32768.0`.
  - When the file has no CRS, envoi uses the catalog `crs` in every read path. Without a catalog `crs`, envoi raises `ValueError`. When the file has a CRS, envoi uses it. The comparison with a catalog `crs` first compares the EPSG codes with the same `to_epsg()` rule as `_inspect_raster`. When either CRS has no EPSG code, `CRS.equals` decides.
  - `extract()` runs these checks for every local dataset of every run config, with the bands that the run config selects, before the first adapter is built. The adapter runs the same function (`_resolve_local_raster_settings`) and stores its result.
  - The `resample_m` tile reads the file through a `rasterio.vrt.WarpedVRT` that declares the catalog CRS or nodata value, because `reproject()` ignores `src_crs` for a `rasterio.band()` source.
  - The sidecar of a local dataset records `nodata` (a number, `"nan"` for NaN, or null) and `nodata_source` (`"file"`, `"catalog"`, or null). `resolution_m` stays informational.
- **Reason:** before, the adapter ignored a catalog `nodata` and `crs`, although the catalog accepted them. For a file without a declared nodata value, pixels with the documented nodata value counted as data in every statistic, without a warning.
  The read paths find the nodata pixels in different ways: the window statistics get the file's nodata mask from `rio_mask`, the point sample compares each pixel with the nodata value, and the `resample_m` tile gives the value to GDAL. To fill a missing value, each path needs only one more mask or one more setting. To replace a value that the file declares, the window reader would need a rewrite.
- **Rejected:**
  - The catalog value replaces the file's value: the read paths mask the file's value in different ways, so a replacement needs a rewrite of the window reader, and a wrong catalog value would silently change results.
  - Reject the keys `nodata` and `crs` for local rasters: files without a CRS or without a declared nodata value could then not be used correctly.

### Accepted date formats

- **Decision:** `_parse_and_validate_dates` reads only a closed list of date values. Every other value raises `ValueError` before any adapter is built. The message names the user's date column and lists up to 10 different values and the total number.
  - Text: `YYYY`; `YYYY-M` or `YYYY-MM`; `YYYY-M-D` up to `YYYY-MM-DD`; `YYYYMMDD`. Each of these forms can have a time of day after `T` (also the basic form `T080000` and the hour alone) or after one space (then at least `HH:MM`), and a time zone (`Z`, `±HH`, `±HH:MM`, `±HHMM`). The day is the day as written: envoi drops the time and does not convert the time zone.
  - ISO 8601 intervals as before (a value with exactly one `/` whose start is a year, a year-month, or a date, optionally with a time), and `YYYY/MM/DD`.
  - Date and datetime values (also pandas Timestamps and NumPy `datetime64` in an object column), and columns of a datetime type. The day is the day in the value's own time zone, as for text with a time zone.
  - Whole numbers with four digits (1000 to 9999) are years, also as floats (`2021.0`), because a column of years with a missing value is a float column. Whole numbers with exactly eight digits are read as `YYYYMMDD` (`20210615`, `20210615.0`), because CSV readers turn such columns into numbers. Both are checked like the text form: an impossible month or day (`15062021`: month 20) raises the error for dates that do not exist. Other numbers, such as `2021.5` or `202106`, raise.
  - envoi builds `YYYY-MM-DD` text for each value and parses it with `pd.to_datetime(..., format="%Y-%m-%d", errors="coerce")`. A day that does not exist (`2021-02-30`, `2021-13-01`) raises a second `ValueError` that lists the values.
  - The warnings stay: a year or a year-month is read as its first day, an interval is cut to its start, and a dropped time of day gives one summary warning (now also for a time after a space).
  - The web app does not check dates at upload. The run shows the error.
- **Reason:** before, envoi gave every value that was not an interval or `YYYY/MM/DD` to `pd.to_datetime(format="mixed")`, which guesses the form of each value. It reads the first number of `05-06-2021` and `05.06.2021` as the month, so these dates became May 6, 2021 without a warning, also when the data meant June 5. A wrong date selects a wrong image, which gives plausible but wrong statistics. A closed list of unambiguous forms covers ISO 8601 and the common GBIF / Darwin Core values. A value outside the list stops the run, and the user converts it once with the format that they know.
- **Rejected:** keep the guessing of `pd.to_datetime`: it silently gives wrong dates for day-first data. Accept day-first forms with a setting: the user can convert the column with `pd.to_datetime(..., format=...)`, and envoi reads datetime columns.

### Band overrides and column names

- **Decision:** a per-call band list in `datasets` (`{"dataset": [bands]}` or `{"dataset": {"bands": [...]}}`) replaces the catalog's bands, and on Earth Engine every such list gives statistic columns named after the bands.
  - `_normalize_dataset_entry` always sets `bands` in the band overrides. For a list of only derived bands (`{"dem_copernicus_glo30": ["aspect"]}`), `bands` is an empty list, so the shallow merge with the catalog entry replaces the catalog's source bands, and the output has only the derived bands. `_build_image` selects only the derived bands for an empty `bands` list, and the sidecar `config.datasets` records `bands: []`.
  - The band overrides of every per-call list also get the mark `band_naming: "per_band"` (`_BAND_NAMING_KEY` in the Earth Engine adapter). The merged config of that dataset carries it, and the Earth Engine adapter then uses `{band}_{reducer}` and `{band}_point` keys also for one band. For example, `{"dem_copernicus_glo30": ["DEM"]}` gives `dem_copernicus_glo30_DEM_mean_200m`, and `[aspect]` gives `dem_copernicus_glo30_aspect_mean_200m` (the circular aspect statistics replace the `aspect_*` keys, as for any multi-band run).
  - The local adapter already gave band-named columns for every band list, because it decides by the form of `bands` (a list). It ignores the mark.
  - A catalog default (a plain dataset name) keeps its column names, also a catalog default `derived_bands`.
- **Reason:** before, a list of only derived bands kept the catalog's source bands, because the override had no `bands` key, so `[aspect]` also gave the `DEM` columns. And on Earth Engine a one-band list gave the bare names of a single-band catalog default (`dem_copernicus_glo30_mean_200m`), against `docs/advanced_usage.md` and the local adapter. With `[aspect]` alone, a bare `mean` column would have looked like the mean elevation. A column name that says which band was selected cannot be misread.
- **Rejected:** band-named columns only for derived-only lists: the two adapters would still differ for one-band source lists. Change the documentation to the old Earth Engine behavior: the bare name hides which band was selected.

### Input columns with output names

- **Decision:** `extract()` raises `ValueError` before any adapter is built when an original input column name (before envoi renames the id, coordinate, and date columns) is the name of a column that a tabular run config of the same call adds (`_validate_no_output_column_names` in `_input_validation.py`).
  - For a dataset `d` of a tabular run config with a window of `N` m, a column matches when it is `d_point`, or when it starts with `d_` and its remaining part ends with `_Nm` or `_point`. These patterns cover the statistic columns and the QC columns.
  - A column that starts with `e_` for a longer catalog dataset `e` whose name starts with `d_` belongs to `e` and does not match for `d` (`dem_copernicus_glo30_mean_100m` is not a column of a dataset `dem`).
  - `<latitude column>_wgs84` and `<longitude column>_wgs84` match in a tabular run config when envoi reprojects the coordinates (the same condition that adds these columns to the output).
  - Raster run configs add no columns, so they are not checked.
  - The message names the columns and the outputs, and says to remove or rename the columns, for example when the input is the output of an earlier envoi run.
- **Reason:** before, envoi replaced the values of an input column with a statistic name without notice, moved an input column with a QC name out of the statistics table, and stopped with an unclear error for a `_wgs84` column after the extraction had used its time and Earth Engine quota. Researchers often run envoi again on the output of an earlier run, so these names occur in practice.
- **Rejected:** overwrite without notice (the old behavior): the user's data is lost silently. Rename the input column automatically: the output would have columns that the user did not name, and scripts that read them would break without notice.
