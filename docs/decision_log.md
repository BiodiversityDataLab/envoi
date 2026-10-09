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
