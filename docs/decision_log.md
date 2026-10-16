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

## 2026-10-08 — Public hosting of the web app on SciLifeLab Serve

These decisions came from the plan and the reviews for a public instance of the web app on SciLifeLab Serve.
In that instance, each user uploads their own Earth Engine service-account key.
`docs/architecture.md` ("Web-app execution model") describes the result.

### Earth Engine identity and process isolation

- **Decision:** each user uploads their own service-account key. The web app runs each extraction job in its own worker process, which initializes Earth Engine with that key only.
  The Streamlit server process never calls `ee.Initialize()`.
  The key stays in the session's upload in server memory until the browser session ends, and in the worker process while the job runs.
  The web app never writes the key to disk, to a log, to a command line, or to an environment variable.
- **Reason:** `earthengine-api` keeps its credentials in one module-global state object, and `ee.Initialize()` replaces them for all threads of the process.
  Only a process boundary keeps the keys of concurrent users apart.
- **Rejected:**
  - An application-owned service account. Serve documents no secret store, and all users would share the quota of one project.
  - Per-user Google login (OAuth). It needs secrets on Serve, Google app verification, and a project ID from each user.
  - A lock around `ee.Initialize()`. A lock does not isolate the credentials during the later requests.
- **Consequences:**
  - Users must be able to create a service-account key. Some organizations forbid this.
  - `init_gee()` gets the public keyword argument `credentials_json`. For an invalid key, it raises a fixed `ValueError` after the `except` block, without a chained exception, because the error of the key parser can contain the key.
    This is a deliberate exception to "Chain converted third-party exceptions" in `docs/coding_guidelines.md`.

### Worker process and message channel

- **Decision:** the job manager starts the worker with `subprocess.Popen([sys.executable, "-m", "envoi_webapp.worker"])`.
  It writes one pickled `JobRequest` to the worker's stdin and closes stdin. The worker answers with ASCII JSON lines on a copy of its original stdout.
  `envoi_webapp/job_protocol.py` defines the request and the messages, so the module that runs as `__main__` defines no class that crosses the boundary.
  Local mode uses the same worker.
- **Reason:** the transport is explicit and testable. Pickle is safe here, because both ends run the same envoi installation, and the worker never unpickles data from a user.
  One execution path for both modes also gives local users "Cancel" and results that stay after a rerun.
- **Rejected:**
  - `multiprocessing` with "spawn". It imports the parent main module again, which under `streamlit run` belongs to Streamlit, and it hides the transport of the arguments.
  - Threads. They share the Earth Engine state.
  - Celery, Redis, or a database. Serve does not allow an attached database, and the job load does not need one.

### Result delivery

- **Decision:** in hosted mode, the worker moves the outputs into one ZIP archive in the job workspace on the container disk.
  `st.download_button` with a callable reads the archive when the user clicks the button.
- **Reason:** this needs no secrets and no storage that other apps share.
- **Rejected:**
  - The Serve project volume. All apps of the project share it, it holds 1–5 GB, and it is not for user data.
  - Object storage with signed URLs. Serve has none, and it needs secrets.
- **Consequences:** a click reads the whole archive into memory. The workspace size limit bounds that read.
  The worker deletes each output file right after it is in the archive, so the disk use stays near the size of the outputs.

### Hosted limits and retention

- **Decision:** hosted mode applies the limits in `HostedLimits` (`src/envoi_webapp/settings.py`). The user decided that these are starting values:
  - upload: 50 MB and 10,001 lines (one header line and 10,000 rows),
  - 10,000 input rows and 10 data-product rows,
  - tabular request budget: points × the number of window sizes over all tabular rows ≤ 20,000. The point value counts as one window size,
  - raster tile budget: points × the number of window sizes over all raster rows ≤ 1,000,
  - window size: at most 10,000 m (tabular) and 2,000 m (raster),
  - run time 60 minutes and workspace size 1 GB for each job,
  - disk budget 4 GB for all workspaces. A new job starts only if (the sizes of the finished workspaces) + (running jobs + 1) × 1 GB ≤ 4 GB,
  - 2 concurrent jobs on the server (`ENVOI_WEBAPP_MAX_JOBS`), one running job for each session, and one running job for each key,
  - abandon time-out: the job stops after 10 minutes without a status request.
- **Decision (retention):** the web app deletes the results of a succeeded job 10 minutes after the first "Download results" click, and at the latest 30 minutes after the job ends.
  A failed, cancelled, or stopped job gives no partial results. The web app deletes its workspace when the job ends.
- **Reason:**
  - Serve gives 2 vCPU and 4 GB RAM by default, and it accepts uploads of up to 100 MB.
  - The Earth Engine adapter makes one request for each point and window, so the budgets bound the run time.
  - Streamlit sees the download click, not the end of the download. The 10-minute grace period lets the user try the download again. Short retention frees the disk budget sooner.
  - Browsers slow down timers in background tabs. A shorter abandon time-out could stop a job that a user still watches.
- **Rejected:**
  - One environment variable for each limit. Most of them would never be set, and each one adds configuration to document and to test.
  - A longer retention, or no time limit for the results. The results would hold the disk budget longer, and the IDs and coordinates of a user's points would stay on a shared server after the user needs them.
- **Consequences:**
  - Load measurements and the reply from Serve can change the values.
  - Only `ENVOI_WEBAPP_MAX_JOBS` is an environment variable. The other limits are constants, so that no unused configuration exists.
  - Data products with many bands can reach the 1 GB limit inside the tile budget. Such a job stops without results.

### Final job state

- **Decision:** the first final state of a job wins: `succeeded`, `failed`, `cancelled`, or `stopped` (by a limit).
  "Cancel" and the housekeeping thread set the state before they stop the worker process.
  When the message channel ends without a `done` or `error` message, the job gets `failed` only if it has no final state yet.
- **Reason:** a cancel or a limit can come at the same time as the last message of the worker. One fixed rule makes the result predictable and testable.
- **Rejected:** the last final state wins. A `done` or `error` message that the worker sends after a cancel or a stop would then replace `cancelled` or `stopped`, so a job that the user cancelled, or that a limit stopped, could show results.
- **Consequence:** a cancelled or stopped job shows only its reason. The worker stops before it can write the run log.

### Key hash for the per-key limit

- **Decision:** the job manager keeps only a SHA-256 hash of each key. The hash covers `client_email`, `private_key_id`, and `private_key`.
  The limit of one running job for each key and "Cancel the earlier job" (`cancel_for_key()`) use this hash.
- **Reason:** the `client_email` is not secret. With a hash of the email only, a forged key with the email of another user could block or cancel the job of that user.
- **Rejected:** a hash of `client_email` only.
- **Consequence:** two keys of the same service account count as two different keys.

### One job manager per process, outside the Streamlit cache

- **Decision:** `envoi_webapp.jobs.get_job_manager()` keeps one `JobManager` for each server process in a module-level variable, under a lock. The web app does not keep the manager in `st.cache_resource`.
- **Reason:** any browser client can send Streamlit a `clear_cache` message, and Streamlit then clears `st.cache_resource` for all users (checked in Streamlit 1.58.0 and 1.65.0). The menu setting `toolbarMode = "viewer"` only hides the menu item.
  A second manager would start with an empty job table, so the hosted job limits, the per-key limit, and the disk budget would start again from zero, and all pages would lose the link to their jobs (code review 2 of the hosting work, finding 1).
- **Rejected:** `st.cache_resource`, and a module-level variable in `app.py` (local mode runs `app.py` again as the page script on each rerun, so the variable does not persist).
- **Consequence:** in local development with the file watcher on, a change to a file in `src/envoi_webapp/` makes Streamlit import `jobs.py` again, which creates a second manager. The hosted image turns the file watcher off.

### Private workspace root in local mode

- **Decision:** in local mode, each server process creates its own workspace root with `tempfile.mkdtemp(prefix="envoi-webapp-jobs-")`, with mode `0o700` and a unique name.
  In hosted mode, the root is the fixed folder `envoi-webapp-jobs`. On POSIX, the job manager refuses that folder when it is a symbolic link, belongs to another user, or gives permissions to the group or to others.
  In both modes, the worker environment has `PYTHONSAFEPATH=1`.
- **Reason:** `python -m` puts the working folder, which is the job workspace, first on `sys.path`.
  With a fixed shared root, another user of the same computer could create the root first and put a module file into a new workspace.
  The worker would then import that module while it holds the key (code review 1 of the hosting work, finding 1).
- **Rejected:** the fixed folder `envoi-webapp-jobs` in the temporary folder in local mode, without an owner check.
- **Consequences:**
  - `PYTHONSAFEPATH` works only on Python 3.11 and later. On Python 3.10, only the private root protects local mode.
  - Each start of the local web app leaves one empty `envoi-webapp-jobs-*` folder in the temporary folder, because Streamlit never calls `JobManager.shutdown()`.
  - Hosted mode is refused on Windows. There, the worker would keep other credential locations of the server account (`APPDATA`, `USERPROFILE`), and the job manager cannot check the owner and the permissions of the root.

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
  - When the file declares no nodata value, envoi uses the catalog `nodata` for every selected band in every read path: the point sample, the window statistics (the adapter masks the value after `rio_mask`), the native tile, and the `resample_m` tile. The value must fit the data type of every selected band: an integer band needs a whole number (an `int`, or a `float` without a fractional part such as `-9999.0`) in the range of the type; a float band accepts NaN, infinity, or a number in the range of the type. envoi converts the value to the data type of the bands once (`_resolve_local_raster_settings`), and every read path and the sidecar use the converted value: a `float32` band stores `-3.4e38` as `-3.3999999521443642e38`, and with NumPy 1.x the point sample would not find its nodata pixels with the unconverted value.
  - When the file declares a nodata value, envoi uses it. A catalog value that differs raises. Equal values do not raise, also NaN and NaN, and `-32768` and `-32768.0`. envoi converts the file's value to the data type of its band in the same way (`_per_band_nodata`, `_nodata_in_band_type`), compares the raw values with the catalog value, and records the converted value.
  - When the file has no CRS, envoi uses the catalog `crs` in every read path. Without a catalog `crs`, envoi raises `ValueError`. When the file has a CRS, envoi uses it. The comparison with a catalog `crs` first compares the EPSG codes with the same `to_epsg()` rule as `_inspect_raster`. When either CRS has no EPSG code, `==` of the two rasterio CRS objects decides (`CRS.equals` does not exist in rasterio 1.3, which envoi supports).
  - `extract()` runs these checks for every local dataset of every run config, with the bands that the run config selects, before the first adapter is built. The adapter runs the same function (`_resolve_local_raster_settings`) and stores its result.
  - The `resample_m` tile reads the file through a `rasterio.vrt.WarpedVRT` that declares the catalog CRS or nodata value, because `reproject()` ignores `src_crs` for a `rasterio.band()` source. The VRT keeps the grid of the file (`transform`, `width`, `height`), so a rotated or skewed raster is resampled only once.
  - The sidecar of a local dataset records `nodata` (a number, the text `"nan"`, `"inf"`, or `"-inf"` for a value that is not finite, or null) and `nodata_source` (`"file"`, `"catalog"`, or null). `resolution_m` stays informational.
- **Reason:** before, the adapter ignored a catalog `nodata` and `crs`, although the catalog accepted them. For a file without a declared nodata value, pixels with the documented nodata value counted as data in every statistic, without a warning.
  The read paths find the nodata pixels in different ways: the window statistics get the file's nodata mask from `rio_mask`, the point sample compares each pixel with the nodata value, and the `resample_m` tile gives the value to GDAL. To fill a missing value, each path needs only one more mask or one more setting. To replace a value that the file declares, the window reader would need a rewrite.
- **Rejected:**
  - The catalog value replaces the file's value: the read paths mask the file's value in different ways, so a replacement needs a rewrite of the window reader, and a wrong catalog value would silently change results.
  - Reject the keys `nodata` and `crs` for local rasters: files without a CRS or without a declared nodata value could then not be used correctly.

### Local window coverage

- **Decision:** for a local window, `coverage_pct` is the share of the window that has valid data, and `had_nodata` is `True` when that share is below 100 %, or when the window has no valid pixel.
  - The total is the number of pixels of the file's pixel grid whose centre is inside the window polygon (the `all_touched=False` rule of `rio_mask`), also beyond the edge of the raster. The grid is continued beyond the raster with the file's transform. The pixels beyond the edge have no data, so they count as missing pixels.
  - The valid pixels (`n_pixels`) are the pixels inside the window and the raster that are not nodata and are finite in every selected band, as before. For a single band, pixels that are NaN or infinite and are not the nodata value now also count as missing, as for several bands.
  - `fetch_values` counts the pixels of the crop with `geometry_mask` on the crop's transform and shape, as `rio_mask` does, and adds the pixels beyond the edge (`_count_window_pixels_beyond_raster`).
- **Reason:** the crop of `rio_mask` is the bounding box of the window, clipped to the raster. With the crop as the total, the pixels of the box outside the window counted as missing: a window without missing data had 82.6 % (100 m at 10 m pixels), `had_nodata` was `True`, and a `min_coverage_pct` above this gave a false "Low coverage" warning. The part of a window beyond the raster edge did not count, so a window half outside the raster still had a high coverage. Earth Engine divides by the area of the window, so the part outside the image counts as missing there too.
- **Rejected:** the crop of `rio_mask` as the total (the old rule). Only the pixels inside the raster as the total: a window half outside the raster would have 100 %.

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

### Image selection per point

- **Decision:** for Earth Engine collections with `collection_date_policy` `nearest` or `contains`, a point that has images, but no own image in the date range that envoi selects for the batch, gets an own image near its date, which Earth Engine selects in the point's own request.
  - Terms. Batch index: the start and end times of the images at the points of a batch, which envoi requests once per batch. Point image: an image whose footprint contains the point (after `filterBounds`). Own image: a point image that also passes the `UTM_ZONE` filter when `use_utm_zone` is set. Centre day C: the sample date (without a date, the latest start of the batch index), limited to the range from the first start to the latest start of the batch index, at 00:00 UTC.
  - Fast path: the own images in the batch-selected range, as before. For `contains`, the range ends at most at C + 1 day, so that the fast path never selects an own image that starts after the centre day when another point's image lasts longer.
  - Fallback, only with a point geometry and a batch index, inside the existing `ee.Algorithms.If`, first with a window W of 32 days, then of 366 days, in whole UTC days:
    - `nearest`: the own image with a start in `[C − W, C + W + 1)` with the fewest whole days between its start day and C, and the earlier start between equal gaps. One numeric sort key (day gap × 10¹¹ + milliseconds since the window start) decides, because Earth Engine does not promise a stable sort.
    - `contains`: the own image that contains C (see "Day rule"), with the latest start; else the latest start in `[C − W, C + 1)`; else the earliest start in `[C + 1, C + 1 + W)`.
  - No own image in the windows: the image stays null, and the point is a failed point. This includes a point whose point images all fail the `UTM_ZONE` filter, so a wrong `use_utm_zone` stays visible. A point without any point image keeps the masked image of the 2026-10-09 "Failed points" decision.
  - Reported dates: the statistics request returns `system:time_start` and `system:time_end` of the selected image as one more branch of its `ee.Dictionary`, with no extra request. `image_time_start`, `image_time_end`, and the sidecar `date_info` come from these values. In raster runs, each tile worker requests the same two values in one small request before the download; a failed request makes a failed tile and removes an older tile of the same name. A point without point images has no date fields. `date_clamped` and `date_source` keep their rules: they compare the sample date with the batch index.
  - Image assets and `collection_date_policy: mosaic` do not change.
- **Reason:** in collections whose images have different dates at different places, one date for the batch does not fit every point. Live check on 2026-10-12: in June 2021, Sentinel-2 had images at Uppsala on 06-02, 05, 07, 10, 12, 15, 17, … and at Göteborg on 06-01, 03, 06, 08, 11, 13, 16, …, with no common day. With the sample date 2021-06-16 in one run, the Uppsala point failed, although each point alone had values. The QC dates also named the batch-selected image, not the image that the point used.
- **Costs (live, 2026-10-12):**
  - Selection on the server for every point, over the whole collection at the point: 12 s (Sentinel-2, 2 671 images) and 27 s (Landsat 8-day NDVI composites, 1 957 images) per point.
  - The fallback, only for the points that need it: about 1.6 s per request with the 32-day window and 4.3 s with the 366-day window (probe), or about 1.1 s more than a fast-path request of the same batch (Sentinel-2: 2.6 s against 1.4 s with `nearest`, 3.2 s against 2.1 s with `contains`).
  - The fast path, measured with the old and new code run alternately (Sentinel-2 at Göteborg): +0.15 s per request (median of 20 requests: 1.55 s before, 1.69 s after; +9 %), and no increase end to end (median of 6 runs: 16.4 s before, 15.6 s after; ten ERA5 points: 0.45 s before, 0.41 s after, per point). The end-to-end time of one Sentinel-2 point is mostly the batch-index request, which varies from 11 to 39 s between runs.
- **Rejected:**
  - Select the image on the server for every point: 6 to 27 s per point for long collections.
  - Document the limitation only: points of such collections would stay failed points, and the QC dates would not name the image used.

### Day rule

- **Decision:** envoi compares image times with sample dates as whole days in UTC.
  - An image starts on or before day D when its start is before D + 1 day.
  - An image contains day D when it starts on or before day D and its end is after the start of day D (00:00 UTC).
  - An image without an end, or with an end on its start day, ends at its start day + 1 day. So such an image contains only its start day.
  - The rule applies to the `contains` selection of the batch-selected range (`_resolve_date_filter_range`), to `date_clamped` and `date_source` for `contains` (`_resolve_date_source`), and to the per-point fallback: the whole-day gap of `nearest` and the containing image of `contains`.
  - The batch-selected image of `nearest` is still the image whose start time is nearest to the sample date (00:00 UTC of its day). Its fast-path range is the whole start day of that image.
- **Reason:** sample dates are days without a time of day (envoi reduces each date value to one day). Images start at any time of the day, for example a Sentinel-2 image at 10:30. A comparison of times would treat an image that starts at 10:00 on the sample day as an image after the sample date, so `contains` selected the image before it or marked the date as clamped. In the fallback, whole days make the images of the day before and the day after the sample date equally near, whatever their time of day, and the earlier start then wins.
- **Rejected:** compare the times as they are: the result depends on the time of day of the image, which the sample date does not have.

## 2026-10-13 — Warning for far fallback images

This decision adds a warning to the per-point image selection of the 2026-10-12 entry ("Image selection per point", branch `pipeline-bugs`). The user approved it on 2026-10-13.

### Warning for far fallback images

- **Decision:** `extract()` gives one warning for each dataset and window (tabular and raster) when at least one point used a fallback image that is more than 32 days from its sample date.
  - A fallback image is an image that the per-point fallback of the 2026-10-12 decision selected, because the date range that envoi selected for the batch had no own image of the point. The Earth Engine server gives this as a flag (`fallback`) in the `image_dates` value of the point's request, from the condition of the fast path. The flag adds no request, and the adapter does not guess it from the dates.
  - The gap is the number of whole UTC days between the sample day and the image (the day rule of 2026-10-12). With `nearest`, it is the days to the image's start day. With `contains`, it is 0 when the image contains the sample day, and otherwise the days to the nearest day that the image contains: its start day for a later image, and its last day for an earlier image (an end at 00:00 does not contain the day that starts then).
  - Only points with a sample date that is not clamped count. A clamped date is outside the dates of the batch's images, and `date_clamped` already reports it.
  - The Earth Engine adapter keeps the gap in the point's meta dict under the internal key `fallback_image_gap_days`. No output writes it: the QC table and the sidecar summaries read only the keys that they name.
  - The warning gives the dataset, the window ("point values" for point-only statistics), the number of these points, the largest gap in days, the IDs of up to 10 of them, and where the image dates are (the QC columns `<dataset>_image_time_start_<N>m` and `<dataset>_image_time_end_<N>m`, or the sidecar `date_info` in raster runs). The sidecar records it in `warnings` under `far_image_<dataset>_<N>m` (`far_image_<dataset>_point` for point-only statistics).
  - The limit of 32 days is the first search window of the fallback.
- **Reason:** a fallback image can be weeks or months from the sample date, for example when a point's own images are rare or have a long gap. Its statistics are then plausible but can describe another season. The QC date columns show the image dates, but nothing told the user to look at them. The fallback is normal for collections whose images have different dates at different places, and the image is then often near the date: in the live Sentinel-2 checks of 2026-10-12 (Uppsala and Göteborg), the fallback image was one day from the date. A limit of 32 days keeps such runs without a warning.
- **Rejected:**
  - Warn for every point that used a fallback image: the warning would come in most Sentinel-2 runs, and users would learn to ignore it.
  - A QC column with the gap: it changes the output contract of the QC table, and the QC date columns already give the image dates.

## 2026-10-13 — Day rule for the nearest policy

This decision extends the "Day rule" of the 2026-10-12 entry to the `nearest` policy (follow-up FU28, branch `pipeline-bugs`). The user asked for it on 2026-10-13.

- **Decision:** the batch-selected image of `nearest` and its `date_clamped` and `date_source` also use whole UTC days (`_find_nearest_timestamp`).
  - The nearest image is the one with the fewest whole days between its start day and the sample day. Of two equally near images, the earlier one, as in the per-point fallback.
  - A date is clamped only when its day is before the start day of the first image or after the start day of the last image.
- **Reason:** with exact times, a date on the start day of the first image was clamped when that image started later in the day, and was then also left out of the far-image warning. An image that started at 23:00 on the day before the sample date also won over an image of the sample day at 10:00. Sample dates have no time of day, so both results depended on a time that the date does not have. The per-point fallback already counted whole days, so the two selections now follow one rule.
- **Consequence:** collections whose images start at 00:00 (all built-in date-based datasets) select the same images as before. Collections with images later in the day can select another image when two images are within one day of the date.
- **Rejected:** keep exact times for `nearest` and change only `date_clamped`: the selection and the flag would then follow different rules.

## 2026-10-14 — Hosted capacity on Serve's default resources

This entry changes three values of "Hosted limits and retention" (2026-10-08). The user asked for more concurrent users than two, on Serve's default 2 vCPU and 4 GB. The user approved these changes on 2026-10-14 and 2026-10-16, after load tests of the container image and code review 3 of the hosting work.

- **Decision:**
  - 3 concurrent jobs on the server by default (was 2). `ENVOI_WEBAPP_MAX_JOBS` still overrides it.
  - Workspace limit 500 MB for each job (was 1 GB).
  - Disk budget 2 GB for all workspaces (was 4 GB). Three running jobs reserve 1.5 GB and leave 0.5 GB for finished results.
  - The web app reads the dataset catalog once per server process (`dataset_catalog()` in `envoi_webapp/helpers.py`), not on every page run.
- **Reason (measurements in local Docker with 2 vCPU and 4 GB, recorded in the hosting plan):**
  - An extraction worker uses about 0.35 CPU cores and 0.2-0.27 GB, for tabular and raster jobs, and also for satellite embeddings (64 bands). With elevation data, four jobs ran 10 % slower than one, and six jobs used 1.9 of 2 cores and ran twice as long.
  - The Streamlit server is one Python process. Its page runs use at most about 1.1 cores, when many sessions are active. Three workers (about 1 core) and a moderately busy page fill the 2 vCPU.
  - A download click reads the whole archive into the memory of the Streamlit process, and Streamlit frees it only after two later script runs anywhere in the app. Every result on disk can therefore be in memory at the same time. So the disk budget, not the workspace limit, bounds the memory of downloads.
    - With 2 GB, the worst case is about 3.8 GB: 0.2 GB for the server, 2 GB of downloads, 0.5 GB for a repeated click, and about 1.1 GB that large uploads leave. Running jobs reserve part of the budget, so the worst case is lower while jobs run.
    - With 4 GB, the worst case was about 5.7 GB. Running out of memory restarts the app, which ends all jobs and deletes all results.
  - With 500 MB workspaces, three running jobs fit in the 2 GB budget with room for finished results. One satellite-embedding tile of 2,000 m is 3.7 MB, so about 140 such tiles fit in one job.
  - Parsing the bundled catalog took about 33 ms of a hosted page run of 52 ms. With the cache, a hosted page run takes 19 ms. With 30 sessions that each rerun the page every 3 s, the median page response fell from 3.6 s to 0.17 s.
  - The web app never registers datasets with `update_catalog()`, so the catalog cannot change while the server process runs. The cache is in `helpers.py`, because local mode runs `app.py` again as the page script on each rerun (see "One job manager per process, outside the Streamlit cache").
- **Rejected:**
  - 4 or more concurrent jobs on 2 vCPU: the workers and the page runs compete for the CPU.
  - Keep the 4 GB disk budget and test several large downloads on Serve first: an out-of-memory restart affects all users, and the smaller budget costs little in normal use.
  - Read only the required columns of the uploaded CSV, to reduce the memory of large uploads: the other input columns appear in the statistics table, so this changes the output contract.
  - A waiting queue instead of the "try again in a few minutes" message: a larger change. It can come later.
- **Consequences:**
  - A new job gets the "not enough free disk space" message when the running jobs and the finished results of other sessions leave less than 500 MB of the budget. This happens only with large results, and finished results wait at most 30 minutes.
  - Raster rows of data products with many bands reach the workspace limit at half the size before. The limit stopped a satellite-embedding job of 160 tiles at 502 MiB, after about 7.5 minutes, with the message that names the limit.
  - With more resources from Serve (for example 4 vCPU and 8 GB), `ENVOI_WEBAPP_MAX_JOBS` can be raised to about 6. That also needs a larger disk budget (a constant in `settings.py`), because 6 running jobs reserve 3 GB.
  - A running local web app shows a changed `ee_catalog.yml` only after a restart.
  - Not measured yet: a large download through Serve's proxy.
