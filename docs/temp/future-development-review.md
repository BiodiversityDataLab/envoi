# Envoi future-development architecture review

**Status:** working review document  
**Reviewed:** 2026-09-14  
**Branch:** `code-review`  
**Companion:** [architecture-overview.md](architecture-overview.md)

## Scope and intended development

This review evaluates the current architecture against two planned development arcs:

1. Add independent integrations for CHELSA, Copernicus, and national or regional data services while retaining `envoi.extract()` as the consistent user entry point.
2. Turn the local Streamlit MVP into a publicly accessible service, potentially executing Earth Engine work under each user's credentials.

The review is architectural and diagnostic. It does not implement the proposed changes.

## Executive assessment

`extract()` can remain Envoi's stable public interface. The present adapter registry and batch calls provide a useful starting seam, but that seam is not yet a safe extension contract. The most important work is to separate stable orchestration from provider-specific validation, authentication, capabilities, result formats, and operational policy.

The current Streamlit app can also remain the frontend. Publishing it is not merely a hosting task, however: Earth Engine state and output storage are shared at process level, so the current application is not safe for concurrent users with independent credentials.

The critical findings are:

1. Earth Engine credentials are not isolated between concurrent users.
2. Hosted output paths are server-local, user-controlled, and collision-prone.
3. The public app has no workload admission or global concurrency control.
4. The adapter contract is incomplete and unvalidated.
5. Generic configuration and catalogs contain provider-specific assumptions.

## Findings

### F1 — Critical: Earth Engine credential contexts are process-global

**Evidence**

- `src/envoi_webapp/helpers.py:318-350` writes each uploaded service-account JSON to a temporary file, calls `init_gee()`, and then calls `extract()` in the same process.
- `src/envoi/auth.py:111-114` calls `ee.Initialize(credentials)`.
- `src/envoi/adapters/earth_engine/_image.py:27-91` maintains a module-global `_gee_initialized` flag and accesses/modifies the Earth Engine SDK's shared HTTP session.
- Earth Engine documents that calling initialization again overwrites earlier settings: <https://developers.google.com/earth-engine/apidocs/ee-initialize>.

**Impact**

Streamlit session state does not isolate Earth Engine's Python module state. Concurrent users supplying different credentials can overwrite the active SDK configuration or run work under an unintended identity, project, or quota context. A module-level boolean can also report initialization as complete after another request changes that state.

**Recommendation**

Choose the production authentication model explicitly:

1. **Application-owned identity:** simplest initial public service. The deployment uses one workload/service identity; application accounts, quotas, and audit records control end-user access.
2. **End-user OAuth:** use when Earth Engine work must consume the user's identity and quota. Earth Engine describes client/user authentication as the alternative to application-owned service accounts: <https://developers.google.com/earth-engine/guides/app_engine_intro>.
3. **Uploaded service-account key:** if temporarily retained, execute each job in a separate process or container and destroy its filesystem and memory state after completion.

Do not rely on locks around `ee.Initialize()` alone: a lock prevents simultaneous initialization but does not isolate credentials during the subsequent network calls.

Google recommends avoiding user-managed service-account keys where possible and treating private keys like passwords: <https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys>.

### F2 — Critical: hosted output paths are unsafe and do not deliver files to users

**Evidence**

- `src/envoi_webapp/app.py:1217-1231` exposes a text field and native operating-system folder chooser.
- `src/envoi_webapp/helpers.py:129-144` expands and creates the supplied path anywhere the server account has permission.
- `src/envoi_webapp/helpers.py:198-201` creates deterministic batch identifiers such as `extract_01_dem`.
- `src/envoi_webapp/app.py:1316-1318` displays result paths rather than delivering their contents.

**Impact**

In a hosted app, the selected directory is on the server rather than the user's computer. Multiple sessions default to the same home-directory location and can overwrite or observe one another's predictable artifacts. User-entered paths can also probe or populate arbitrary writable server locations. No retention or cleanup policy exists.

**Recommendation**

- Remove the output-directory control in hosted mode.
- Allocate an unpredictable job ID and private workspace under an application-controlled root.
- Make output creation atomic where practical.
- Package small results into a downloadable archive.
- Put large artifacts in object storage and return an expiring signed URL.
- Delete temporary workspaces after download or a documented TTL.
- Return an artifact manifest from the core rather than exposing server paths as the service API.

Streamlit provides `st.download_button`, but directly supplied data remains in memory while the user is connected, so object storage is preferable for large raster outputs: <https://docs.streamlit.io/develop/api-reference/widgets/st.download_button>.

### F3 — Critical for unrestricted public use: no admission control or job isolation

**Evidence**

- Neither file uploader specifies an application-specific size limit (`src/envoi_webapp/app.py:1153-1158` and `1200-1205`).
- CSV validation has no row-count ceiling.
- Dataset count, number of windows, window dimensions, and estimated output volume are not globally bounded.
- `GeeRasterAdapter` reads a default `max_workers` of 20 (`src/envoi/adapters/earth_engine/adapter.py:140-142`) and creates a pool for each statistics or tile batch.
- Extraction runs synchronously in the Streamlit script thread; there is no queue, cancellation token, job deadline, or persistent state.

**Impact**

A few concurrent users can create hundreds of worker threads, consume shared memory/disk, and exhaust provider quotas. Browser disconnects and Streamlit reruns do not provide a reliable job lifecycle. Streamlit's default per-file upload limit is currently 200 MB: <https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader>.

**Recommendation**

- Define limits for input bytes, parsed rows, datasets, windows, estimated pixels, and artifact bytes.
- Add per-user and global request/concurrency quotas.
- Pass a global worker budget into adapters rather than letting every adapter create its own unrestricted pool.
- Add deadlines and cooperative cancellation.
- Use a background job queue and separate worker processes for production-scale or credential-isolated work.
- Persist job status independently of the browser session.

### F4 — High: the adapter interface is incomplete and unenforced

**Evidence**

- `src/envoi/adapters/base.py` is neither an abstract base class nor a `Protocol`.
- The base class declares `fetch_values`, `fetch_batch`, `fetch_stats_batch`, and `build_dataset_meta`, but not `export_tiles`.
- `src/envoi/extract.py:641` calls `adapter.export_tiles()` unconditionally in raster mode.
- `src/envoi/adapters/__init__.py:6-7` registers any name/class pair without validation.
- The Earth Engine adapter does not provide a usable raw `fetch_values()` implementation despite inheriting the base method.
- No test registers a third-party/dummy adapter and runs it through the public `extract()` entry point.

**Impact**

A new adapter may import and register successfully but fail only after expensive work begins. Every source is implicitly expected to support both tabular statistics and raster tile export, which will not hold for all APIs.

**Recommendation**

Define and validate a formal provider contract. A provider registration should include:

```python
ProviderDefinition(
    name=...,
    adapter_factory=...,
    catalog_schema=...,
    capabilities=...,
    auth_scheme=...,
    reducer_translator=...,
)
```

Capabilities should at least express tabular sampling, raster export, temporal queries, server-side reduction, supported output formats, and asynchronous delivery. Build an adapter conformance suite and run every built-in adapter through it.

### F5 — High: generic configuration parsing contains Earth Engine policy

**Evidence**

- `src/envoi/_config_parsing.py:30` imports `KNOWN_DERIVED_BANDS` from the Earth Engine adapter package.
- `src/envoi/_config_parsing.py:200-213` identifies derived bands centrally and rejects them for every source other than `earth_engine`.
- Per-dataset call overrides are centrally limited to `bands`.

**Impact**

Adding a provider-specific variable, aggregation, transformation, temporal option, or derived product requires modifying generic parsing code. That reverses the intended dependency direction and makes `extract()` a change hotspot.

**Recommendation**

Keep parsing of the common request shape in the core, then ask the selected provider definition to validate and normalize its provider-specific options. Derived variables should be described as transformations/capabilities rather than hard-coded Earth Engine names.

### F6 — High: the catalog schema assumes every source is addressed by a raster-like path

**Evidence**

- `src/envoi/catalog.py:14-18` universally requires `data_source` and `path` and adds only an Earth Engine-specific `data_type` rule.
- Dictionary-catalog errors at `src/envoi/catalog.py:213-223` state that valid sources are only Earth Engine and local despite the internal registry accepting arbitrary strings.
- Catalog dictionaries are not versioned.

**Impact**

Remote providers may require product IDs, variables, versions, API endpoints, temporal resolution, licensing constraints, request formats, or separate discovery/download identifiers. Placing all of this in an unvalidated dictionary makes migrations and compatibility difficult.

**Recommendation**

Introduce a versioned catalog model with common and provider-owned sections:

```yaml
catalog_version: 2
datasets:
  chelsa_bioclim:
    provider: chelsa
    data_type: continuous
    provider_config:
      product: bioclim
      version: "2.1"
      variables: [bio1, bio12]
```

Provide a compatibility loader that translates existing `data_source`/`path` entries into the new internal model. Reject unknown keys where they would otherwise hide configuration typos.

### F7 — High: adapter result dictionaries form an undocumented internal API

**Evidence**

- `fetch_stats_batch()` returns `list[tuple[dict, dict]]`.
- `src/envoi/_output_assembly.py:96-157` discovers columns from arbitrary statistics keys.
- `src/envoi/qc.py:57-78` assumes every point metadata dictionary has `in_extent`, `n_pixels`, `had_nodata`, and `coverage_pct`.
- Return cardinality is not validated immediately after adapter execution.

**Impact**

A provider can return missing fields, inconsistent bands, or the wrong number/order of results. The resulting failure occurs in output/QC code and is difficult to attribute to the provider boundary.

**Recommendation**

Introduce typed models such as `PointResult`, `PointQuality`, `DatasetResult`, and `TileArtifact`. Validate one result per requested point and required fields as soon as the adapter returns. Keep an explicitly namespaced dictionary only for genuinely provider-specific metadata.

### F8 — High: operational failures are conflated with valid no-data results

**Evidence**

- `src/envoi/adapters/earth_engine/adapter.py:1010-1021` catches every exception from a point-statistics future and substitutes an empty result.
- `src/envoi/adapters/earth_engine/adapter.py:1203-1209` catches every tile-export exception and substitutes `None`.
- Detailed failure reasons go to logs, while the normal QC schema mainly represents extent, nodata, and coverage.

**Impact**

A provider timeout, expired credential, quota response, and genuine absence of environmental data may all appear as missing values or zero coverage. This becomes more severe when different provider APIs have different error and retry semantics.

**Recommendation**

Add structured point statuses such as:

- `success`
- `no_data`
- `outside_extent`
- `provider_error`
- `rate_limited`
- `cancelled`

Include safe reason codes and retryability. Support an extraction failure policy: `strict` aborts on operational failure; `best_effort` returns partial data with explicit error records.

### F9 — High: eager provider imports will make optional integrations difficult

**Evidence**

- `src/envoi/adapters/__init__.py:22-26` eagerly imports built-in adapters for registration.
- The register function is internal and is not exported from `envoi.__init__`.
- Ruff 0.15.16 reports `E402` on the late self-registration import.

**Impact**

Every added provider SDK risks becoming an import-time dependency even when unused. One unavailable optional SDK could prevent `import envoi`. Third-party integration has no supported discovery path.

**Recommendation**

- Replace side-effect imports with lazy factories.
- Keep provider dependencies in extras such as `[copernicus]`, `[chelsa]`, and `[all-providers]`.
- Expose a supported registration function.
- When external adapters become useful, discover them through a namespaced `importlib.metadata` entry-point group.

An adapter should normally represent a provider or access protocol, not each individual data product. Products sharing transport, authentication, and query semantics should remain catalog entries under one provider.

### F10 — High: no generic job-scoped execution context exists

**Evidence**

- Adapters are constructed with only a merged dataset specification.
- Earth Engine authentication is performed externally through process-global initialization.
- Output directories, worker settings, HTTP sessions, caching, logging, and cancellation are not represented as one execution dependency.

**Impact**

Provider-specific infrastructure leaks into catalogs or global modules. It becomes difficult to test providers, isolate jobs, manage multiple credentials, or apply uniform service policies.

**Recommendation**

Introduce an `ExtractionContext` containing job-scoped dependencies:

```python
ExtractionContext(
    job_id=...,
    provider_credentials=...,
    artifact_store=...,
    http_clients=...,
    concurrency_budget=...,
    deadline=...,
    cancellation_token=...,
    cache=...,
    logger=...,
)
```

Credentials must not be serialized into run configuration or provenance metadata.

### F11 — Medium: catalog registration is process-global

**Evidence**

- `src/envoi/catalog.py:20-30` defines process-global caches and `_user_catalog_datasets`.
- `update_catalog()` mutates that dictionary for every subsequent extraction in the process.

**Impact**

One server request can change which dataset definitions another request sees. Dataset-name shadowing is particularly risky in a multi-user service.

**Recommendation**

Resolve an immutable catalog snapshot as part of each `ExtractionPlan`. Keep `update_catalog()` for backward-compatible local use, but do not use process-global mutation for user-specific hosted catalogs.

### F12 — Medium: adapter lifetime repeats setup across windows

**Evidence**

- `src/envoi/extract.py:287-291` loops through dataset/window combinations.
- `src/envoi/extract.py:531-532` and `635-636` construct the adapter inside the per-window helpers.

**Impact**

Authentication checks, source metadata lookup, timestamp discovery, downloads, open-file work, and caches may be repeated for every requested window. This will be expensive for remote providers.

**Recommendation**

Build the extraction plan first, create one adapter/session per dataset/provider job, and let it execute all required windows before closing. Adapters may still choose smaller internal batches.

### F13 — Medium: reducer definitions and semantics are distributed

**Evidence**

Reducer knowledge appears in:

- `_config_parsing._ALL_KNOWN_REDUCERS`
- `reducers._REGISTRY`
- `reducers.SPECIAL_REDUCERS`
- `adapters/earth_engine/_reducers._EE_REDUCER_MAP`
- categorical compatibility sets used by the webapp
- special `point` and `class_fraction` branches

**Impact**

Adding or changing a reducer requires coordinated edits across generic, local, Earth Engine, and UI layers. A future provider may support only a subset or give a reducer different numerical semantics.

**Recommendation**

Create a canonical reducer specification describing name, output shape, applicable data types, point/window semantics, and backend translators. Validate provider support while constructing the extraction plan. Consider modelling `point` as a sampling operation rather than pretending it is a normal window reducer.

### F14 — Medium: the webapp duplicates and imports private core policy

**Evidence**

- `src/envoi_webapp/helpers.py:17-21` imports a private input validator and core reducer constants.
- Web helpers duplicate coordinate, dataset, reducer, and run-config validation.
- `run_extraction()` always requires and initializes Earth Engine credentials, even if selected datasets could be local or handled by a future provider.
- UI copy calls every selection an Earth Engine product (`src/envoi_webapp/app.py:1243-1248`).

**Impact**

Core and UI validation can drift. Adding providers requires editing provider-neutral UI code and may unnecessarily request sensitive credentials.

**Recommendation**

Expose public `validate_request()` and `build_plan()` functions. Let the UI query provider capabilities and authentication requirements from the plan, then display only the fields needed by the selected providers.

### F15 — Medium: core output is filesystem-first and weakly typed

**Evidence**

- `extract()` takes one `output_dir` and writes metadata/QC and most primary outputs as side effects.
- Return values can be a bare DataFrame or a dictionary mixing `Path` and `DataFrame` values.
- QC and metadata outputs are not represented directly in the return value.

**Impact**

The service layer must understand core naming conventions and scan server paths to assemble downloads. Alternate storage backends and programmatic consumers are harder to support.

**Recommendation**

Return a stable `ExtractionResult` containing primary tables, QC artifacts, metadata, tiles, warnings, and statuses. Delegate persistence to an `ArtifactStore` implementation:

- `LocalArtifactStore` for the current Python API.
- `TemporaryArtifactStore` for small hosted jobs.
- `ObjectArtifactStore` for scalable deployment.

Preserve current return values through a compatibility adapter at the public façade.

### F16 — Medium: configuration accepts some silent typos and lacks schema versioning

**Evidence**

- `_parse_run_config()` validates recognized values but does not reject unknown keys at the run or settings level.
- Dataset override and typed-statistics dictionaries do reject unknown keys, producing inconsistent strictness.
- Neither run configuration nor catalog has an explicit schema version.

**Impact**

Misspelled options can be silently ignored. Future changes cannot distinguish legacy configuration from new semantics reliably.

**Recommendation**

Adopt strict, versioned request and catalog models. Provide explicit migrations and deprecation warnings rather than silently accepting ambiguous structures.

### F17 — Medium: production deployment and service tests are absent

**Evidence**

- `src/envoi_webapp/app.py:1326-1345` hard-codes `--server.address localhost` in the packaged launcher.
- `.streamlit/config.toml` contains only theme configuration.
- CI installs `.[dev]`, not `.[webapp]`, before tests.
- There is no deployment manifest/container definition, hosted-mode configuration, health check, or browser-level extraction smoke test.
- `streamlit>=1.36`, Ruff, Black, and Pytest are unbounded above in `pyproject.toml`.

**Impact**

The repository does not currently prove that a clean deployment can import and start the actual application. Unbounded tooling and UI dependencies can make CI or production change without a source modification.

**Recommendation**

- Define one deployment target and entry point.
- Make address, port, hosted mode, storage, and authentication configurable.
- Install the webapp dependencies in at least one CI job.
- Add a Streamlit `AppTest` or equivalent startup/form smoke test.
- Add isolation, path traversal, upload-limit, cleanup, and concurrent-authentication tests.
- Pin deployment dependencies through a lock file or constrained compatible ranges.

Streamlit Community Cloud can provide a public URL, but its resources are shared and limited, making it better suited to a controlled MVP than large raster workloads: <https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app>.

## Target architecture

```text
Python caller                         Streamlit UI
     |                                    |
     +---------------+--------------------+
                     |
                     v
             extract() public facade
                     |
             validate_request()
                     |
             immutable ExtractionPlan
                     |
             ExtractionEngine.execute
                     |
          +----------+------------+
          |                       |
  ExtractionContext        ProviderRegistry
  - credentials            - definitions
  - artifact store         - schemas
  - cancellation           - capabilities
  - limits                 - lazy factories
  - telemetry              - reducer support
          |                       |
          +----------+------------+
                     |
       +-------------+--------------------+
       |             |          |         |
 Earth Engine      local      CHELSA   Copernicus/other
       |             |          |         |
       +-------------+----------+---------+
                     |
                typed results
                     |
                artifact store
                     |
       local paths / download / object URLs
```

### Stable public façade

Keep `extract()` as the main user experience. Existing calls should continue to work. Internally, it should become a compatibility façade over explicit planning and execution:

```python
def extract(df, config, output_dir="outputs", *, context=None, ...):
    plan = build_plan(df, config, ...)
    result = ExtractionEngine(...).execute(plan, context or local_context(output_dir))
    return legacy_return_shape(result)
```

This preserves familiarity while making the engine usable by the hosted service without pretending the server filesystem is the user's filesystem.

## Recommended delivery sequence

### Phase 0 — decisions and safety boundaries

1. Decide whether the hosted service uses an application Earth Engine identity, user OAuth, or isolated BYO credentials.
2. Define initial public workload limits and supported output sizes.
3. Decide whether Streamlit Community Cloud is a demonstration target or whether jobs require a dedicated worker deployment.

**Exit criterion:** no ambiguity about which identity pays provider quotas or how concurrent credential contexts are isolated.

### Phase 1 — formalize the core without breaking callers

1. Define `ProviderDefinition`, capability declarations, and an adapter `Protocol`/ABC.
2. Add `ExtractionPlan`, `ExtractionContext`, typed results, structured failures, and an artifact-store interface.
3. Move provider-specific validation out of `_config_parsing.py`.
4. Consolidate reducer definitions.
5. Create immutable per-run catalog snapshots and introduce schema versions.
6. Retain `extract()` and current output shapes through compatibility wrappers.

**Exit criterion:** a dummy third adapter passes conformance tests through the public `extract()` API without changing generic orchestration.

### Phase 2 — hosted-safe Streamlit MVP

1. Make Streamlit a thin caller of a service/application layer.
2. Remove the hosted output-directory picker.
3. Use unique job workspaces and downloadable packages.
4. Add input complexity limits, a global concurrency semaphore, deadlines, and cleanup.
5. Deploy with an application-owned Earth Engine identity or completed OAuth isolation.
6. Add hosted startup and security regression tests.

**Exit criterion:** two simultaneous users cannot access each other's credentials, catalog changes, working files, outputs, or execution budgets.

### Phase 3 — reference provider integration

Implement one new source against the new contract. A relatively simple CHELSA access path may be useful as the first proof; a provider requiring authentication, asynchronous delivery, or request packaging should follow to test the richer capabilities.

**Exit criterion:** adding the provider requires a provider package/catalog entry and registration, but no source-specific branch in `extract()` or generic config parsing.

### Phase 4 — scalable service execution

1. Move long jobs to process-isolated background workers.
2. Persist job states and make execution idempotent.
3. Add cancellation, retries with provider-specific policy, and structured observability.
4. Store large artifacts outside the web process and return signed URLs.
5. Add per-user accounting, quotas, and operational dashboards.

**Exit criterion:** browser reruns, disconnects, or web-process restarts do not corrupt or orphan active jobs.

## Recommended first implementation slice

Before implementing a full plugin system, the smallest valuable slice is:

1. Add an `Adapter` protocol with capability flags and `export_tiles()`.
2. Add a typed `PointResult` with explicit status/error fields.
3. Add an `ExtractionContext` carrying credentials, artifact storage, and concurrency budget.
4. Move adapter construction outside the per-window loop.
5. Add a fake stats-only provider and contract tests.
6. Keep all existing `extract()` calls and outputs unchanged.

This slice tests the central design without forcing an immediate rewrite of every catalog entry or output writer.

## Existing strengths to retain

- `extract()` is a clear user-facing concept and should remain the façade.
- Catalog-driven source selection keeps provider identifiers out of ordinary user code.
- Local and Earth Engine source logic are already separated.
- Batch adapter methods permit source-specific optimization.
- Shared validation, output assembly, QC, naming, progress, and metadata reduce duplication.
- Context-manager cleanup is already part of adapter usage.
- Credential temporary files use unique names, restrictive permissions, and `finally` cleanup in the local MVP.
- Credential redaction is present for displayed exceptions, although it is not a substitute for production credential isolation.

## Verification performed

- `conda run -n envoi_dev pytest -q -m "not gee"`: **314 passed, 46 deselected**.
- `black --check src tests`: **passed**; 38 files unchanged.
- `conda run -n envoi_dev ruff check src tests`: **failed with one existing `E402`** at the eager adapter imports in `src/envoi/adapters/__init__.py`.
- Live Earth Engine behaviour was not executed because it requires credentials and network access.
- No implementation files were changed as part of the review.

## Summary recommendation

Do not add several provider modules directly to the current registry and postpone the boundary work. Formalize the provider contract first, using one new provider as its proof. In parallel, treat public Streamlit publication as a service architecture project—especially authentication, per-job storage, and workload isolation—rather than only a deployment configuration change.
