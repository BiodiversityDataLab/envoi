# Advanced usage

The [README's quick-start](../README.md#quick-start) covers the common case — one DataFrame, one dataset, one output. The patterns below build on that for runs that involve more than one output, time-varying datasets, mixed continuous and categorical inputs, per-call band selection, multiple window sizes, and your own dataset catalog. Everything documented here is configured through the second argument to `extract()` (a Python dict, a list of dicts, or a path to a YAML file) — nothing requires changes to envoi itself.

## Table of contents

- [Multiple outputs in one call](#multiple-outputs-in-one-call)
- [Date-aware extraction](#date-aware-extraction)
- [Mixing categorical and continuous datasets](#mixing-categorical-and-continuous-datasets)
- [Multiple window sizes in one call](#multiple-window-sizes-in-one-call)
- [Selecting bands per call](#selecting-bands-per-call)
- [Custom datasets](#custom-datasets)
- [Running interactively](#running-interactively)

## Multiple outputs in one call

`extract()` accepts a list of run configs and produces one output per entry in a single call. This is the right pattern when you want:

- both summary statistics and GeoTIFF tiles for the same set of points,
- different datasets pulled into separate files so downstream analysts can mix and match them,
- the same dataset summarised under different settings (e.g. different reducers per output).

```python
outputs = extract(sample_points, [
    {
        "batch_id": "terrain_stats",
        "datasets": ["dem_copernicus_glo30"],
        "settings": {
            "output_type": "tabular",
            "statistics": ["mean", "std"],
            "window_size_m": 200,
        },
    },
    {
        "batch_id": "terrain_tiles",
        "datasets": ["dem_copernicus_glo30"],
        "settings": {
            "output_type": "raster",
            "window_size_m": 200,
            "resample_m": 10,
        },
    },
])
```

Each entry produces its own output file (or its own folder of raster tiles) and its own JSON metadata sidecar, keyed by `batch_id`. The input DataFrame, the CRS validation step, the date-parsing pass, and (for Earth Engine) the authenticated session are shared across all entries — running a batched list is cheaper than calling `extract()` once per output and avoids per-call startup overhead.

The `outputs` dict returned by `extract()` keys results by `batch_id` for tabular runs (`outputs["terrain_stats"]`) and by `<batch_id>:<dataset>` for raster runs (`outputs["terrain_tiles:dem_copernicus_glo30"]`), so downstream code can pick up either by name without scanning the filesystem.

## Date-aware extraction

Time-varying Earth Engine ImageCollections (climate reanalyses, satellite vegetation indices, monthly aggregates) only make sense when each point is paired with a date. envoi reads this from the input DataFrame's `eventDate` column whenever it exists, and silently skips the date branch when it doesn't — there's no need to declare upfront whether your run is time-aware.

```python
sample_points = pd.DataFrame({
    "gbifID":     ["a", "b"],
    "decimalLatitude":  [59.85, 59.86],
    "decimalLongitude": [17.63, 17.64],
    "eventDate":        ["2022-06-15", "2023-08-01"],
})

extract(sample_points, {
    "batch_id": "ndvi",
    "datasets": ["ndvi_landsat_annual"],
    "settings": {"output_type": "tabular", "statistics": ["mean"], "window_size_m": 200},
})
```

**Date-format flexibility.** envoi follows the GBIF / Darwin Core convention and accepts the ISO 8601 variants commonly seen in occurrence records: plain `YYYY-MM-DD`, `YYYY-MM`, year-only `YYYY`, intervals like `2022-06-15/2022-06-20`, and timestamps with time-of-day or timezone. All forms are collapsed to a single `YYYY-MM-DD` per row before extraction. Incomplete or unparseable dates are recorded in the metadata sidecar so they can be audited after the run rather than killing it.

**Per-dataset selection policy.** For each point, envoi selects one image from the collection. The choice is controlled per-dataset by `collection_date_policy` in the catalog:

- `"nearest"` (default) — the image whose start timestamp is closest to the point's date. Right for snapshot collections and most yearly composites.
- `"contains"` — the image whose `[start, end)` time range contains the point's date. Use this for interval-based collections (monthly aggregates, seasonal composites) where "the image valid for July 2022" is more meaningful than "the image whose start is closest to July 15".

**Out-of-range dates.** When a point's date falls before the earliest image in the collection or after the latest, the date is clamped to the nearest boundary and the substitution is recorded in the metadata sidecar — extraction proceeds rather than raising. This keeps a single rogue date from killing a long batch; the sidecar makes it easy to spot and decide whether to drop those points downstream.

**No date column.** Without an `eventDate` column, envoi falls back to the most recent non-masked image in each collection (`ee.ImageCollection.mosaic()` semantics) and emits a warning that is also captured in the sidecar. This is the right behaviour for static datasets or when "most recent" is genuinely what you need, but it does mean a quiet date-less run on a time-varying dataset is hiding an assumption — the sidecar is where to verify it.

## Mixing categorical and continuous datasets

Realistic ecological feature sets often combine continuous variables (elevation, climate) with categorical ones (land cover, biome class). The two call for different families of reducers — `mean` is meaningless on a land-cover code, and `mode` is rarely what you want on a precipitation grid. To let a single run handle both, pass a typed dict for `statistics` keyed by `continuous` / `categorical`:

```python
extract(sample_points, {
    "batch_id": "mixed",
    "datasets": ["dem_copernicus_glo30", "lulc_worldcover_2021"],
    "settings": {
        "output_type": "tabular",
        "statistics": {
            "continuous":  ["mean", "std", "q10", "q90"],
            "categorical": ["mode", "class_fraction"],
        },
        "window_size_m": 200,
    },
})
```

Each dataset's `data_type` in the catalog (`continuous` or `categorical`) selects which list applies. The same reducer can appear in both lists if it makes sense for both — `mode`, for instance, is sensible on either continuous or categorical data. Datasets without an explicit `data_type` (common for ad-hoc local rasters) default to `continuous`.

A flat list (`"statistics": ["mean", "std"]`) is still accepted and applies to every dataset regardless of type — preferable when the run is homogeneous, since the typed-dict form is only useful when reducers differ by data type.

**Categorical reducers expand per class.** When you request `class_count` or `class_fraction` on a categorical raster, envoi appends one column per class actually observed somewhere in the batch — e.g. `lulc_worldcover_2021_class_10_fraction_200m`, `..._class_20_fraction_200m`, and so on. A row whose window didn't see a given class is filled with `0` (count) or `0.0` (fraction); a class that never appears anywhere in the input batch produces no column at all, which keeps the output narrow on focused study areas. Pixel-validity is tracked separately in the QC sidecar, so the zero-fill never blurs "this class was absent" with "this window had no valid pixels".

## Multiple window sizes in one call

`window_size_m` accepts a list as well as a single integer. Each window size produces its own set of reducer columns (tabular) or its own folder of tiles (raster), all in the same output file or batch folder:

```python
extract(sample_points, {
    "batch_id": "terrain_multiscale",
    "datasets": ["dem_copernicus_glo30"],
    "settings": {
        "output_type": "tabular",
        "statistics": ["mean", "std"],
        "window_size_m": [100, 500, 1000],
    },
})
```

The window size is appended to every reducer column (`dem_copernicus_glo30_mean_100m`, `..._mean_500m`, `..._mean_1000m`), so a multi-scale run is the natural way to compare how a variable behaves at different spatial extents — useful when training a model whose response is scale-sensitive, or when probing a habitat across multiple radii. All sizes share the same per-point round-trip planning in the GEE adapter, so this is materially cheaper than calling `extract()` three times.

## Selecting bands per call

The catalog defines each dataset's default bands. To override them for a single call without re-registering the catalog, replace a string entry in `datasets` with a single-key dict whose value is the band list:

```python
extract(sample_points, {
    "batch_id": "satellite",
    "datasets": [
        "dem_copernicus_glo30",                            # catalog defaults
        {"sr_landsat_annual": ["B4", "B5"]},               # narrow bands for this run only
        {"dem_copernicus_glo30": ["DEM", "slope"]},        # mix source + derived band names
    ],
    "settings": {"output_type": "tabular", "statistics": ["mean"], "window_size_m": 200},
})
```

The override **replaces** the catalog's `bands` list for that dataset in that run (no merging) — if you want to keep a band the catalog already defines, list it again explicitly.

**Derived bands.** Names recognised as derived (currently `slope` and `aspect`, both computed on the fly from the dataset's first band) are split out automatically — you don't need a separate key. Derived bands only make physical sense for some datasets (DEMs, primarily), so each catalog entry that supports them declares a `supported_derived_bands` whitelist. A per-call request that isn't in that list is rejected up front with a clear error, rather than silently producing a meaningless layer. Derived bands are also Earth-Engine-only — requesting `slope` on a local raster raises `ValueError` immediately, since the local adapter doesn't run the gradient computation.

**Column-naming consequences.** A one-element band override (`{"sr_landsat_annual": ["B4"]}`) keeps the multi-band column-naming style (`sr_landsat_annual_B4_mean_200m`). Use the catalog default if you instead want the bare single-band form (`sr_landsat_annual_mean_200m`) — the distinction reflects "the user picked one band out of many" versus "the dataset has only one band", which matters when downstream code keys off column names.

> **Band identifiers differ by data source.** Earth Engine identifies bands by *string* names (`"B4"`, `"DEM"`, `"slope"`) because GEE exposes named bands. Local GeoTIFFs identify bands by *integer* index (`1`, `2`, `[1, 2, 3]`) because rasterio reads bands positionally. A GEE catalog entry might set `bands: ["DEM"]` and a local one `bands: 1`, and per-call overrides follow the same convention as the dataset they target.

## Custom datasets

Datasets that aren't in the built-in catalog can be registered once via `update_catalog()`. Registered datasets are then available in every subsequent `extract()` call within the same Python session — there is no need to repeat the registration per call.

```python
from envoi import update_catalog

# From a dict — quick for one-off local rasters.
update_catalog({
    "datasets": {
        "my_dem": {"data_source": "local", "path": "data/dem.tif"},
    },
})

# From a YAML file — better for multi-dataset projects under version control.
update_catalog("my_catalog.yml")
```

Earth Engine assets that aren't in the built-in catalog are registered the same way — just set `data_source: earth_engine` and point `path` at the GEE asset ID. Only `data_source`, `path`, and `data_type` are required (bands are optional — all are loaded if you omit them), and the asset type (`Image` vs `ImageCollection`) is auto-detected, so there's nothing else to declare for a static image:

```python
# An Earth Engine asset not in the built-in catalog — OpenLandMap soil pH (H2O).
update_catalog({
    "datasets": {
        "soil_ph_openlandmap": {
            "data_source": "earth_engine",
            "path": "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02",
            "data_type": "continuous",
            "bands": ["b0", "b10"],   # soil pH at 0 cm and 10 cm depth
        },
    },
})
```

**Precedence.** User-registered datasets override built-ins with the same name. This is intentional: you can swap an upstream catalog entry for a higher-resolution local copy without renaming every downstream config that references it. Re-registering the same name later overwrites the earlier definition.

**Auto-detection for local rasters.** For `data_source: local` entries, envoi reads the file with rasterio at registration time and fills in CRS, native pixel size, nodata values, and band count automatically — you only need to provide `data_source` and `path`. Set `data_type: categorical` explicitly if the raster is a class map; local entries default to `continuous` otherwise.

**Resetting the catalog.** Call `reset_catalog()` to drop everything registered at runtime and revert to the built-in catalog alone. Mostly useful in tests or notebooks where you want a clean slate between cells.

For the full catalog schema, see the commented reference block at the top of [src/envoi/configs/ee_catalog.yml](../src/envoi/configs/ee_catalog.yml) — it's a copy-paste starting point for adding a new entry. A runnable starter that registers both a local raster and an Earth Engine asset lives at [examples/catalog.yml](../examples/catalog.yml).

## Running interactively

A few `extract()` knobs are aimed at notebook and exploratory use, where writing files to disk on every iteration is overkill:

- `output_file_format: "dataframe"` (tabular only) — returns the stats table as an in-memory DataFrame instead of writing CSV or Parquet. When you pass a single run config (a dict, not a list), `extract()` hands the DataFrame back directly; with a list of configs it stays keyed by `batch_id` like any other run. Useful when you want to immediately filter, plot, or join the result without round-tripping through a file.
- `write_metadata=False` (keyword argument on `extract()`) — suppresses both the JSON metadata sidecar and the per-point QC table. The main output is still written (unless you also asked for `"dataframe"` format), but the auxiliary files don't accumulate while you iterate.

```python
stats = extract(
    sample_points,
    {
        "batch_id": "scratch",
        "datasets": ["dem_copernicus_glo30"],
        "settings": {
            "output_type": "tabular",
            "statistics": ["mean"],
            "window_size_m": 200,
            "output_file_format": "dataframe",
        },
    },
    write_metadata=False,
)

stats.head()  # a single config in "dataframe" mode returns the table directly
```

For production runs, leave both at their defaults — the JSON sidecar is the only record of which images and dates were actually used per point, and the QC table is what you filter on to drop low-coverage rows. Both are essential for reproducibility once a run leaves the notebook.
