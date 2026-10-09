# Advanced usage

The [README's quick-start](../README.md#quick-start) covers the common case — one DataFrame, one dataset, one output. The patterns below build on that for runs that involve more than one output, time-varying datasets, mixed continuous and categorical inputs, per-call band selection, multiple window sizes, and your own dataset catalog. Everything documented here is configured through the second argument to `extract()` (a Python dict, a list of dicts, or a path to a YAML file) — nothing requires changes to envoi itself.

## Table of contents

- [Multiple outputs in one call](#multiple-outputs-in-one-call)
- [Date-aware extraction](#date-aware-extraction)
- [Mixing categorical and continuous datasets](#mixing-categorical-and-continuous-datasets)
- [Multiple window sizes in one call](#multiple-window-sizes-in-one-call)
- [Selecting bands per call](#selecting-bands-per-call)
- [Circular aspect statistics](#circular-aspect-statistics)
- [Custom datasets](#custom-datasets)
- [Failed points](#failed-points)
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

Each entry produces its own output file (or its own folder of raster tiles) and its own JSON metadata sidecar, keyed by `batch_id`. So every entry needs a different `batch_id`. An entry without `batch_id` gets the name `output<N>`, where N is its position in the list (starting at 1). `extract()` checks every entry before it starts the first one, and stops with an error if two entries have the same `batch_id` or if any entry has an invalid setting. The input DataFrame, the CRS validation step, the date-parsing pass, and (for Earth Engine) the authenticated session are shared across all entries — running a batched list is cheaper than calling `extract()` once per output and avoids per-call startup overhead.

The `outputs` dict returned by `extract()` keys results by `batch_id` for tabular runs (`outputs["terrain_stats"]`) and by `<batch_id>:<dataset>` for raster runs (`outputs["terrain_tiles:dem_copernicus_glo30"]`), so downstream code can pick up either by name without scanning the filesystem.

## Date-aware extraction

Time-varying Earth Engine ImageCollections (climate reanalyses, satellite vegetation indices, monthly aggregates) only make sense when each point is paired with a date. envoi reads this from the input DataFrame's `eventDate` column whenever it exists, and silently skips the date branch when it doesn't — there's no need to declare upfront whether your run is time-aware. If your dates are in a column with another name, pass that name with `date_column=`. envoi does not read a column named `date` unless you pass `date_column="date"`.

```python
sample_points = pd.DataFrame({
    "occurrenceID":     ["a", "b"],
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

**Date-format flexibility.** envoi follows the GBIF / Darwin Core convention and accepts the ISO 8601 variants commonly seen in occurrence records: plain `YYYY-MM-DD`, `YYYY-MM`, year-only `YYYY`, intervals like `2022-06-15/2022-06-20`, and timestamps with time-of-day or timezone. It also accepts `YYYY/MM/DD` (for example `2022/06/15`), which is the same date as `YYYY-MM-DD`. All forms are collapsed to a single `YYYY-MM-DD` per row before extraction: an interval (a value with exactly one `/` whose start is an ISO 8601 date or date-time) becomes its start day. These cases do not stop the run. envoi gives a warning, which is also recorded in the metadata sidecar:

- A row with a missing date is skipped.
- A year-only or year-month date is read as the first day of that year or month (`2022` becomes `2022-01-01`).
- An interval is cut to its start day, and a time of day is removed.

These date values stop the run with a `ValueError` before any extraction:

- A value with `/` in another form, such as `15/06/2022`, `06/15/2022`, `06/2022`, or `2022/6/15`. envoi does not guess the order of day and month. The message names your date column and lists the values. Write these dates as `YYYY-MM-DD`.
- A value that envoi cannot read as a date, such as `spring 2022` or `2022-13-45`.

**Per-dataset selection policy.** For each point, envoi selects one image from the collection. The choice is controlled per-dataset by `collection_date_policy` in the catalog:

- `"nearest"` (default) — the image whose start timestamp is closest to the point's date. Right for snapshot collections and most yearly composites.
- `"contains"` — the image whose `[start, end)` time range contains the point's date. Use this for interval-based collections (monthly aggregates, seasonal composites) where "the image valid for July 2022" is more meaningful than "the image whose start is closest to July 15".

**Out-of-range dates.** When a point's date falls before the earliest image in the collection or after the latest, the date is clamped to the nearest boundary and the substitution is recorded in the metadata sidecar — extraction proceeds rather than raising. This keeps a single rogue date from killing a long batch; the sidecar makes it easy to spot and decide whether to drop those points downstream. With `"contains"`, the collection's range ends at the end of the last image's `[start, end)` time range, not at its start. So a date inside the last time range is not clamped. A date is clamped only when it is before the first image's start, or on or after the end of the last time range. When the collection has no end times, the last time range ends one day after the last image's start. The last time range always includes the whole start day of the last image, as the image selection does, so a date on that day is not clamped. With `"nearest"`, a date before the first image's start or after the last image's start is clamped.

**No date column.** Without an `eventDate` column, envoi uses the most recent image of each collection. envoi does not combine images for this. For each point, it takes the latest start time of the images that intersect the points of the batch, and selects the point's image that starts on that day (with `"contains"`, the point's image of the last time range). A pixel that this image masks stays masked, also when an older image has a value there. When the point has no image that starts on that day (possible in collections whose tiles have different dates), the point is a failed point. The QC column `<dataset>_date_source_<N>m` shows `most_recent_no_date`. Two kinds of collection differ: with `collection_date_policy: mosaic`, envoi ignores dates and mosaics the tiles at each point; and a collection without an image at any point of the batch is used as a mosaic of the whole collection (`ee.ImageCollection.mosaic()`), which gives these points no data. envoi prints a notice that there is no date column, and the sidecar records it in `warnings`. This is the right behaviour for static datasets or when "most recent" is genuinely what you need, but it does mean a quiet date-less run on a time-varying dataset is hiding an assumption — the sidecar is where to verify it.

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

**Categorical reducers expand per class.** When you request `class_count` or `class_fraction` on a categorical raster, envoi appends one column per class actually observed somewhere in the batch — e.g. `lulc_worldcover_2021_class_10_fraction_200m`, `..._class_20_fraction_200m`, and so on. A row whose window didn't see a given class is filled with `0` (count) or `0.0` (fraction); a class that never appears anywhere in the input batch produces no column at all, which keeps the output narrow on focused study areas. Pixel-validity is tracked separately in the QC sidecar, so the zero-fill never blurs "this class was absent" with "this window had no valid pixels". A [failed point](#failed-points) is not zero-filled: its class columns are empty, because its classes are unknown.

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

## Circular aspect statistics

The derived band `aspect` is the direction that a slope faces, in degrees clockwise from north: 0° is north, 90° east, 180° south, and 270° west. A direction is a circular value. For example, 350° and 10° are only 20° apart, but their ordinary (linear) mean is 180°, which points south. For this reason, envoi computes three window statistics of `aspect` as circular statistics:

- **`mean`** is the circular mean direction, in degrees from 0 to less than 360. envoi converts each pixel's direction to a unit vector (the sine and cosine of the angle), takes the mean of these vectors, and gives the direction of the mean vector.
- **`std`** is the circular standard deviation, sqrt(−2 ln R), in degrees. R is the mean resultant length: the length of the mean vector. R is 1 when all pixels face the same direction (then `std` is 0), and goes towards 0 when the directions cancel out.
- **`var`** is the square of `std`, in square degrees.

**Flat pixels.** A pixel with slope 0 faces no direction, but Earth Engine gives it the aspect 0 (north). envoi does not use pixels with slope 0 for these three statistics. Otherwise, flat areas would pull the mean towards north.

**Empty values.** The three statistics are empty when the window has no pixel with a slope above 0 (for example, a lake), or when R is below 10⁻¹² (the directions are spread evenly around the circle, so there is no mean direction). The QC columns do not change: flat pixels count as valid pixels there.

**Other statistics of aspect.** `median`, `min`, `max`, the quantiles (`q05` to `q95`), `sum`, `mode`, and `point` treat aspect as a linear value from 0 to 360, and include flat pixels (as 0). Interpret them with care for slopes that face north: a window with directions on both sides of north can have a `min` near 0 and a `max` near 360. `count` counts every pixel with an aspect value, also the flat pixels.

The column names are the usual ones (for example `dem_copernicus_glo30_aspect_mean_200m`). envoi computes the circular values in the same Earth Engine request as the other statistics of the point.

Only the Earth Engine derived band `aspect` gets circular statistics. Local rasters have no derived bands: if your GeoTIFF has an aspect band, envoi treats it like any other band, and all its statistics are linear. The same is true for a band of an Earth Engine asset that has the name `aspect` but is not the derived band.

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

**Auto-detection for local rasters.** For `data_source: local` entries, you only need to provide `data_source` and `path`. When an extraction opens the file, envoi reads the CRS, native pixel size, nodata values, and band count from it with rasterio. When you register a YAML file, envoi also reads each local raster at registration time and adds the missing `crs`, `resolution_m` (in metres also for rasters whose CRS is in degrees or feet), `nodata`, and `band_count` to its catalog entry, which `list_datasets("full")` shows. Entries that you register as a dict are stored without this step. Set `data_type: categorical` explicitly if the raster is a class map; local entries default to `continuous` otherwise.

**Resetting the catalog.** Call `reset_catalog()` to drop everything registered at runtime and revert to the built-in catalog alone. Mostly useful in tests or notebooks where you want a clean slate between cells.

For the full catalog schema, see the commented reference block at the top of [src/envoi/configs/ee_catalog.yml](../src/envoi/configs/ee_catalog.yml) — it's a copy-paste starting point for adding a new entry. A runnable starter that registers both a local raster and an Earth Engine asset lives at [examples/catalog.yml](../examples/catalog.yml).

## Failed points

A *failed point* is a point whose data request or read raised an error, for example a temporary Earth Engine error, a network timeout, or a damaged block in a local GeoTIFF. A point outside the dataset's extent, or a point whose window has only nodata pixels, is not a failed point: it has no error, and the QC columns show it as before. This also applies to an Earth Engine collection that has no image at the point on any date (for example an ocean point of a dataset that covers only land). Such a point gets the values of a point without data: its statistics are empty, except `count`, `sum`, and the per-class columns of `class_count` and `class_fraction`, which are `0`; `in_extent` is `False`; and in raster runs its tile has only masked pixels. A point that has images of the collection, but none that envoi selected (none on the selected dates, or none in the point's UTM zone with `use_utm_zone`), is a failed point, so that a problem with the image selection is not reported as missing data. Its error says that a parameter "is required and may not be null" (for example `Image.reduceRegion: Parameter 'image' is required and may not be null.`): envoi found no image at the point on the selected dates. Check the dataset's `collection_date_policy` (and `use_utm_zone`) in the catalog, and the point's sample date against the dates of the collection's images.

A failed point does not stop the run. envoi continues with the other points and reports the failed point:

- **Statistics.** Every statistic of a failed point is empty, also the `class_count` and `class_fraction` columns, which are otherwise `0` for an absent class. A failed point adds no column of its own. With the local adapter, when a point needs both a window read and a pixel read (window statistics together with `point`), an error in either read makes the point a failed point.
- **QC table.** The column `<dataset>_error_<N>m` (`<dataset>_error_point` for point-only statistics) holds the error of each failed point, as one line of text in the form `<error type>: <message>`, at most 500 characters. The other points have an empty value. The column exists only when at least one point of that dataset and window failed. The QC flags of a failed point are those of a point outside the extent: `in_extent` is `False`, `n_pixels` is `0`, `coverage_pct` is `0`, and `had_nodata` is `False`.
- **Tile manifest (raster runs).** A failed tile has `exported` set to `False`, and the manifest gets an `error` column with the same text. The column exists only when at least one tile failed. A point without data has no error, and the two kinds of dataset handle it differently. For an Earth Engine dataset, envoi writes a tile in which every pixel is masked: `exported` is `True`, and `quality.tiles.n_exported` counts the tile. For a local dataset, envoi writes no tile for a point outside the raster: `exported` is `False`, and `quality.tiles.n_failed` counts the point.
- **Metadata sidecar.** Each coverage summary (`quality.<N>m` or `quality.point`) and each tile summary (`quality.tiles`) has `n_errors` (the number of failed points) and `ids_errors` (their IDs). Both are always present, with `0` and `[]` when nothing failed. In a coverage summary, a failed point is also listed in `ids_no_data`, because it has no data. In `quality.tiles`, `n_failed` counts every tile that was not exported, for any reason; `n_errors` counts only the failed tiles.
- **Warning.** envoi gives one warning for each dataset and window with failed points. It names the dataset, the window, the number of failed points, the IDs of up to 10 of them, where the errors are, and the error of the first failed point. With `write_metadata=False`, the QC table is not written, and the warning says so; the tile manifest of a raster run is always written. The sidecar records the warning in `warnings`, under the key `failed_<dataset>_<N>m` (`failed_<dataset>_point` for point-only statistics).

The error text can contain your Earth Engine Cloud project ID or a temporary download URL. Check it before you share the QC table, the tile manifest, or the metadata sidecar.

When **every** point of one dataset and window fails, the cause usually affects all points (for example no network, or no access to the dataset), so envoi stops the run with a `RuntimeError`. The message names the dataset and the window and shows the error of the first point. Outputs that were written before the error (for example the tiles of an earlier dataset, or the files of an earlier run config in a list) stay on disk. Correct the problem and run the extraction again.

To repeat only the failed points, select the rows whose ID is in `ids_errors` (or whose error column is not empty) and run `extract()` again for them.

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
