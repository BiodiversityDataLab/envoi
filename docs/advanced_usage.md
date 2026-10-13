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
- [Known limitations](#known-limitations)

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

**Accepted date values.** envoi follows the GBIF / Darwin Core convention and reads only these date values, so that it never has to guess the order of day and month:

- Text dates `YYYY`, `YYYY-MM`, and `YYYY-MM-DD`. The month and the day can have one digit (`2022-6-5` is June 5, 2022).
- Text dates `YYYYMMDD` (`20220615`) and `YYYY/MM/DD` (`2022/06/15`). Both are the same date as `YYYY-MM-DD`.
- Each of these forms except `YYYY/MM/DD` can also have a time of day after `T` or one space, and a time zone, such as `2022-06-15T08:00`, `2022-06-15T08:00:00Z`, `2022-06-15T08:00+02:00`, or `2022-06-15 08:00`. After a space, the time needs hours and minutes (`08:00`).
- ISO 8601 intervals: a value with exactly one `/` whose start is `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`, optionally with a time of day, such as `2022-06-15/2022-06-20`. In an interval start, the month and the day need two digits: `2022-6-15/2022-6-20` stops the run.
- Date and datetime values, for example a column that pandas read as dates (`datetime64`), or `datetime.date` values.
- Whole numbers with four digits as years, such as `2022` or `2022.0`, and whole numbers with eight digits as `YYYYMMDD` dates, such as `20220615` or `20220615.0`. A CSV reader stores such columns as numbers (as decimal numbers when a value is missing).

envoi reduces each value to one day (`YYYY-MM-DD`) before extraction. The day is the day as written: envoi drops the time of day and does not convert the time zone, also for datetime values with a time zone. These cases do not stop the run. envoi gives a warning, which is also recorded in the metadata sidecar:

- A row with a missing date is skipped.
- A year or a year-month is read as the first day of that year or month (`2022` becomes `2022-01-01`).
- An interval is cut to its start day, and a time of day in a text date is removed.

All other date values stop the run with a `ValueError` before any extraction. The message names your date column and lists up to 10 of the values, with their total number:

- A date in another form, such as `15/06/2022`, `06/15/2022`, `05-06-2022`, `05.06.2022`, `June 5, 2022`, `06/2022`, or `2022/6/15`. envoi does not guess the order of day and month. Write these dates as `YYYY-MM-DD`.
- A value that is not a date, such as `spring 2022`, or a number that is not a whole number with four or eight digits, such as `2022.5` or `202206`.
- A date that does not exist, such as `2022-02-30`, `2022-13-01`, or the number `15062022` (read as `YYYYMMDD`, its month would be 20).

**Per-dataset selection policy.** For each point, envoi selects one image from the collection. The choice is controlled per-dataset by `collection_date_policy` in the catalog:

- `"nearest"` (default) — the image with the fewest whole days (UTC) between its start day and the point's date; of two equally near images, the earlier one. An image that starts later on the point's date counts as on that day. Right for snapshot collections and most yearly composites.
- `"contains"` — the image whose `[start, end)` time range contains the point's date. Use this for interval-based collections (monthly aggregates, seasonal composites) where "the image valid for July 2022" is more meaningful than "the image whose start is closest to July 15". envoi compares whole days in UTC: an image contains the point's date when it starts on or before that day and ends after the start of that day. So an image that starts at 10:00 on the point's date contains that date. An image without an end time, or with an end on its start day, ends at the end of its start day.

**Out-of-range dates.** When a point's date falls before the earliest image in the collection or after the latest, the date is clamped to the nearest boundary and the substitution is recorded in the metadata sidecar — extraction proceeds rather than raising. This keeps a single rogue date from killing a long batch; the sidecar makes it easy to spot and decide whether to drop those points downstream. With `"contains"`, the collection's range ends at the end of the last image's `[start, end)` time range, not at its start. So a date inside the last time range is not clamped. A date is clamped only when it is before the first image's start day, or on or after the end of the last time range. When the collection has no end times, the last time range ends at the end of the last image's start day. The last time range always includes the whole start day of the last image, as the image selection does, so a date on that day is not clamped. With `"nearest"`, a date before the first image's start day or after the last image's start day is clamped; a date on either of these days is not clamped, whatever the time of day of the image.

**Images with different dates at different places.** envoi selects the image date of each point from the start and end times of the images at the points of the batch, which it reads once per batch. In some collections, the images have different dates at different places, for example the tiles of Sentinel-2. A point can then have no image on the selected date, although the collection has images at the point. Such a point gets an image of its own, which Earth Engine selects in the point's own request, first within 32 days of the point's date, then within 366 days (with `"contains"`, an image that contains the date is used at any distance):

- `"nearest"` — the point's image with the fewest whole days between its start day and the point's date. With an equal number of days, the image that starts earlier.
- `"contains"` — the point's image that contains the point's date, with the latest start. Otherwise the image with the latest start on or before the point's date, and then the image with the earliest start after it. Also when the batch's selected image lasts longer, envoi does not use a point's image that starts after the point's date if the point has an image that contains the date.

envoi compares whole days in UTC. The point's date is limited to the range from the first to the latest image start of the batch, so a date after the latest start (also a date inside the last image's time range) searches around the latest start day. The search costs time only for these points: in live checks with Sentinel-2, the request of such a point took about 1 s longer, and up to about 3 s longer when the search needs the 366-day window. When the point has no image in the search (and, with `"contains"`, none that contains its date), it is a [failed point](#failed-points). The QC columns `<dataset>_image_time_start_<N>m` and `<dataset>_image_time_end_<N>m`, and `image_date_ranges_used` in the sidecar `date_info`, name the image that each point actually used. `<dataset>_date_clamped_<N>m` and `<dataset>_date_source_<N>m` compare the point's date with the images of the batch, as before. `image_time_end` is empty when the image has no end time. A point where the collection has no image at all (see [Failed points](#failed-points)) has no date values, and `date_info` does not count it. In raster runs, envoi sends one more small request for each tile to read these dates. When this request fails, the tile is a failed tile.

envoi gives one warning for each dataset and window when a point used an image of its own that is more than 32 days from the point's date: in whole days to the image's start day with `"nearest"`, and to the nearest day of the image with `"contains"` (0 when the image contains the date). Points without a date, and points whose date is clamped, do not count. The warning gives the number of these points, the largest gap in days, and the IDs of up to 10 of them, and the sidecar records it in `warnings` under the key `far_image_<dataset>_<N>m` (`far_image_<dataset>_point` for point-only statistics).

**No date column.** Without an `eventDate` column, envoi uses the most recent image of each collection. envoi does not combine images for this. For each point, it takes the latest start time of the images that intersect the points of the batch, and selects the point's image that starts on that day (with `"contains"`, the point's image of the last time range). A pixel that this image masks stays masked, also when an older image has a value there. When the point has no image that starts on that day (possible in collections whose tiles have different dates), it gets an image of its own near that day, as "Images with different dates at different places" above describes. The QC column `<dataset>_date_source_<N>m` shows `most_recent_no_date`. Two kinds of collection differ: with `collection_date_policy: mosaic`, envoi ignores dates and mosaics the tiles at each point; and a collection without an image at any point of the batch is used as a mosaic of the whole collection (`ee.ImageCollection.mosaic()`), which gives these points no data. envoi prints a notice that there is no date column, and the sidecar records it in `warnings`. This is the right behaviour for static datasets or when "most recent" is genuinely what you need, but it does mean a quiet date-less run on a time-varying dataset is hiding an assumption — the sidecar is where to verify it.

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

`window_size_m` accepts a list as well as a single integer. Each window size produces its own set of reducer columns (tabular) or its own set of tiles (raster), all in the same output file or batch folder. In raster mode, the tiles of every window size are in the same dataset folder; each file name ends with `-<N>m`, and each window size has its own manifest, `tiles_manifest-<N>m.csv`:

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
        {"sr_landsat_annual": ["red", "nir"]},             # narrow bands for this run only
        {"dem_copernicus_glo30": ["DEM", "slope"]},        # mix source + derived band names
    ],
    "settings": {"output_type": "tabular", "statistics": ["mean"], "window_size_m": 200},
})
```

The override **replaces** the catalog's `bands` list for that dataset in that run (no merging) — if you want to keep a band the catalog already defines, list it again explicitly. This is also true for a list of only derived bands: `{"dem_copernicus_glo30": ["aspect"]}` gives only the aspect band, without the catalog's `DEM` band.

**Derived bands.** Names recognised as derived (currently `slope` and `aspect`, both computed on the fly from the dataset's first band) are split out automatically — you don't need a separate key. Derived bands only make physical sense for some datasets (DEMs, primarily), so each catalog entry that supports them declares a `supported_derived_bands` whitelist. A per-call request that isn't in that list is rejected up front with a clear error, rather than silently producing a meaningless layer. Derived bands are also Earth-Engine-only — requesting `slope` on a local raster raises `ValueError` immediately, since the local adapter doesn't run the gradient computation.

**Column-naming consequences.** Every per-call band list, also a one-element list, gives columns named after the bands, for Earth Engine and local datasets. For example, `{"sr_landsat_annual": ["red"]}` gives `sr_landsat_annual_red_mean_200m`, `{"dem_copernicus_glo30": ["DEM"]}` gives `dem_copernicus_glo30_DEM_mean_200m`, and `{"dem_copernicus_glo30": ["aspect"]}` gives only `dem_copernicus_glo30_aspect_mean_200m`. Use the catalog default if you instead want the bare single-band form of a dataset with one band (`dem_copernicus_glo30_mean_200m`) — the distinction reflects "the user picked bands" versus "the dataset has only one band", which matters when downstream code keys off column names.

> **Band identifiers differ by data source.** Earth Engine identifies bands by *string* names (`"red"`, `"DEM"`, `"slope"`) because GEE exposes named bands. Local GeoTIFFs identify bands by *integer* index (`1`, `2`, `[1, 2, 3]`) because rasterio reads bands positionally. A GEE catalog entry might set `bands: ["DEM"]` and a local one `bands: 1`, and per-call overrides follow the same convention as the dataset they target.

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

**Auto-detection for local rasters.** For `data_source: local` entries, you only need to provide `data_source` and `path`. When an extraction opens the file, envoi reads the CRS, native pixel size, nodata value, and band count from it with rasterio. When you register a YAML file, envoi also reads each local raster at registration time and adds the missing `crs`, `resolution_m` (in metres also for rasters whose CRS is in degrees or feet), `nodata`, and `band_count` to its catalog entry, which `list_datasets("full")` shows. Entries that you register as a dict are stored without this step. `resolution_m` and `band_count` are for information only: envoi always uses the pixel size and the band count of the file. Set `data_type: categorical` explicitly if the raster is a class map; local entries default to `continuous` otherwise.

The catalog keys `crs` and `nodata` of a local entry only fill a value that the file does not declare. They never replace a value of the file:

- `nodata` is one number for all bands (a GeoTIFF has one nodata value for all its bands). When the file declares no nodata value, envoi treats the pixels with the catalog `nodata` as missing in every selected band: in the `point` statistic, in the window statistics, and in the tiles, which declare this value as their nodata value. The value must fit the data type of each selected band: for an integer band, a whole number in the range of the type (for example 0 to 255 for `uint8`; `-9999.0` is accepted as `-9999`); for a float band, NaN (`.nan` in YAML) or a number in the range of the type (for `float32`, from about -3.4e38 to 3.4e38). envoi uses the value as the band stores it: a `float32` band stores `-3.4e38` as `-3.3999999521443642e38`, and envoi uses and records this number. When the file declares a nodata value, envoi uses it, and a catalog `nodata` with another value stops the run with a `ValueError` that names both values.
- `crs`: When the file has no CRS, envoi uses the catalog `crs` for all reads and tiles. A file without a CRS and without a catalog `crs` stops the run with a `ValueError`. When the file has a CRS, envoi uses it, and a catalog `crs` that is a different CRS stops the run with a `ValueError`. The values that envoi fills in from the file at registration never cause these errors.

envoi checks these settings for every local dataset of an `extract()` call, with the bands that each run config selects, before it reads any data or writes any output. envoi uses a nodata value of the file also as the band stores it. The metadata sidecar records the nodata value that envoi used (`nodata`: a number, the text `"nan"`, `"inf"`, or `"-inf"` for a value that is not finite, or `null` when there is none) and where it came from (`nodata_source`: `"file"`, `"catalog"`, or `null`).

**Resetting the catalog.** Call `reset_catalog()` to drop everything registered at runtime and revert to the built-in catalog alone. Mostly useful in tests or notebooks where you want a clean slate between cells.

For the full catalog schema, see the commented reference block at the top of [src/envoi/configs/ee_catalog.yml](../src/envoi/configs/ee_catalog.yml) — it's a copy-paste starting point for adding a new entry. A runnable starter that registers both a local raster and an Earth Engine asset lives at [examples/catalog.yml](../examples/catalog.yml).

## Failed points

A *failed point* is a point whose data request or read raised an error, for example a temporary Earth Engine error, a network timeout, or a damaged block in a local GeoTIFF. A point outside the dataset's extent, or a point whose window has only nodata pixels, is not a failed point: it has no error, and the QC columns show it as before. This also applies to an Earth Engine collection that has no image at the point on any date (for example an ocean point of a dataset that covers only land). Such a point gets the values of a point without data: its statistics are empty, except `count` and the per-class columns of `class_count` and `class_fraction`, which are `0`; `in_extent` is `False`; and in raster runs its tile has only masked pixels. A point that has images of the collection, but none on the selected dates, gets an image of its own near its date (see "Images with different dates at different places" in [Date-aware extraction](#date-aware-extraction)). It is a failed point only when envoi selects no image for it: when none of its images starts within 366 days of its centre day (and, with `"contains"`, none contains the centre day), or when none is in the point's UTM zone with `use_utm_zone`. The centre day is the point's date (without a date, the latest image start of the batch), limited to the range from the first to the latest image start of the batch, as "Images with different dates at different places" in [Date-aware extraction](#date-aware-extraction) describes. A problem with the image selection is then not reported as missing data. Its error says that a parameter "is required and may not be null", for example `Image.reduceRegion: Parameter 'image' is required and may not be null.` or `Element.get: Parameter 'object' is required and may not be null.`: envoi found no image at the point after the date and UTM-zone filters. Check the dataset's `collection_date_policy` (and `use_utm_zone`) in the catalog, and the point's sample date against the dates of the collection's images.

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

## Known limitations

The items below are known limitations of envoi. They do not stop a run, but they change how you can read some outputs. Each item tells what you see, why, and what to do.

**Pixels at the edge of the window.** `n_pixels`, `coverage_pct`, and `had_nodata` can differ between Earth Engine and local datasets for the same window. They can also differ between two Earth Engine runs with different statistics. For example, an Earth Engine run whose only statistic is `count` can show a `coverage_pct` a few percent below 100 for a window without missing data. `had_nodata` is then `True`. The cause is that different rules apply to the pixels at the edge of the window:

- Earth Engine also counts the pixels that are only partly inside the window, each with a weight below 1.
- When `count` is the only statistic, Earth Engine counts only whole pixels.
- For a local dataset, envoi counts a pixel as one whole pixel when its centre is inside the window. It does not count the other pixels, neither as valid pixels nor in the total of `coverage_pct`. So a local window without missing data has `coverage_pct` 100. The total also counts the pixels of the window beyond the edge of the raster file (on the raster's pixel grid), which have no data. So a window half outside the raster has a `coverage_pct` of about 50, as on Earth Engine.

Use these QC values as approximate values. Compare them only between runs with the same kind of dataset and the same statistics. For the same reason, the window statistics of the two kinds of dataset can differ a little (see `std` and `var` in [Tabular outputs](../README.md#tabular)).

**`in_extent` of windows without valid pixels.** For a window without valid pixels, for example a window with only nodata pixels, `in_extent` differs between the two kinds of dataset. Earth Engine datasets set it to `False`. Local datasets set it to `True` when the window overlaps the raster, with `n_pixels` `0`. [Tabular outputs](../README.md#tabular) in the README describes this difference and the related difference in `count`. For a local dataset, `in_extent` tells whether the window overlaps the raster file. For an Earth Engine dataset, it tells whether the window has at least one valid pixel. To find the windows without valid pixels in both kinds of dataset, use `n_pixels` or `coverage_pct`, which are `0` for such a window. Do not use `in_extent` for this.

**Class columns when points failed.** When a point failed, the `class_count` columns of that dataset and window hold decimal numbers, for example `3.0` in a CSV file. Without failed points, they hold whole numbers. So the column types of a Parquet file (its schema) can differ between two runs with the same settings. The cause is that the class columns of a [failed point](#failed-points) are empty, because its classes are unknown. pandas stores a column of whole numbers with empty values as decimal numbers. A point without data is not a failed point: its class columns are `0` (see [Mixing categorical and continuous datasets](#mixing-categorical-and-continuous-datasets)). The `class_fraction` columns always hold decimal numbers. After you read the output, set the type of the class columns that you use. For example, `df[columns] = df[columns].astype("Int64")` in pandas gives whole numbers and keeps the empty values. Or give a fixed schema when you read or combine Parquet files.

**Tiles from an earlier run.** A tile folder can contain tiles that the current run did not write. envoi writes the tiles to `<output_dir>/<batch_id>/<dataset>/` and replaces a file with the same name. It does not remove the other files in the folder. So a tile from an earlier run with the same `output_dir` and `batch_id` stays when the current run writes no tile with that name. For example:

- The earlier run had other points.
- The tile name of an ID changed, for example because the ID now gets a hash suffix (see [Raster outputs](../README.md#raster)).
- The point has no tile in the current run, for example a local point outside the raster. envoi removes an older tile with the same name only in some cases, for example when the download of an Earth Engine tile fails.

A tile manifest from an earlier run with other window sizes also stays. The tile manifest of the current run (`tiles_manifest.csv`, or `tiles_manifest-<N>m.csv` with several window sizes) is the record of what the current run wrote. A tile comes from the current run only when this manifest lists it with `exported` `True`. Use a new output folder (`output_dir`) or a new `batch_id` for each run. Or delete the old tile folder before you run the extraction again.

**Paths in the metadata sidecar.** The metadata sidecar records the `path` of each dataset as you registered it, in `datasets.<dataset>.path`. For a local dataset, this can be an absolute path on your computer, for example `/home/<name>/data/dem.tif` or `C:\Users\<name>\data\dem.tif`. Such a path can show your user name and your folder structure. envoi records the path so that the sidecar shows which file the run used. Check the sidecar before you share it, and change or remove the path when necessary. A relative path, such as `data/dem.tif`, does not show these details. A relative path starts at the current working folder of Python. The error texts in the outputs can also contain private details (see the end of [Failed points](#failed-points)).

**Date information in runs with several windows.** In a run config with several window sizes, the sidecar has one `date_info` for each dataset. It describes only the first window size in your `window_size_m` list. The same is true for the `band_coverage_pct` summary of an Earth Engine dataset with several bands. envoi writes these values of a dataset once, from the first window. Only the summaries under `quality.<N>m` are per window. The window size does not change which image envoi selects for a point, so the windows usually use the same images. But the counts in `date_info` leave out failed points, and a point can fail in one window and not in another. For the dates of each window in a tabular run, use the date QC columns of that window, for example `<dataset>_image_time_start_<N>m` (see [Date-aware extraction](#date-aware-extraction)). A raster run has no QC table. For a raster run, put each window size in its own run config. Each run config gets its own sidecar (see [Multiple outputs in one call](#multiple-outputs-in-one-call)).
