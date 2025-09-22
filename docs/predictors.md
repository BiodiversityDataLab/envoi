# Predictors (v1)

- `dem_elev_mean_500m` (m): mean elevation in 500 m window
- `slope_mean_500m` (deg): derived from DEM (stretch goal)
- `cop_lc_majority_500m` (class): majority land-cover class in 500 m window
- `temp_mean_month` (°C): ERA5-Land monthly mean at event month (if chosen)

Each predictor carries: `_source`, `_version`, `_license`, `_resolution_m`, `_reducer`, `_sample_method`, `_temporal_rule`.
