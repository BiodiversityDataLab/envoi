# QC Spec

## Schema
- Input: id, lat (-90..90), lon (-180..180), [date ISO]
- Output: id unique, required predictor columns present

## Value checks
- elevation: -430..9000
- temp_mean_month (°C): -60..60
- land-cover: valid classes

## Coverage
- ≥95% non-null predictors
