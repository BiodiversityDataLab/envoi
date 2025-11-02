import pandas as pd
from biodata.enrich import enrich

df = pd.read_csv("data/points_sample.csv")
cfg = {
  "groups": [{
    "name": "dem_100m",
    "predictors": ["dem_mini"],
    "output": {"kind": "tabular", "reducers": ["mean","std","q10","q90"], "window_m": 100}
  }],
  "min_coverage_pct": 80, "project_crs": "EPSG:3006",
}

enrich(df, groups=cfg, catalog="configs/catalog.yml", out_dir="out")





# Project_crs -- Projected coordinate system (for distance in meters)
# Predictors -- Features
# Redcuers -- Summary_statistics
# kind -- output_datatype
# window_m -- buffer_radius

# Sanity checks -- What if window size is smaller/larger than raster resolution?
# Image_output --  
# If predictor doesnt exist -- Error
# If predictor exists but not in catalog -- Error

# Retry functionality for GEE etc -- If some points fail, retry only those points




# import pandas as pd
# df = pd.read_parquet("out/dem_100m.parquet")
# df.head()                 # preview
# df.filter(like="dem_mini")# only predictor + QA cols
# df[df["dem_mini_coverage_pct"]<80][["id","dem_mini_coverage_pct"]]

# print(df)



# cfg = {
#   "groups": [
#     {
#       "name": "terrain_100m",
#       "predictors": ["dem_mini", "slope_mini"],      # both must be in catalog.yml
#       "output": {"kind":"tabular","reducers":["mean","std"],"window_m":100}
#     },
#     {
#       "name": "terrain_300m",
#       "predictors": ["dem_mini"],
#       "output": {"kind":"tabular","reducers":["mean","q10","q90"],"window_m":300}
#     }
#   ],
#   "min_coverage_pct": 50,
#   "project_crs": "EPSG:3006",
# }


# 