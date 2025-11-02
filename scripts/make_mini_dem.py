# scripts/make_mini_dem.py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

out = Path("tests/data/mini_dem.tif")
out.parent.mkdir(parents=True, exist_ok=True)

# 10x10 grid, values 0..99
arr = np.arange(100, dtype=np.float32).reshape(10, 10)

# Cover roughly Sweden east/central: left=16.5, top=60.5,
# pixel size 0.2 degrees => right=18.5, bottom=58.5
transform = from_origin(16.5, 60.5, 0.2, 0.2)

with rasterio.open(
    out,
    "w",
    driver="GTiff",
    height=arr.shape[0],
    width=arr.shape[1],
    count=1,
    dtype=arr.dtype,
    crs="EPSG:4326",
    transform=transform,
) as dst:
    dst.write(arr, 1)

print(f"Wrote {out}")
