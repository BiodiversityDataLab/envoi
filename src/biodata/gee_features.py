# src/biodata/gee_features.py
"""
Generic GEE image-building utilities.

All operations are driven by config (feature_spec) rather than
hard-coded dataset knowledge.  The adapter calls these functions
with whatever keys the catalog provides.
"""
from __future__ import annotations

import logging

import pandas as pd

try:
    import ee
except ImportError:
    ee = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection reduction
# ---------------------------------------------------------------------------

_COLLECTION_REDUCERS = {
    "mean": lambda col: col.mean(),
    "median": lambda col: col.median(),
    "mode": lambda col: col.mode(),
    "mosaic": lambda col: col.mosaic(),
    "min": lambda col: col.min(),
    "max": lambda col: col.max(),
    "sum": lambda col: col.sum(),
    "first": lambda col: col.first(),
}


def reduce_collection(col: ee.ImageCollection, reducer: str) -> ee.Image:
    """Reduce an ImageCollection to a single Image using *reducer* name."""
    fn = _COLLECTION_REDUCERS.get(reducer)
    if fn is None:
        logger.warning("Unknown collection reducer '%s', falling back to mean", reducer)
        return col.mean()
    return fn(col)


# ---------------------------------------------------------------------------
# Date filtering
# ---------------------------------------------------------------------------


def filter_collection_by_date(
    col: ee.ImageCollection,
    date: pd.Timestamp,
    feature_spec: dict,
) -> ee.ImageCollection:
    """Filter a collection around a sample date using feature_spec config.

    Supported keys:
        temporal_window_days: int  — symmetric window ±(days-1)/2
        start_date / end_date: str — fixed global range (ignores sample date)
    """
    start_date = feature_spec.get("start_date")
    end_date = feature_spec.get("end_date")

    if start_date and end_date:
        return col.filterDate(start_date, end_date)

    days = feature_spec.get("temporal_window_days")
    if days and date is not None:
        half = (days - 1) / 2
        start = (date - pd.DateOffset(days=half)).strftime("%Y-%m-%d")
        end = (date + pd.DateOffset(days=half)).strftime("%Y-%m-%d")
        return col.filterDate(start, end)

    return col


# ---------------------------------------------------------------------------
# Cloud masking (optical sensors)
# ---------------------------------------------------------------------------


def mask_clouds_s2(image: ee.Image) -> ee.Image:
    """Mask clouds and cirrus for Sentinel-2 using QA60 band."""
    qa = image.select("QA60")
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
    mask = cloud_mask.And(cirrus_mask)
    return (
        image.updateMask(mask)
        .divide(10000)
        .select("B.*")
        .copyProperties(image, ["system:time_start"])
    )


_CLOUD_MASK_FNS = {
    "s2": mask_clouds_s2,
}


def apply_cloud_mask(col: ee.ImageCollection, mask_type: str) -> ee.ImageCollection:
    """Apply a cloud mask function to a collection, if mask_type is known."""
    fn = _CLOUD_MASK_FNS.get(mask_type)
    if fn is None:
        logger.warning("Unknown cloud_mask type '%s', skipping", mask_type)
        return col
    return col.map(fn)


# ---------------------------------------------------------------------------
# Derived bands
# ---------------------------------------------------------------------------


def apply_derived_band(img: ee.Image, derived: str) -> ee.Image:
    """Compute a derived band from an image.

    Supported values:
        NDVI, EVI, slope, aspect
    """
    if derived == "NDVI":
        return img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    if derived == "EVI":
        return img.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
        ).rename("EVI")
    if derived == "slope":
        return ee.Terrain.slope(img)
    if derived == "aspect":
        return ee.Terrain.aspect(img)

    logger.warning("Unknown derived_band '%s', returning image unchanged", derived)
    return img


# ---------------------------------------------------------------------------
# Generic image builder
# ---------------------------------------------------------------------------


def build_image(feature_spec: dict, date: pd.Timestamp | None = None) -> ee.Image:
    """Build an ee.Image from a feature_spec config dict.

    Handles the full pipeline:
        1. Load image or collection
        2. Filter collection by date (if applicable)
        3. Apply cloud masking (if configured)
        4. Reduce collection to single image
        5. Select band (if configured)
        6. Apply derived band (if configured)

    Catalog feature_spec keys:
        image: str                  — single image asset ID
        collection: str             — image collection asset ID
        collection_reducer: str     — how to reduce collection (default: mean)
        temporal_window_days: int   — date window for collection filtering
        start_date / end_date: str  — fixed date range
        cloud_mask: str             — cloud mask type (e.g. "s2")
        cloud_pct_max: int          — max cloud percentage filter
        band: str | int             — band to select
        derived_band: str           — derived band to compute (NDVI, EVI, slope, aspect)
    """
    img = None

    if "image" in feature_spec:
        img = ee.Image(feature_spec["image"])

    elif "collection" in feature_spec:
        col = ee.ImageCollection(feature_spec["collection"])

        # Date filtering
        if date is not None:
            col = filter_collection_by_date(col, date, feature_spec)
        elif feature_spec.get("start_date") and feature_spec.get("end_date"):
            col = col.filterDate(feature_spec["start_date"], feature_spec["end_date"])

        # Cloud percentage filter
        cloud_pct = feature_spec.get("cloud_pct_max")
        if cloud_pct is not None:
            col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))

        # Cloud masking
        cloud_mask = feature_spec.get("cloud_mask")
        if cloud_mask:
            col = apply_cloud_mask(col, cloud_mask)

        # Reduce collection
        reducer = feature_spec.get("collection_reducer", "mean")
        img = reduce_collection(col, reducer)

    else:
        raise ValueError("feature_spec must contain 'image' or 'collection'")

    # Band selection
    band = feature_spec.get("band")
    if band is not None:
        img = img.select(band)

    # Derived band
    derived = feature_spec.get("derived_band")
    if derived:
        img = apply_derived_band(img, derived)

    return img
