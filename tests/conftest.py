"""Shared pytest fixtures for the envoi test suite.

The tests historically pointed at a real DEM GeoTIFF and a sample CSV that
lived under ``data/for_testing/`` — files only the original author had on
disk. To make the suite portable (so CI and contributors can run it
without the binary fixtures), the equivalents are generated synthetically
here.

The synthetic DEM and sample DataFrame mirror the spatial extent, CRS,
and schema of the originals so the existing assertions in test_extract.py
still hold without modification.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from envoi import reset_catalog


@pytest.fixture(autouse=True)
def _reset_user_catalog():
    """Clear the session-wide user catalog around every test.

    ``envoi.update_catalog()`` stores registered datasets in a module-level
    dict that persists for the lifetime of the Python process. Without this
    fixture, a test that forgets to call ``reset_catalog()`` in a teardown
    block leaks its datasets into the next test, producing order-dependent
    failures. Running ``reset_catalog()`` both before and after each test
    guarantees a clean slate regardless of how the test exits (pass, fail,
    or interrupted mid-body).
    """
    reset_catalog()
    yield
    reset_catalog()


# Spatial parameters copied from the original DEM fixture so the synthetic
# data covers the same area the sample points fall inside. The points are
# around (lat 62.97-62.98N, lon 18.02-18.03E) — i.e. UTM 34N — and the
# bounds below extend ~5 km in each direction around that cluster.
_DEM_CRS = "EPSG:32634"  # UTM zone 34N
_DEM_RES_M = 10.0
_DEM_HEIGHT = 500
_DEM_WIDTH = 500
_DEM_ORIGIN_X = 347020.0  # west edge (UTM x)
_DEM_ORIGIN_Y = 6988980.0  # north edge (UTM y; transform expects top-left)
_DEM_BAND_COUNT = 3  # matches the original 3-band file used by multi_band tests


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Six sample points inside the synthetic DEM's extent.

    Coordinates and ids are copied from the original ``adrian_example.csv``
    so any test that compares against id strings or relies on the points'
    relative spatial layout keeps working unchanged.
    """
    return pd.DataFrame(
        {
            "id": ["T2T1DJ", "T6WPKD", "TCSZX1", "TG4NHB", "TVCC3Y", "T4P82F"],
            "n_otu": [2307, 1597, 1058, 2294, 3854, 1700],
            "lat": [62.976878, 62.981296, 62.976671, 62.976843, 62.976773, 62.978500],
            "lon": [18.026823, 18.030991, 18.021154, 18.030649, 18.016718, 18.025000],
            "date": [
                "2025-06-18",
                "2020-12-12",
                "1960-04-02",
                "1975-02-28",
                "1999-06-30",
                "2010-03-15",
            ],
        }
    )


@pytest.fixture(scope="session")
def dem_tif(tmp_path_factory) -> Path:
    """Three-band float32 synthetic DEM, written once per test session.

    Session scope keeps the file alive across every test that uses it, so
    we only pay the rasterio.write cost once. The path lives under pytest's
    tmp directory and is cleaned up automatically when the session ends.

    Band value ranges roughly match the original raster (elevation, slope,
    aspect) so tests that compute coverage statistics observe data in the
    same magnitude as before. Exact values are not asserted on by any
    existing test — only schema, coverage, and column presence are.
    """
    # Seeded RNG so the file is byte-stable across runs. Tests that look at
    # specific stat values would otherwise flake from one CI run to the next.
    rng = np.random.default_rng(seed=42)
    elevation_band = rng.uniform(26.0, 286.0, (_DEM_HEIGHT, _DEM_WIDTH)).astype(np.float32)
    slope_band = rng.uniform(0.0, 46.0, (_DEM_HEIGHT, _DEM_WIDTH)).astype(np.float32)
    aspect_band = rng.uniform(0.0, 355.0, (_DEM_HEIGHT, _DEM_WIDTH)).astype(np.float32)

    # Use tmp_path_factory rather than tmp_path because tmp_path is
    # function-scoped — it would create (and tear down) a fresh DEM for
    # every test, which is wasteful for a session-stable fixture.
    fixtures_dir = tmp_path_factory.mktemp("envoi_fixtures")
    dem_path = fixtures_dir / "synthetic_dem.tif"

    # rasterio.transform.from_origin builds a north-up affine from the
    # raster's top-left corner — the convention used by every GeoTIFF in
    # the wild. Passing pixel size for both x and y keeps the pixels square.
    transform = from_origin(_DEM_ORIGIN_X, _DEM_ORIGIN_Y, _DEM_RES_M, _DEM_RES_M)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=_DEM_HEIGHT,
        width=_DEM_WIDTH,
        count=_DEM_BAND_COUNT,
        dtype="float32",
        crs=_DEM_CRS,
        transform=transform,
    ) as dst:
        dst.write(elevation_band, 1)
        dst.write(slope_band, 2)
        dst.write(aspect_band, 3)

    return dem_path


# Constant value (in metres) baked into every pixel of the constant-valued DEM.
# Picked to be:
#   * non-zero (so accidental zeros from a default fill would be obvious),
#   * non-integer-friendly (50.0 not 1.0) so mean/min/max look distinct in output,
#   * inside the realistic elevation range so the typed-stats validator doesn't warn.
_CONSTANT_DEM_VALUE = 50.0


@pytest.fixture(scope="session")
def constant_dem_tif(tmp_path_factory) -> Path:
    """Single-band float32 DEM where every pixel equals _CONSTANT_DEM_VALUE.

    Used by the numerical-correctness tests: when every pixel inside a window
    is the same value, mean / min / max must equal that value and std must be
    zero. No randomness, no rounding tolerance needed — exact equality checks.

    Spatial parameters (CRS, origin, resolution, size) match ``dem_tif`` so the
    same sample points fall inside both rasters.
    """
    # Same CRS/transform/dimensions as the random DEM above — keeps the
    # synthetic-fixture layout uniform so tests using either fixture share
    # the same sample-point coverage.
    constant_band = np.full((_DEM_HEIGHT, _DEM_WIDTH), _CONSTANT_DEM_VALUE, dtype=np.float32)

    fixtures_dir = tmp_path_factory.mktemp("envoi_const_fixtures")
    constant_dem_path = fixtures_dir / "constant_dem.tif"

    transform = from_origin(_DEM_ORIGIN_X, _DEM_ORIGIN_Y, _DEM_RES_M, _DEM_RES_M)
    with rasterio.open(
        constant_dem_path,
        "w",
        driver="GTiff",
        height=_DEM_HEIGHT,
        width=_DEM_WIDTH,
        count=1,
        dtype="float32",
        crs=_DEM_CRS,
        transform=transform,
    ) as dst:
        dst.write(constant_band, 1)

    return constant_dem_path


@pytest.fixture(scope="session")
def constant_dem_value() -> float:
    """The exact value stored in every pixel of the ``constant_dem_tif`` fixture.

    Exposed as a fixture so tests don't have to import the private
    ``_CONSTANT_DEM_VALUE`` constant — they pull both the file and the value
    via the public fixture API.
    """
    return _CONSTANT_DEM_VALUE
