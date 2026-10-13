"""Unit tests for envoi.geo: the raster tile grid and pixel sizes in metres.

The tile helpers define the pixel grid of every raster tile on both data
sources, so the tests check the rule that the README states: round(window /
pixel) pixels per side with ties rounded up, corners on the pixel grid, and
the point in the centre pixel for odd sizes. The expected values come from
independent calculations, not from the helpers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rasterio.crs import CRS

from envoi.geo import get_utm_zone_label, pixel_size_m, tile_pixel_count, tile_top_left

# ------------------------------------------------------------------
# tile_pixel_count — pixels per side of a tile
# ------------------------------------------------------------------


class TestTilePixelCount:
    @pytest.mark.parametrize(
        ("window_m", "pixel_m", "expected_pixels"),
        [
            (200, 30, 7),  # 6.67
            (200, 10, 20),
            (200, 50, 4),
            (200, 25, 8),
            (40, 30, 1),  # 1.33
            (500, 30.92, 16),  # 16.17, a native Earth Engine scale
        ],
    )
    def test_rounds_to_the_nearest_whole_number(self, window_m, pixel_m, expected_pixels):
        """The count is window / pixel rounded to the nearest whole number."""
        assert tile_pixel_count(window_m, pixel_m) == expected_pixels

    @pytest.mark.parametrize(
        ("window_m", "pixel_m", "expected_pixels"),
        [
            (45, 30, 2),  # 1.5
            (75, 30, 3),  # 2.5; round() would give 2
            (105, 30, 4),  # 3.5
            (25, 10, 3),  # 2.5; round() would give 2
        ],
    )
    def test_ties_round_up(self, window_m, pixel_m, expected_pixels):
        """A window of exactly n + 0.5 pixels gives n + 1 pixels, also for even n."""
        assert tile_pixel_count(window_m, pixel_m) == expected_pixels

    @pytest.mark.parametrize("window_m", [1, 10, 14.9])
    def test_minimum_is_one_pixel(self, window_m):
        """A window smaller than half a pixel still gives a 1 × 1 tile."""
        assert tile_pixel_count(window_m, 30) == 1


# ------------------------------------------------------------------
# tile_top_left — tile corners on the pixel grid
# ------------------------------------------------------------------

# Seeded UTM points: eastings of the UTM range, northings in the northern
# hemisphere and in the southern hemisphere (false northing 10 000 000 m).
_random_generator = np.random.default_rng(seed=19)
_UTM_POINTS = [
    *zip(
        _random_generator.uniform(166_000, 834_000, 20), _random_generator.uniform(0, 9_300_000, 20)
    ),
    *zip(
        _random_generator.uniform(166_000, 834_000, 20),
        _random_generator.uniform(1_100_000, 9_999_000, 20),
    ),
    # A southern-hemisphere point near the equator (northing close to 10 000 000 m).
    (500_123.4, 9_999_876.5),
]
_PIXEL_SIZES_M = [10, 30, 2.5, 30.92]


def _is_multiple(value: float, step: float) -> bool:
    """Return True when ``value`` is a whole multiple of ``step`` (within rounding error)."""
    return math.isclose(value / step, round(value / step), abs_tol=1e-6)


class TestTileTopLeft:
    @pytest.mark.parametrize("n_pixels", [1, 3, 7, 9])
    @pytest.mark.parametrize("pixel_m", _PIXEL_SIZES_M)
    def test_odd_size_has_the_point_in_the_centre_pixel(self, n_pixels, pixel_m):
        """For odd n, the pixel that contains the point is the centre pixel of the tile."""
        centre_index = (n_pixels - 1) // 2
        for x_m, y_m in _UTM_POINTS:
            x_left, y_top = tile_top_left(x_m, y_m, n_pixels, pixel_m)

            # The column counts pixels east of the west edge, the row counts
            # pixels south of the north edge (raster order).
            column = math.floor((x_m - x_left) / pixel_m)
            row = math.floor((y_top - y_m) / pixel_m)
            assert (column, row) == (centre_index, centre_index)

    @pytest.mark.parametrize("n_pixels", [2, 6, 20])
    @pytest.mark.parametrize("pixel_m", _PIXEL_SIZES_M)
    def test_even_size_is_centred_on_the_nearest_grid_line(self, n_pixels, pixel_m):
        """For even n, the tile centre is the grid line nearest to the point."""
        half_side_m = n_pixels * pixel_m / 2
        for x_m, y_m in _UTM_POINTS:
            x_left, y_top = tile_top_left(x_m, y_m, n_pixels, pixel_m)

            nearest_x_line = _nearest_grid_line(x_m, pixel_m)
            nearest_y_line = _nearest_grid_line(y_m, pixel_m)
            assert x_left + half_side_m == pytest.approx(nearest_x_line, abs=1e-6)
            assert y_top - half_side_m == pytest.approx(nearest_y_line, abs=1e-6)

    @pytest.mark.parametrize("n_pixels", [1, 2, 7, 20])
    @pytest.mark.parametrize("pixel_m", _PIXEL_SIZES_M)
    def test_corners_are_on_the_pixel_grid(self, n_pixels, pixel_m):
        """Both corners of every tile are multiples of the pixel size."""
        for x_m, y_m in _UTM_POINTS:
            x_left, y_top = tile_top_left(x_m, y_m, n_pixels, pixel_m)
            assert _is_multiple(x_left, pixel_m)
            assert _is_multiple(y_top, pixel_m)

    def test_200_m_at_30_m(self):
        """A worked example: 7 pixels of 30 m around a point inside pixel (11570, 232818)."""
        x_left, y_top = tile_top_left(347_123.4, 6_984_567.8, 7, 30)
        # The point's pixel starts at 347 100 (x) and 6 984 540 (y). Three
        # pixels lie west of it, and three pixels north of it.
        assert x_left == pytest.approx(347_100 - 3 * 30)
        assert y_top == pytest.approx(6_984_540 + 30 + 3 * 30)


def _nearest_grid_line(coordinate_m: float, pixel_m: float) -> float:
    """Return the multiple of ``pixel_m`` nearest to ``coordinate_m``."""
    below = math.floor(coordinate_m / pixel_m) * pixel_m
    above = below + pixel_m
    return below if coordinate_m - below < above - coordinate_m else above


# ------------------------------------------------------------------
# pixel_size_m — pixel width in metres
# ------------------------------------------------------------------


class TestPixelSizeM:
    def test_degrees_use_the_length_of_one_degree_at_the_equator(self):
        """0.001° gives 111.31949 m (111 319.49 m per degree)."""
        assert pixel_size_m(CRS.from_epsg(4326), 0.001) == pytest.approx(111.31949)

    def test_other_angular_unit_is_converted_to_degrees(self):
        """A geographic CRS in grads (NTF Paris): 0.01 grad = 0.009°."""
        assert pixel_size_m(CRS.from_epsg(4807), 0.01) == pytest.approx(0.009 * 111_319.49)

    def test_metres_are_unchanged(self):
        """A projected CRS in metres (UTM 34N) keeps the pixel width."""
        assert pixel_size_m(CRS.from_epsg(32634), 30.0) == pytest.approx(30.0)

    def test_international_feet(self):
        """A projected CRS in international feet: 1 ft = 0.3048 m."""
        assert pixel_size_m(CRS.from_epsg(2222), 10.0) == pytest.approx(3.048)

    def test_us_survey_feet(self):
        """EPSG:2263 is in US survey feet: 1 ft = 1200 / 3937 m (about 0.3048 m)."""
        assert pixel_size_m(CRS.from_epsg(2263), 10.0) == pytest.approx(10 * 1200 / 3937)

    def test_sign_is_ignored(self):
        """A negative pixel width (a flipped transform) gives a positive size."""
        assert pixel_size_m(CRS.from_epsg(4326), -0.001) == pytest.approx(111.31949)

    def test_no_crs_gives_none(self):
        """A raster without a CRS has no pixel size in metres."""
        assert pixel_size_m(None, 10.0) is None

    def test_engineering_crs_gives_none(self):
        """A CRS that is neither geographic nor projected gives None."""
        local_crs = CRS.from_wkt(
            'LOCAL_CS["arbitrary",UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
        )
        assert pixel_size_m(local_crs, 10.0) is None


# ------------------------------------------------------------------
# get_utm_zone_label — label for the GEE use_utm_zone filter
# ------------------------------------------------------------------


class TestUtmZoneLabel:
    def test_plain_zones_without_the_norway_and_svalbard_exceptions(self):
        """Bergen is in 31N and Ny-Ålesund in 32N, as the satellite-embedding tiles are labelled.

        UTM has exceptions in south-west Norway (Bergen in 32V) and on Svalbard
        (Ny-Ålesund in 33X). The ``UTM_ZONE`` property of the satellite-embedding
        tiles uses the plain 6° zones instead: a live check on 2026-10-12 found
        Bergen tiles labelled 31N, and Ny-Ålesund tiles labelled 32N and 33N.
        The ``use_utm_zone`` filter compares this label with that property, so
        the label must keep the plain zones.
        """
        assert get_utm_zone_label(5.32, 60.39) == "31N"
        assert get_utm_zone_label(11.93, 78.92) == "32N"
