"""Tests for the catalog ``nodata`` and ``crs`` of local rasters.

The catalog values fill a nodata value or a CRS that the file does not
declare, in every read path: the point sample, the window statistics, the
native tile, and the ``resample_m`` tile. A catalog value that differs from
the file's own value raises ``ValueError`` before the first adapter is built.

Every raster is generated in ``tmp_path`` from a seeded random generator.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.transform import Affine, from_origin

from envoi import update_catalog
from envoi.adapters.local_adapter import LocalRasterAdapter
from envoi.catalog import load_catalog
from envoi.extract import extract

# ------------------------------------------------------------------
# Synthetic raster and points
# ------------------------------------------------------------------

_RASTER_CRS = "EPSG:32634"  # UTM zone 34N, the UTM zone of both points
_PIXEL_SIZE_M = 10.0
_RASTER_SIZE = 60  # pixels per side
_NODATA = -9999

# Point P is in pixel (row 30, column 30), 2.5 m from the west and north edges
# of that pixel. Point Q is 150 m east of P, in pixel (30, 45). Pixel centres
# are then never on the edge of a 200 m window: the window of P has the pixel
# centres of rows 20-39 and columns 20-39, and the window of Q those of rows
# 20-39 and columns 35-54.
_POINT_P_LAT, _POINT_P_LON = 62.976878, 18.026823
_TO_UTM = Transformer.from_crs("EPSG:4326", _RASTER_CRS, always_xy=True)
_POINT_P_X, _POINT_P_Y = _TO_UTM.transform(_POINT_P_LON, _POINT_P_LAT)
_POINT_Q_LON, _POINT_Q_LAT = Transformer.from_crs(
    _RASTER_CRS, "EPSG:4326", always_xy=True
).transform(_POINT_P_X + 150.0, _POINT_P_Y)
_RASTER_WEST = _POINT_P_X - 302.5
_RASTER_NORTH = _POINT_P_Y + 302.5
_RASTER_TRANSFORM = from_origin(_RASTER_WEST, _RASTER_NORTH, _PIXEL_SIZE_M, _PIXEL_SIZE_M)

_WINDOW_SIZE_M = 200
_WINDOW_P = (slice(20, 40), slice(20, 40))
_WINDOW_Q = (slice(20, 40), slice(35, 55))
_PIXEL_P = (30, 30)
_PIXEL_Q = (30, 45)

_POINTS = pd.DataFrame(
    {
        "occurrenceID": ["P", "Q"],
        "decimalLatitude": [_POINT_P_LAT, _POINT_Q_LAT],
        "decimalLongitude": [_POINT_P_LON, _POINT_Q_LON],
    }
)


def _pixel_values_with_nodata_pixels(band_count: int = 1) -> np.ndarray:
    """Return float32 values (bands, rows, columns) from 100 to 200, with -9999 pixels.

    A block of 10 × 10 pixels (rows and columns 25-34) around the pixel of P
    is -9999 in band 1, so P's point sample has no data. Six more -9999
    pixels are in the window of Q, but not at Q; one of them, (21, 38), is
    also in the window of P.
    """
    random_generator = np.random.default_rng(seed=61)
    pixel_values = random_generator.uniform(
        100.0, 200.0, (band_count, _RASTER_SIZE, _RASTER_SIZE)
    ).astype(np.float32)
    pixel_values[0, 25:35, 25:35] = _NODATA
    for row, column in [(21, 38), (22, 50), (27, 47), (33, 44), (36, 52), (39, 40)]:
        pixel_values[0, row, column] = _NODATA
    return pixel_values


def _write_raster(
    path: Path,
    pixel_values: np.ndarray,
    *,
    nodata: float | None = None,
    crs: str | None = _RASTER_CRS,
    transform: Affine = _RASTER_TRANSFORM,
) -> Path:
    """Write ``pixel_values`` (bands, rows, columns) as a GeoTIFF on the test grid.

    ``nodata`` and ``crs`` are written only when they are not None, so the
    file can lack a nodata value or a CRS. ``transform`` replaces the test
    grid, for example with a rotated grid.
    """
    profile = {
        "driver": "GTiff",
        "height": pixel_values.shape[1],
        "width": pixel_values.shape[2],
        "count": pixel_values.shape[0],
        "dtype": str(pixel_values.dtype),
        "transform": transform,
    }
    if crs is not None:
        profile["crs"] = crs
    if nodata is not None:
        profile["nodata"] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(pixel_values)
    return path


def _register(dataset_spec: dict) -> None:
    """Register one local dataset with the name ``grid``."""
    update_catalog({"datasets": {"grid": {"data_source": "local", **dataset_spec}}})


def _run_tabular(output_dir: Path, statistics: list[str], *, datasets=None) -> pd.DataFrame:
    """Run a tabular extraction of ``grid`` at both points and return the statistics table."""
    outputs = extract(
        _POINTS,
        {
            "batch_id": "table",
            "datasets": datasets or ["grid"],
            "settings": {
                "output_type": "tabular",
                "statistics": statistics,
                "window_size_m": _WINDOW_SIZE_M,
            },
        },
        output_dir=output_dir,
    )
    return pd.read_csv(outputs["table"])


def _run_raster(output_dir: Path, **settings) -> Path:
    """Export the tiles of ``grid`` at both points and return the tile folder."""
    extract(
        _POINTS,
        {
            "batch_id": "tiles",
            "datasets": ["grid"],
            "settings": {"output_type": "raster", "window_size_m": _WINDOW_SIZE_M, **settings},
        },
        output_dir=output_dir,
    )
    return output_dir / "tiles" / "grid"


def _sidecar_entry(sidecar_path: Path) -> dict:
    """Return the sidecar entry of the dataset ``grid``."""
    return json.loads(sidecar_path.read_text(encoding="utf-8"))["datasets"]["grid"]


def _source_pixel_of_tile_pixels(tile_transform) -> tuple[np.ndarray, np.ndarray]:
    """Return the source row and column of the centre of every pixel of a tile.

    Both the tile and the source raster are in ``_RASTER_CRS``.
    """
    rows, columns = np.mgrid[0:20, 0:20] + 0.5
    x_centres = tile_transform.c + columns * tile_transform.a
    y_centres = tile_transform.f + rows * tile_transform.e
    source_rows = np.floor((_RASTER_NORTH - y_centres) / _PIXEL_SIZE_M).astype(int)
    source_columns = np.floor((x_centres - _RASTER_WEST) / _PIXEL_SIZE_M).astype(int)
    return source_rows, source_columns


# ------------------------------------------------------------------
# Catalog nodata for a file without a nodata value
# ------------------------------------------------------------------


class TestCatalogNodataFillsMissingFileNodata:
    """A catalog ``nodata`` is used in every read path when the file declares none."""

    @pytest.fixture
    def pixel_values(self) -> np.ndarray:
        return _pixel_values_with_nodata_pixels()

    @pytest.fixture
    def catalog_nodata_raster(self, tmp_path, pixel_values) -> Path:
        """Register a raster without a nodata value, with ``nodata: -9999`` in the catalog."""
        raster_path = _write_raster(tmp_path / "rasters" / "no_nodata.tif", pixel_values)
        with rasterio.open(raster_path) as src:
            assert src.nodata is None
        _register({"path": str(raster_path), "nodata": _NODATA})
        return raster_path

    def test_point_sample_and_window_statistics_skip_nodata_pixels(
        self, tmp_path, pixel_values, catalog_nodata_raster
    ):
        """-9999 pixels are not data in the point sample and the window statistics."""
        stats_df = _run_tabular(tmp_path / "out", ["mean", "min", "count", "point"])
        qc_df = pd.read_csv(tmp_path / "out" / "table_qc.csv")

        band_values = pixel_values[0]
        valid_p = band_values[_WINDOW_P][band_values[_WINDOW_P] != _NODATA]
        valid_q = band_values[_WINDOW_Q][band_values[_WINDOW_Q] != _NODATA]
        # The window of P has the 10 x 10 nodata block and one more -9999
        # pixel. The window of Q has 6 -9999 pixels.
        assert (valid_p.size, valid_q.size) == (299, 394)
        assert stats_df["grid_mean_200m"].tolist() == pytest.approx(
            [valid_p.mean(), valid_q.mean()], abs=1e-4
        )
        assert stats_df["grid_min_200m"].tolist() == pytest.approx(
            [valid_p.min(), valid_q.min()], abs=1e-4
        )
        assert stats_df["grid_count_200m"].tolist() == [299, 394]
        assert qc_df["grid_n_pixels_200m"].tolist() == [299, 394]
        assert qc_df["grid_had_nodata_200m"].tolist() == [True, True]

        # P's pixel is -9999, so P has no point value.
        assert pd.isna(stats_df.loc[0, "grid_point"])
        assert stats_df.loc[1, "grid_point"] == pytest.approx(band_values[_PIXEL_Q], abs=1e-4)

        # A point-only run has QC columns for the point sample: P has the QC
        # values of a nodata pixel.
        _run_tabular(tmp_path / "point_only", ["point"])
        point_qc_df = pd.read_csv(tmp_path / "point_only" / "table_qc.csv")
        assert point_qc_df["grid_in_extent_point"].tolist() == [False, True]
        assert point_qc_df["grid_had_nodata_point"].tolist() == [True, False]

        sidecar_entry = _sidecar_entry(tmp_path / "out" / "table_metadata.json")
        assert sidecar_entry["nodata"] == _NODATA
        assert sidecar_entry["nodata_source"] == "catalog"

    def test_two_integer_bands_give_window_statistics(self, tmp_path):
        """Window statistics of a two-band uint16 raster work and skip the nodata pixels."""
        random_generator = np.random.default_rng(seed=67)
        pixel_values = random_generator.integers(
            100, 200, (2, _RASTER_SIZE, _RASTER_SIZE), dtype=np.uint16
        )
        pixel_values[0][25:35, 25:35] = 0
        raster_path = _write_raster(
            tmp_path / "rasters" / "two_uint16_bands.tif", pixel_values, nodata=0
        )
        _register({"path": str(raster_path)})

        stats_df = _run_tabular(tmp_path / "out", ["mean"], datasets=[{"grid": [1, 2]}])

        valid_mask_p = pixel_values[0][_WINDOW_P] != 0
        assert stats_df.loc[0, "grid_b1_mean_200m"] == pytest.approx(
            pixel_values[0][_WINDOW_P][valid_mask_p].mean(), abs=1e-4
        )
        assert stats_df.loc[0, "grid_b2_mean_200m"] == pytest.approx(
            pixel_values[1][_WINDOW_P][valid_mask_p].mean(), abs=1e-4
        )

    def test_two_bands_skip_pixels_that_are_nodata_in_one_band(self, tmp_path):
        """With two bands, a pixel that is -9999 in band 1 is not used for either band."""
        pixel_values = _pixel_values_with_nodata_pixels(band_count=2)
        raster_path = _write_raster(tmp_path / "rasters" / "two_bands.tif", pixel_values)
        _register({"path": str(raster_path), "nodata": _NODATA})

        stats_df = _run_tabular(tmp_path / "out", ["mean"], datasets=[{"grid": [1, 2]}])

        valid_mask_p = pixel_values[0][_WINDOW_P] != _NODATA
        assert stats_df.loc[0, "grid_b1_mean_200m"] == pytest.approx(
            pixel_values[0][_WINDOW_P][valid_mask_p].mean(), abs=1e-4
        )
        assert stats_df.loc[0, "grid_b2_mean_200m"] == pytest.approx(
            pixel_values[1][_WINDOW_P][valid_mask_p].mean(), abs=1e-4
        )

    def test_native_tile_declares_and_masks_the_catalog_nodata(
        self, tmp_path, pixel_values, catalog_nodata_raster
    ):
        """The native tile declares -9999, and its -9999 pixels are the source's nodata pixels."""
        tile_dir = _run_raster(tmp_path / "out")

        with rasterio.open(tile_dir / "P-grid.tif") as tile:
            assert tile.nodata == _NODATA
            assert tile.crs.to_epsg() == 32634
            tile_values = tile.read(1)
            tile_transform = tile.transform
        # The tile is the native grid around P: its first pixel is source
        # pixel (row 20, column 20).
        first_row = round((_RASTER_NORTH - tile_transform.f) / _PIXEL_SIZE_M)
        first_column = round((tile_transform.c - _RASTER_WEST) / _PIXEL_SIZE_M)
        assert (first_row, first_column) == (20, 20)
        source_values = pixel_values[0][
            20 : 20 + tile_values.shape[0], 20 : 20 + tile_values.shape[1]
        ]

        # Every -9999 pixel of the source is a nodata pixel of the tile, and
        # every pixel with data in the tile has the source value. The data
        # pixels are exactly the 299 valid pixels of the window.
        assert (tile_values[source_values == _NODATA] == _NODATA).all()
        data_mask = tile_values != _NODATA
        np.testing.assert_array_equal(tile_values[data_mask], source_values[data_mask])
        assert int(data_mask.sum()) == 299

        sidecar_entry = _sidecar_entry(tmp_path / "out" / "tiles" / "tiles_metadata.json")
        assert sidecar_entry["nodata"] == _NODATA
        assert sidecar_entry["nodata_source"] == "catalog"

    def test_resampled_tile_does_not_blend_nodata_pixels_into_data(
        self, tmp_path, pixel_values, catalog_nodata_raster
    ):
        """With resample_m, -9999 pixels become nodata and never enter the interpolation."""
        tile_dir = _run_raster(tmp_path / "out", resample_m=10)

        with rasterio.open(tile_dir / "P-grid.tif") as tile:
            assert tile.nodata == _NODATA
            assert tile.shape == (20, 20)
            tile_values = tile.read(1)
            source_rows, source_columns = _source_pixel_of_tile_pixels(tile.transform)

        # A tile pixel whose source pixel and its 8 neighbours are all -9999
        # has no data. A tile pixel whose 3 x 3 source neighbourhood has only
        # data has a value. Every value is in the data range 100-200: a
        # bilinear mix with -9999 would be far below it.
        source_is_nodata = pixel_values[0] == _NODATA
        neighbourhood_nodata_count = sum(
            source_is_nodata[source_rows + row_offset, source_columns + column_offset].astype(int)
            for row_offset in (-1, 0, 1)
            for column_offset in (-1, 0, 1)
        )
        tile_is_nodata = tile_values == _NODATA
        assert (neighbourhood_nodata_count == 9).any()
        assert tile_is_nodata[neighbourhood_nodata_count == 9].all()
        assert not tile_is_nodata[neighbourhood_nodata_count == 0].any()
        assert (
            (tile_values[~tile_is_nodata] >= 100.0) & (tile_values[~tile_is_nodata] <= 200.0)
        ).all()

    def test_every_read_path_matches_a_file_that_declares_the_nodata(
        self, tmp_path, pixel_values, catalog_nodata_raster
    ):
        """The catalog nodata gives the same statistics and tiles as a file that declares -9999."""
        catalog_stats = _run_tabular(tmp_path / "catalog", ["mean", "std", "count", "point"])
        catalog_native_dir = _run_raster(tmp_path / "catalog_native")
        catalog_resampled_dir = _run_raster(tmp_path / "catalog_resampled", resample_m=10)

        declared_path = _write_raster(
            tmp_path / "rasters" / "declared.tif", pixel_values, nodata=_NODATA
        )
        _register({"path": str(declared_path)})
        declared_stats = _run_tabular(tmp_path / "declared", ["mean", "std", "count", "point"])
        declared_native_dir = _run_raster(tmp_path / "declared_native")
        declared_resampled_dir = _run_raster(tmp_path / "declared_resampled", resample_m=10)

        pd.testing.assert_frame_equal(catalog_stats, declared_stats)
        for catalog_dir, declared_dir in [
            (catalog_native_dir, declared_native_dir),
            (catalog_resampled_dir, declared_resampled_dir),
        ]:
            for tile_name in ("P-grid.tif", "Q-grid.tif"):
                with (
                    rasterio.open(catalog_dir / tile_name) as catalog_tile,
                    rasterio.open(declared_dir / tile_name) as declared_tile,
                ):
                    assert catalog_tile.nodata == declared_tile.nodata == _NODATA
                    assert catalog_tile.transform == declared_tile.transform
                    np.testing.assert_array_equal(catalog_tile.read(), declared_tile.read())

        declared_entry = _sidecar_entry(tmp_path / "declared" / "table_metadata.json")
        assert declared_entry["nodata"] == _NODATA
        assert declared_entry["nodata_source"] == "file"

    def test_rotated_grid_gives_the_resampled_tile_of_a_file_that_declares_the_nodata(
        self, tmp_path, pixel_values
    ):
        """On a rotated grid, the resample_m tile equals the tile of a file that declares -9999."""
        # The grid is rotated by 20 degrees around P. The file is read through
        # a WarpedVRT that declares the catalog nodata. Without the grid of
        # the file, the VRT would resample the file to a north-up grid first,
        # so the tile would be resampled twice.
        rotated_transform = (
            Affine.translation(_POINT_P_X, _POINT_P_Y)
            * Affine.rotation(20.0)
            * Affine.translation(-_POINT_P_X, -_POINT_P_Y)
            * _RASTER_TRANSFORM
        )
        catalog_path = _write_raster(
            tmp_path / "rasters" / "rotated.tif", pixel_values, transform=rotated_transform
        )
        _register({"path": str(catalog_path), "nodata": _NODATA})
        catalog_dir = _run_raster(tmp_path / "catalog", resample_m=10)

        declared_path = _write_raster(
            tmp_path / "rasters" / "rotated_declared.tif",
            pixel_values,
            nodata=_NODATA,
            transform=rotated_transform,
        )
        _register({"path": str(declared_path)})
        declared_dir = _run_raster(tmp_path / "declared", resample_m=10)

        for tile_name in ("P-grid.tif", "Q-grid.tif"):
            with (
                rasterio.open(catalog_dir / tile_name) as catalog_tile,
                rasterio.open(declared_dir / tile_name) as declared_tile,
            ):
                assert catalog_tile.nodata == declared_tile.nodata == _NODATA
                assert catalog_tile.transform == declared_tile.transform
                np.testing.assert_allclose(catalog_tile.read(), declared_tile.read(), rtol=1e-6)

        # The tile of P has the nodata block and data around it.
        with rasterio.open(declared_dir / "P-grid.tif") as declared_tile:
            declared_values = declared_tile.read(1)
        assert (declared_values == _NODATA).any()
        assert (declared_values != _NODATA).any()


class TestCatalogNodataRules:
    """The catalog ``nodata`` must be one number that fits the bands and agrees with the file."""

    def test_differing_file_nodata_raises_and_names_both_values(self, tmp_path):
        """File nodata 0 and catalog nodata -9999 raise a ValueError that names both values."""
        raster_path = _write_raster(
            tmp_path / "rasters" / "zero.tif", _pixel_values_with_nodata_pixels(), nodata=0
        )
        _register({"path": str(raster_path), "nodata": _NODATA})

        with pytest.raises(ValueError, match=r"'nodata' to -9999, but .* nodata value 0\.0"):
            _run_tabular(tmp_path / "out", ["mean"])

    def test_equal_file_nodata_runs_and_records_the_file_as_source(self, tmp_path):
        """File nodata 0 and catalog nodata 0 run, and the sidecar names the file as source."""
        raster_path = _write_raster(
            tmp_path / "rasters" / "zero.tif", _pixel_values_with_nodata_pixels(), nodata=0
        )
        _register({"path": str(raster_path), "nodata": 0})

        stats_df = _run_tabular(tmp_path / "out", ["mean"])

        assert stats_df["grid_mean_200m"].notna().all()
        sidecar_entry = _sidecar_entry(tmp_path / "out" / "table_metadata.json")
        assert sidecar_entry["nodata"] == 0
        assert sidecar_entry["nodata_source"] == "file"

    def test_nodata_that_inspection_copies_from_an_int16_file_runs(self, tmp_path):
        """An int16 file with nodata, registered from YAML without nodata, runs (-32768.0 is filled)."""
        int16_values = np.full((1, _RASTER_SIZE, _RASTER_SIZE), 150, dtype=np.int16)
        int16_values[0, _PIXEL_P[0], _PIXEL_P[1]] = -32768
        raster_path = _write_raster(tmp_path / "rasters" / "int16.tif", int16_values, nodata=-32768)
        catalog_path = tmp_path / "catalog.yml"
        catalog_path.write_text(
            yaml.safe_dump(
                {"datasets": {"grid": {"data_source": "local", "path": str(raster_path)}}}
            ),
            encoding="utf-8",
        )
        # Raster inspection copies the file's nodata (a float) into the entry.
        assert load_catalog(catalog_path)["datasets"]["grid"]["nodata"] == -32768.0
        update_catalog(str(catalog_path))

        stats_df = _run_tabular(tmp_path / "out", ["mean", "point"])

        assert pd.isna(stats_df.loc[0, "grid_point"])
        assert stats_df["grid_mean_200m"].tolist() == [150.0, 150.0]
        sidecar_entry = _sidecar_entry(tmp_path / "out" / "table_metadata.json")
        assert sidecar_entry["nodata"] == -32768
        assert sidecar_entry["nodata_source"] == "file"

    def test_list_nodata_raises(self, tmp_path):
        """A list of nodata values raises a ValueError."""
        raster_path = _write_raster(
            tmp_path / "rasters" / "grid.tif", _pixel_values_with_nodata_pixels()
        )
        _register({"path": str(raster_path), "nodata": [_NODATA, _NODATA]})

        with pytest.raises(ValueError, match="'nodata' to a list"):
            _run_tabular(tmp_path / "out", ["mean"])

    @pytest.mark.parametrize(
        ("dtype", "catalog_nodata"),
        [
            ("uint8", -9999),
            ("uint8", 256),
            ("int16", 1.5),
            ("int16", math.nan),
            # Outside the float32 range: the band would store -infinity.
            ("float32", -3.5e38),
        ],
    )
    def test_catalog_nodata_outside_the_band_type_raises(self, tmp_path, dtype, catalog_nodata):
        """A catalog nodata that the band type cannot store raises, for a file without nodata."""
        integer_values = np.full((1, _RASTER_SIZE, _RASTER_SIZE), 7, dtype=dtype)
        raster_path = _write_raster(tmp_path / "rasters" / "integer.tif", integer_values)
        _register({"path": str(raster_path), "nodata": catalog_nodata})

        with pytest.raises(ValueError, match=f"data type {dtype}, which cannot store"):
            _run_tabular(tmp_path / "out", ["mean"])

    def test_integer_valued_float_nodata_fits_an_integer_band(self, tmp_path):
        """A float nodata without a fractional part (-9999.0) is used for an int16 band."""
        int16_values = np.full((1, _RASTER_SIZE, _RASTER_SIZE), 150, dtype=np.int16)
        int16_values[0, _PIXEL_P[0], _PIXEL_P[1]] = _NODATA
        raster_path = _write_raster(tmp_path / "rasters" / "int16.tif", int16_values)
        _register({"path": str(raster_path), "nodata": -9999.0})

        stats_df = _run_tabular(tmp_path / "out", ["min", "point"])

        assert pd.isna(stats_df.loc[0, "grid_point"])
        assert stats_df["grid_min_200m"].tolist() == [150.0, 150.0]

    def test_nan_nodata_on_a_float_file_without_nodata(self, tmp_path):
        """A NaN catalog nodata runs on a float file without nodata; the sidecar records "nan"."""
        float_values = _pixel_values_with_nodata_pixels()
        float_values[float_values == _NODATA] = np.nan
        raster_path = _write_raster(tmp_path / "rasters" / "nan.tif", float_values)
        _register({"path": str(raster_path), "nodata": float("nan")})

        stats_df = _run_tabular(tmp_path / "out", ["count", "point"])
        tile_dir = _run_raster(tmp_path / "tiles_out")

        assert stats_df["grid_count_200m"].tolist() == [299, 394]
        assert pd.isna(stats_df.loc[0, "grid_point"])
        sidecar_entry = _sidecar_entry(tmp_path / "out" / "table_metadata.json")
        assert sidecar_entry["nodata"] == "nan"
        assert sidecar_entry["nodata_source"] == "catalog"
        with rasterio.open(tile_dir / "P-grid.tif") as tile:
            assert math.isnan(tile.nodata)

    def test_printed_float32_minimum_is_accepted_and_masks_its_pixels(self, tmp_path):
        """Catalog nodata -3.4028235e38, the printed float32 minimum, is accepted on float32."""
        # -3.4028235e38 is a little below the exact float32 minimum, but
        # float32 rounds it to that minimum.
        float32_minimum = float(np.finfo(np.float32).min)
        assert -3.4028235e38 < float32_minimum
        pixel_values = _pixel_values_with_nodata_pixels()
        pixel_values[pixel_values == _NODATA] = np.finfo(np.float32).min
        raster_path = _write_raster(tmp_path / "rasters" / "float32_minimum.tif", pixel_values)
        _register({"path": str(raster_path), "nodata": -3.4028235e38})

        stats_df = _run_tabular(tmp_path / "out", ["count", "point"])

        assert pd.isna(stats_df.loc[0, "grid_point"])
        assert stats_df["grid_count_200m"].tolist() == [299, 394]
        sidecar_entry = _sidecar_entry(tmp_path / "out" / "table_metadata.json")
        assert sidecar_entry["nodata"] == float32_minimum

    def test_float32_error_names_the_exact_limits(self, tmp_path):
        """-3.5e38 does not fit float32; the error names the limits at full precision."""
        float32_values = np.full((1, _RASTER_SIZE, _RASTER_SIZE), 7, dtype=np.float32)
        raster_path = _write_raster(tmp_path / "rasters" / "float32.tif", float32_values)
        _register({"path": str(raster_path), "nodata": -3.5e38})

        float32_maximum = float(np.finfo(np.float32).max)
        assert repr(float32_maximum) == "3.4028234663852886e+38"
        expected_text = f"Use NaN or a number from {-float32_maximum!r} to {float32_maximum!r}."
        with pytest.raises(ValueError, match=re.escape(expected_text)):
            _run_tabular(tmp_path / "out", ["mean"])

    def test_float32_band_gets_the_nodata_value_that_it_can_store(self, tmp_path):
        """Catalog nodata -3.4e38 on a float32 file: every read path masks the float32 value."""
        # float32 cannot store -3.4e38 exactly. The nodata pixels of the file
        # hold the nearest float32 number, which envoi uses and records.
        float32_nodata = float(np.float32(-3.4e38))
        assert float32_nodata != -3.4e38
        pixel_values = _pixel_values_with_nodata_pixels()
        pixel_values[pixel_values == _NODATA] = np.float32(-3.4e38)
        raster_path = _write_raster(tmp_path / "rasters" / "float32.tif", pixel_values)
        _register({"path": str(raster_path), "nodata": -3.4e38})

        stats_df = _run_tabular(tmp_path / "out", ["count", "point"])
        native_dir = _run_raster(tmp_path / "native")
        resampled_dir = _run_raster(tmp_path / "resampled", resample_m=10)

        # Point sample and window statistics: the nodata pixels are not data.
        assert pd.isna(stats_df.loc[0, "grid_point"])
        assert stats_df["grid_count_200m"].tolist() == [299, 394]
        sidecar_entry = _sidecar_entry(tmp_path / "out" / "table_metadata.json")
        assert sidecar_entry["nodata"] == float32_nodata
        assert sidecar_entry["nodata_source"] == "catalog"

        # Native tile: declares the float32 value, and has the 299 valid pixels.
        with rasterio.open(native_dir / "P-grid.tif") as tile:
            assert tile.nodata == float32_nodata
            native_values = tile.read(1)
        assert int((native_values != np.float32(float32_nodata)).sum()) == 299

        # Resampled tile: nodata pixels, and data only from the range 100-200
        # (a bilinear mix with -3.4e38 would be far below it).
        with rasterio.open(resampled_dir / "P-grid.tif") as tile:
            assert tile.nodata == float32_nodata
            resampled_values = tile.read(1)
        resampled_is_nodata = resampled_values == np.float32(float32_nodata)
        assert resampled_is_nodata.any()
        resampled_data = resampled_values[~resampled_is_nodata]
        assert ((resampled_data >= 100.0) & (resampled_data <= 200.0)).all()

    def test_float32_file_nodata_is_used_as_the_band_stores_it(self, tmp_path):
        """A float32 file that declares nodata -3.4e38: envoi uses and records the float32 value."""
        float32_nodata = float(np.float32(-3.4e38))
        pixel_values = _pixel_values_with_nodata_pixels()
        pixel_values[pixel_values == _NODATA] = np.float32(-3.4e38)
        tiff_path = _write_raster(tmp_path / "rasters" / "float32.tif", pixel_values)
        # GDAL's GeoTIFF driver writes the nodata value of a float32 band
        # rounded to float32. A VRT keeps the declared text, so the file
        # declares -3.4e38 exactly, as files from other tools can.
        geo_transform = ", ".join(repr(value) for value in _RASTER_TRANSFORM.to_gdal())
        raster_path = tmp_path / "rasters" / "float32_declared.vrt"
        raster_path.write_text(
            f"""<VRTDataset rasterXSize="{_RASTER_SIZE}" rasterYSize="{_RASTER_SIZE}">
  <SRS>{_RASTER_CRS}</SRS>
  <GeoTransform>{geo_transform}</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <NoDataValue>-3.4e38</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{tiff_path.name}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
""",
            encoding="utf-8",
        )
        with rasterio.open(raster_path) as src:
            assert src.nodata == -3.4e38
        _register({"path": str(raster_path)})

        stats_df = _run_tabular(tmp_path / "out", ["count", "point"])
        native_dir = _run_raster(tmp_path / "native")

        # Point sample and window statistics: the nodata pixels are not data.
        assert pd.isna(stats_df.loc[0, "grid_point"])
        assert stats_df.loc[1, "grid_point"] == pytest.approx(pixel_values[0][_PIXEL_Q], abs=1e-4)
        assert stats_df["grid_count_200m"].tolist() == [299, 394]
        sidecar_entry = _sidecar_entry(tmp_path / "out" / "table_metadata.json")
        assert sidecar_entry["nodata"] == float32_nodata
        assert sidecar_entry["nodata_source"] == "file"

        # The native tile declares the float32 value and has the 299 valid pixels.
        with rasterio.open(native_dir / "P-grid.tif") as tile:
            assert tile.nodata == float32_nodata
            native_values = tile.read(1)
        assert int((native_values != np.float32(float32_nodata)).sum()) == 299


# ------------------------------------------------------------------
# Catalog CRS
# ------------------------------------------------------------------


class TestCatalogCrs:
    """A catalog ``crs`` is used for a file without a CRS and must agree with a file's CRS."""

    @pytest.fixture
    def pixel_values(self) -> np.ndarray:
        random_generator = np.random.default_rng(seed=62)
        return random_generator.uniform(100.0, 200.0, (1, _RASTER_SIZE, _RASTER_SIZE)).astype(
            np.float32
        )

    def test_file_without_crs_uses_the_catalog_crs_in_every_read_path(self, tmp_path, pixel_values):
        """Statistics and tiles of a file without CRS equal those of the same file with its CRS."""
        reference_path = _write_raster(tmp_path / "rasters" / "with_crs.tif", pixel_values)
        _register({"path": str(reference_path)})
        reference_stats = _run_tabular(tmp_path / "reference", ["mean", "max", "point"])
        reference_native_dir = _run_raster(tmp_path / "reference_native")
        reference_resampled_dir = _run_raster(tmp_path / "reference_resampled", resample_m=10)

        no_crs_path = _write_raster(tmp_path / "rasters" / "no_crs.tif", pixel_values, crs=None)
        with rasterio.open(no_crs_path) as src:
            assert not src.crs
        _register({"path": str(no_crs_path), "crs": _RASTER_CRS})
        stats = _run_tabular(tmp_path / "catalog", ["mean", "max", "point"])
        native_dir = _run_raster(tmp_path / "catalog_native")
        resampled_dir = _run_raster(tmp_path / "catalog_resampled", resample_m=10)

        # Point sample and window statistics: the values of the reference,
        # which are the source pixels.
        pd.testing.assert_frame_equal(stats, reference_stats)
        assert stats["grid_point"].tolist() == pytest.approx(
            [pixel_values[0][_PIXEL_P], pixel_values[0][_PIXEL_Q]], abs=1e-4
        )
        assert stats.loc[0, "grid_mean_200m"] == pytest.approx(
            pixel_values[0][_WINDOW_P].mean(), abs=1e-4
        )

        # Native tiles: in the catalog CRS, with the reference pixels.
        # Resampled tiles: in the UTM zone of the point, with the pixel
        # values of the reference tile.
        for tile_dir, reference_dir in [
            (native_dir, reference_native_dir),
            (resampled_dir, reference_resampled_dir),
        ]:
            for tile_name in ("P-grid.tif", "Q-grid.tif"):
                with (
                    rasterio.open(tile_dir / tile_name) as tile,
                    rasterio.open(reference_dir / tile_name) as reference_tile,
                ):
                    assert tile.crs.to_epsg() == 32634
                    assert tile.transform == reference_tile.transform
                    np.testing.assert_array_equal(tile.read(), reference_tile.read())

        # The resampled tile of P has data in every pixel, from the source.
        with rasterio.open(resampled_dir / "P-grid.tif") as tile:
            resampled_values = tile.read(1)
        assert np.isfinite(resampled_values).all()
        assert ((resampled_values >= 100.0) & (resampled_values <= 200.0)).all()

        sidecar_entry = _sidecar_entry(tmp_path / "catalog" / "table_metadata.json")
        assert sidecar_entry["native_crs"] == _RASTER_CRS
        assert sidecar_entry["native_spatial_resolution_m"] == _PIXEL_SIZE_M

    def test_file_without_crs_and_without_catalog_crs_raises(self, tmp_path, pixel_values):
        """A file without CRS needs a catalog crs."""
        raster_path = _write_raster(tmp_path / "rasters" / "no_crs.tif", pixel_values, crs=None)
        _register({"path": str(raster_path)})

        with pytest.raises(ValueError, match="has no CRS, and the catalog entry has no 'crs'"):
            _run_tabular(tmp_path / "out", ["mean"])

    def test_differing_catalog_crs_raises(self, tmp_path, pixel_values):
        """A catalog crs that differs from the file's CRS raises and names both."""
        raster_path = _write_raster(tmp_path / "rasters" / "grid.tif", pixel_values)
        _register({"path": str(raster_path), "crs": "EPSG:32633"})

        with pytest.raises(
            ValueError, match=r"'crs' to 'EPSG:32633', but the file .* has the CRS EPSG:32634"
        ):
            _run_tabular(tmp_path / "out", ["mean"])

    @pytest.mark.parametrize(
        "file_crs",
        [
            _RASTER_CRS,
            # UTM zone 34N with a false easting 10 m larger: a CRS without an
            # EPSG code, so raster inspection writes its WKT.
            "+proj=tmerc +lat_0=0 +lon_0=21 +k=0.9996 +x_0=500010 +y_0=0 +datum=WGS84 +units=m",
        ],
    )
    def test_crs_that_inspection_copies_from_the_file_runs(self, tmp_path, pixel_values, file_crs):
        """The crs that YAML registration fills in from the file never raises."""
        raster_path = _write_raster(tmp_path / "rasters" / "grid.tif", pixel_values, crs=file_crs)
        catalog_path = tmp_path / "catalog.yml"
        catalog_path.write_text(
            yaml.safe_dump(
                {"datasets": {"grid": {"data_source": "local", "path": str(raster_path)}}}
            ),
            encoding="utf-8",
        )
        filled_crs = load_catalog(catalog_path)["datasets"]["grid"]["crs"]
        if file_crs == _RASTER_CRS:
            assert filled_crs == _RASTER_CRS
        else:
            assert not filled_crs.startswith("EPSG:")
        update_catalog(str(catalog_path))

        stats_df = _run_tabular(tmp_path / "out", ["mean"])

        assert stats_df["grid_mean_200m"].notna().all()

    def test_crs_with_only_a_non_epsg_authority_code_runs(self, tmp_path, pixel_values):
        """A CRS with an ESRI code and no EPSG code, filled in by YAML registration, never raises."""
        esri_code = "ESRI:102013"  # Europe Albers Equal Area Conic
        try:
            esri_crs = CRS.from_user_input(esri_code)
        except CRSError:
            pytest.skip(f"The installed PROJ database has no {esri_code}.")
        assert esri_crs.to_epsg() is None
        # The test grid around P, in the ESRI CRS.
        point_x, point_y = Transformer.from_crs("EPSG:4326", esri_code, always_xy=True).transform(
            _POINT_P_LON, _POINT_P_LAT
        )
        raster_path = _write_raster(
            tmp_path / "rasters" / "esri.tif",
            pixel_values,
            crs=esri_code,
            transform=from_origin(point_x - 302.5, point_y + 302.5, _PIXEL_SIZE_M, _PIXEL_SIZE_M),
        )
        catalog_path = tmp_path / "catalog.yml"
        catalog_path.write_text(
            yaml.safe_dump(
                {"datasets": {"grid": {"data_source": "local", "path": str(raster_path)}}}
            ),
            encoding="utf-8",
        )
        filled_crs = load_catalog(catalog_path)["datasets"]["grid"]["crs"]
        if filled_crs != esri_code:
            pytest.skip(
                f"GDAL did not keep the code {esri_code} in the file (read {filled_crs!r})."
            )
        update_catalog(str(catalog_path))

        stats_df = _run_tabular(tmp_path / "out", ["mean"])

        assert stats_df["grid_mean_200m"].notna().all()
        sidecar_entry = _sidecar_entry(tmp_path / "out" / "table_metadata.json")
        assert sidecar_entry["native_crs"] == esri_code


# ------------------------------------------------------------------
# When the checks run
# ------------------------------------------------------------------


class TestSettingsCheckedBeforeTheFirstAdapter:
    """The settings of every local dataset are checked before any output is written."""

    def test_conflict_in_a_later_run_config_stops_the_call_before_any_output(self, tmp_path):
        """A conflicting nodata in the second run config raises before the first one runs."""
        pixel_values = _pixel_values_with_nodata_pixels()
        good_path = _write_raster(tmp_path / "rasters" / "good.tif", pixel_values)
        conflict_path = _write_raster(tmp_path / "rasters" / "conflict.tif", pixel_values, nodata=0)
        update_catalog(
            {
                "datasets": {
                    "good": {"data_source": "local", "path": str(good_path)},
                    "conflict": {"data_source": "local", "path": str(conflict_path), "nodata": 1},
                }
            }
        )
        output_dir = tmp_path / "out"

        with pytest.raises(ValueError, match="Output 'second': dataset 'conflict'"):
            extract(
                _POINTS,
                [
                    {
                        "batch_id": "first",
                        "datasets": ["good"],
                        "settings": {"output_type": "tabular", "statistics": ["mean"]},
                    },
                    {
                        "batch_id": "second",
                        "datasets": ["conflict"],
                        "settings": {"output_type": "tabular", "statistics": ["mean"]},
                    },
                ],
                output_dir=output_dir,
            )
        assert not output_dir.exists()

    def test_band_override_outside_the_file_raises_before_any_output(self, tmp_path):
        """A per-call band that the file does not have raises a ValueError before any output."""
        raster_path = _write_raster(
            tmp_path / "rasters" / "grid.tif", _pixel_values_with_nodata_pixels()
        )
        _register({"path": str(raster_path)})

        with pytest.raises(ValueError, match=r"band\(s\) \[2\] are not in the file"):
            _run_tabular(tmp_path / "out", ["mean"], datasets=[{"grid": [1, 2]}])
        assert not (tmp_path / "out").exists()

    def test_adapter_closes_the_file_when_the_check_fails(self, tmp_path, monkeypatch):
        """LocalRasterAdapter raises the same error and closes every file it opened."""
        raster_path = _write_raster(
            tmp_path / "rasters" / "zero.tif", _pixel_values_with_nodata_pixels(), nodata=0
        )
        opened_datasets = []
        original_open = rasterio.open

        def recording_open(*args, **kwargs):
            dataset = original_open(*args, **kwargs)
            opened_datasets.append(dataset)
            return dataset

        monkeypatch.setattr(rasterio, "open", recording_open)

        with pytest.raises(ValueError, match=r"'nodata' to -9999, but .* nodata value 0\.0"):
            LocalRasterAdapter({"path": str(raster_path), "nodata": _NODATA})
        assert opened_datasets
        assert all(dataset.closed for dataset in opened_datasets)
