"""Offline tests of the Earth Engine adapter.

These tests never contact Earth Engine, so they need no credentials and run on
every machine. Two rules keep them offline and meaningful:

* An autouse fixture makes the Earth Engine network calls raise. Creating an
  ``ee.Geometry`` or ``ee.ImageCollection`` loads the API signatures from the
  server (``ee.data.getAlgorithms``), and ``getInfo()`` sends a compute request
  (``ee.data.computeValue``). An offline test that reaches one of them fails.
* ``_make_offline_adapter`` builds a ``GeeRasterAdapter`` without
  ``__post_init__``, which initializes Earth Engine and inspects the asset.
  Each test sets only the attributes that the method under test reads.

When the code under test builds Earth Engine objects, the test replaces the
``ee`` names that the code uses with a stub, and asserts on values that only
the stub can give. A test then cannot pass because of an unrelated ``ee`` error.
"""

from __future__ import annotations

import importlib
import json
import math
from types import SimpleNamespace

import ee
import pandas as pd
import pytest

from envoi import update_catalog
from envoi._output_assembly import _append_stat_columns
from envoi.adapters.earth_engine import _image
from envoi.adapters.earth_engine import adapter as adapter_module
from envoi.adapters.earth_engine._image import (
    _get_collection_time_bounds,
    _resolve_date_filter_range,
)
from envoi.adapters.earth_engine.adapter import GeeRasterAdapter
from envoi.extract import extract
from envoi.metadata import summarize_date_info


@pytest.fixture(autouse=True)
def _block_earth_engine_network(monkeypatch):
    """Make the Earth Engine network calls raise during every test in this file."""

    def refuse_network_call(*args, **kwargs):
        raise AssertionError("An offline test made an Earth Engine network call.")

    for function_name in ("getAlgorithms", "computeValue", "getInfo", "getAsset"):
        monkeypatch.setattr(ee.data, function_name, refuse_network_call)


def _make_offline_adapter(**attributes) -> GeeRasterAdapter:
    """Return a ``GeeRasterAdapter`` without ``__post_init__``, with only the given attributes."""
    adapter = object.__new__(GeeRasterAdapter)
    for attribute_name, value in attributes.items():
        setattr(adapter, attribute_name, value)
    return adapter


def _epoch_ms(timestamp_text: str) -> int:
    """Return a timestamp as milliseconds since 1970-01-01 (the ``system:time_*`` unit)."""
    elapsed = pd.Timestamp(timestamp_text) - pd.Timestamp("1970-01-01")
    return elapsed // pd.Timedelta(milliseconds=1)


# ------------------------------------------------------------------
# _get_collection_time_bounds — the collection time index
# ------------------------------------------------------------------


class _StubImageCollection:
    """Stands in for ``ee.ImageCollection``. ``aggregate_array`` returns fixed values."""

    def __init__(self, property_values: dict[str, list]):
        self._property_values = property_values

    def aggregate_array(self, property_name: str) -> list:
        return self._property_values[property_name]


class _StubDictionary:
    """Stands in for ``ee.Dictionary``. ``getInfo`` returns the wrapped values."""

    def __init__(self, values: dict):
        self._values = values

    def getInfo(self) -> dict:
        return dict(self._values)


def _stub_collection_timestamps(monkeypatch, starts: list, ends: list) -> list[str]:
    """Replace ``ee`` in ``_image`` so the timestamp request returns ``starts`` and ``ends``.

    Returns a list that records the collection IDs that the code under test opened.
    """
    opened_collection_ids: list[str] = []

    def open_collection(collection_id):
        opened_collection_ids.append(collection_id)
        return _StubImageCollection({"system:time_start": starts, "system:time_end": ends})

    stub_ee = SimpleNamespace(ImageCollection=open_collection, Dictionary=_StubDictionary)
    monkeypatch.setattr(_image, "ee", stub_ee)
    return opened_collection_ids


class TestCollectionTimeBoundsDuplicateStarts:
    """The time index never has duplicate start times (a duplicate breaks the image lookup)."""

    def test_duplicate_start_keeps_the_latest_end(self, monkeypatch):
        """Two images with one start and different ends give one index row with the later end."""
        # Unsorted, as Earth Engine can return them.
        starts = [_epoch_ms("2021-01-17"), _epoch_ms("2021-01-01"), _epoch_ms("2021-01-01")]
        ends = [_epoch_ms("2021-01-25"), _epoch_ms("2021-01-17"), _epoch_ms("2021-01-09")]
        opened_collection_ids = _stub_collection_timestamps(monkeypatch, starts, ends)

        start_index, end_index = _get_collection_time_bounds("users/test/collection")

        assert opened_collection_ids == ["users/test/collection"]
        assert start_index is not None and end_index is not None
        assert start_index.is_unique
        assert list(start_index) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-17")]
        # Aligned with the starts: the end of each start, at the same position.
        assert list(end_index) == [pd.Timestamp("2021-01-17"), pd.Timestamp("2021-01-25")]

    def test_index_with_duplicate_start_supports_the_nearest_lookup(self, monkeypatch):
        """The nearest-image lookup works on the index built from a duplicate start."""
        starts = [_epoch_ms("2021-01-01"), _epoch_ms("2021-01-01"), _epoch_ms("2021-02-01")]
        ends = [_epoch_ms("2021-01-02"), _epoch_ms("2021-01-03"), _epoch_ms("2021-02-02")]
        _stub_collection_timestamps(monkeypatch, starts, ends)
        start_index, end_index = _get_collection_time_bounds("users/test/collection")

        adapter = _make_offline_adapter(
            _collection_time_starts=start_index,
            _collection_time_ends=end_index,
            _dataset_spec={"collection": "users/test/collection"},
        )
        assert adapter._resolve_date_info("2021-01-05") == {
            "date_clamped": False,
            "date_source": "nearest_to_sample",
            "image_time_start": "2021-01-01",
            "image_time_end": "2021-01-03",
        }

    def test_duplicate_start_without_end_times_is_unique(self, monkeypatch):
        """Without a complete set of end times, the start index is unique and ends are None."""
        starts = [_epoch_ms("2021-01-01"), _epoch_ms("2021-01-01"), _epoch_ms("2021-02-01")]
        ends = [_epoch_ms("2021-01-09")]
        _stub_collection_timestamps(monkeypatch, starts, ends)

        start_index, end_index = _get_collection_time_bounds("users/test/collection")

        assert end_index is None
        assert list(start_index) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01")]


class TestCollectionTimeBoundsMissingValues:
    """Each start keeps the end of its own image when some timestamps are missing."""

    def test_missing_start_and_missing_end_are_not_mispaired(self, monkeypatch):
        """One image without a start and another without an end give a start-only index."""
        # Dropping the None values first would pair 2021-01-17 with 2021-01-09
        # (the end of the image without a start).
        starts = [_epoch_ms("2021-01-01"), None, _epoch_ms("2021-01-17")]
        ends = [_epoch_ms("2021-01-02"), _epoch_ms("2021-01-09"), None]
        _stub_collection_timestamps(monkeypatch, starts, ends)

        start_index, end_index = _get_collection_time_bounds("users/test/collection")

        assert end_index is None
        assert list(start_index) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-17")]

    def test_image_without_a_start_is_dropped_with_its_end(self, monkeypatch):
        """An image without a start is dropped together with its end."""
        starts = [_epoch_ms("2021-01-01"), None, _epoch_ms("2021-01-17")]
        ends = [_epoch_ms("2021-01-02"), _epoch_ms("2021-01-09"), _epoch_ms("2021-01-18")]
        _stub_collection_timestamps(monkeypatch, starts, ends)

        start_index, end_index = _get_collection_time_bounds("users/test/collection")

        assert list(start_index) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-17")]
        assert list(end_index) == [pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-18")]


# ------------------------------------------------------------------
# _resolve_date_info — date metadata for "contains" collections
# ------------------------------------------------------------------

# Three 8-day intervals, the shape of the built-in "contains" collections.
_EIGHT_DAY_STARTS = pd.DatetimeIndex(["2021-01-01", "2021-01-09", "2021-01-17"])
_EIGHT_DAY_ENDS = pd.DatetimeIndex(["2021-01-09", "2021-01-17", "2021-01-25"])


def _contains_adapter(time_starts, time_ends) -> GeeRasterAdapter:
    """Return an offline adapter for a ``contains`` collection with the given time index."""
    return _make_offline_adapter(
        _collection_time_starts=time_starts,
        _collection_time_ends=time_ends,
        _dataset_spec={
            "collection": "users/test/collection",
            "collection_date_policy": "contains",
        },
    )


class TestResolveDateInfoContains:
    """For ``contains``, a date is clamped only when it is before the first image
    start, or at or after the end of the last image interval.
    """

    @pytest.mark.parametrize(
        ("sample_date", "clamped", "image_start", "image_end"),
        [
            ("2020-12-31", True, "2021-01-01", "2021-01-09"),  # before the first start
            ("2021-01-01", False, "2021-01-01", "2021-01-09"),  # on the first start
            ("2021-01-12", False, "2021-01-09", "2021-01-17"),  # in range
            ("2021-01-17", False, "2021-01-17", "2021-01-25"),  # on the last start
            ("2021-01-24", False, "2021-01-17", "2021-01-25"),  # inside the last interval
            ("2021-01-25", True, "2021-01-17", "2021-01-25"),  # on the last end
            ("2022-06-01", True, "2021-01-17", "2021-01-25"),  # after the last interval
        ],
    )
    def test_with_end_times(self, sample_date, clamped, image_start, image_end):
        """With end times, the last interval ends at the last image's end time."""
        adapter = _contains_adapter(_EIGHT_DAY_STARTS, _EIGHT_DAY_ENDS)

        date_info = adapter._resolve_date_info(sample_date)

        assert date_info == {
            "date_clamped": clamped,
            "date_source": "clamped_to_nearest" if clamped else "contains_sample_date",
            "image_time_start": image_start,
            "image_time_end": image_end,
        }
        # The reported image is the image that the filterDate range selects.
        assert _resolve_date_filter_range(
            pd.Timestamp(sample_date), "contains", _EIGHT_DAY_STARTS, _EIGHT_DAY_ENDS
        ) == (image_start, image_end)

    @pytest.mark.parametrize(
        ("sample_date", "clamped", "image_start"),
        [
            ("2020-12-31", True, "2021-01-01"),  # before the first start
            ("2021-01-01", False, "2021-01-01"),  # on the first start
            ("2021-01-12", False, "2021-01-09"),  # in range
            ("2021-01-17", False, "2021-01-17"),  # inside the last interval (one day)
            ("2021-01-18", True, "2021-01-17"),  # on the last end (last start + 1 day)
            ("2022-06-01", True, "2021-01-17"),  # after the last interval
        ],
    )
    def test_without_end_times(self, sample_date, clamped, image_start):
        """Without end times, the last interval ends one day after the last start."""
        adapter = _contains_adapter(_EIGHT_DAY_STARTS, None)

        date_info = adapter._resolve_date_info(sample_date)

        assert date_info == {
            "date_clamped": clamped,
            "date_source": "clamped_to_nearest" if clamped else "contains_sample_date",
            "image_time_start": image_start,
        }
        assert (
            _resolve_date_filter_range(pd.Timestamp(sample_date), "contains", _EIGHT_DAY_STARTS)[0]
            == image_start
        )

    def test_end_equal_to_start(self):
        """With instantaneous images (end == start), a date on an image's start day is not clamped."""
        time_starts = pd.DatetimeIndex(["2021-06-01", "2021-06-15"])
        adapter = _contains_adapter(time_starts, time_starts.copy())

        assert adapter._resolve_date_info("2021-06-01") == {
            "date_clamped": False,
            "date_source": "contains_sample_date",
            "image_time_start": "2021-06-01",
            "image_time_end": "2021-06-01",
        }
        # The filterDate range covers the start day of the last image, so this
        # date selects that image, and is not clamped.
        assert adapter._resolve_date_info("2021-06-15") == {
            "date_clamped": False,
            "date_source": "contains_sample_date",
            "image_time_start": "2021-06-15",
            "image_time_end": "2021-06-15",
        }
        assert _resolve_date_filter_range(
            pd.Timestamp("2021-06-15"), "contains", time_starts, time_starts.copy()
        ) == ("2021-06-15", "2021-06-16")
        # The next day is after the last interval.
        assert adapter._resolve_date_info("2021-06-16") == {
            "date_clamped": True,
            "date_source": "clamped_to_nearest",
            "image_time_start": "2021-06-15",
            "image_time_end": "2021-06-15",
        }

    def test_last_image_ends_later_on_its_start_day(self):
        """A last image that ends during its start day covers the whole start day."""
        time_starts = pd.DatetimeIndex(["2021-06-01", "2021-06-15 00:00:00"])
        time_ends = pd.DatetimeIndex(["2021-06-15", "2021-06-15 06:00:00"])
        adapter = _contains_adapter(time_starts, time_ends)

        assert adapter._resolve_date_info("2021-06-15")["date_clamped"] is False
        assert adapter._resolve_date_info("2021-06-16")["date_clamped"] is True

    def test_one_day_intervals(self):
        """With one-day intervals, a date on the last day is not clamped and the next day is."""
        time_starts = pd.DatetimeIndex(["2021-06-14", "2021-06-15"])
        time_ends = pd.DatetimeIndex(["2021-06-15", "2021-06-16"])
        adapter = _contains_adapter(time_starts, time_ends)

        assert adapter._resolve_date_info("2021-06-15")["date_clamped"] is False
        assert adapter._resolve_date_info("2021-06-16")["date_clamped"] is True


# ------------------------------------------------------------------
# summarize_date_info — the sidecar date_info summary
# ------------------------------------------------------------------


class TestSummarizeDateInfo:
    """The sidecar ``date_info`` uses every point that has date information."""

    def test_first_point_without_date_keys(self):
        """A first meta dict without date keys (a failed point) does not remove the summary."""
        meta_list = [
            {"in_extent": False, "n_pixels": 0},
            {
                "date_clamped": False,
                "date_source": "contains_sample_date",
                "image_time_start": "2021-01-09",
                "image_time_end": "2021-01-17",
            },
        ]

        assert summarize_date_info(meta_list) == {
            "n_nearest_to_sample": 0,
            "n_contains_sample_date": 1,
            "n_clamped_to_nearest": 0,
            "n_most_recent_no_date": 0,
            "image_date_ranges_used": [["2021-01-09", "2021-01-17"]],
        }

    def test_counts_every_date_source(self):
        """Each date source has its own count, including ``contains_sample_date``."""
        sources = [
            "contains_sample_date",
            "contains_sample_date",
            "clamped_to_nearest",
            "nearest_to_sample",
            "most_recent_no_date",
        ]
        meta_list = [
            {"date_source": source, "date_clamped": False, "image_time_start": "2021-01-01"}
            for source in sources
        ]

        date_info = summarize_date_info(meta_list)

        assert date_info["n_contains_sample_date"] == 2
        assert date_info["n_clamped_to_nearest"] == 1
        assert date_info["n_nearest_to_sample"] == 1
        assert date_info["n_most_recent_no_date"] == 1
        assert date_info["image_date_ranges_used"] == [["2021-01-01", "2021-01-01"]]

    def test_no_point_with_date_keys(self):
        """Without date information on any point, there is no summary."""
        assert summarize_date_info([{"in_extent": True}, {"in_extent": False}]) is None
        assert summarize_date_info([]) is None


# ------------------------------------------------------------------
# Errors that affect every point stop the run
# ------------------------------------------------------------------


class _StubEarthEngineError(Exception):
    """An error that only the stubs in this file raise, as a failed Earth Engine request."""


class _StubImage:
    """Stands in for an ``ee.Image``. Its band-name request fails ``failures`` times first."""

    def __init__(self, band_names: list[str], failures: int = 0):
        self._band_names = band_names
        self._failures_left = failures
        self.n_band_name_requests = 0

    def bandNames(self) -> SimpleNamespace:
        return SimpleNamespace(getInfo=self._request_band_names)

    def _request_band_names(self) -> list[str]:
        self.n_band_name_requests += 1
        if self._failures_left > 0:
            self._failures_left -= 1
            raise _StubEarthEngineError("stub: band-name request failed")
        return list(self._band_names)


def _tile_export_adapter(**attributes) -> GeeRasterAdapter:
    """Return an offline adapter for a two-band image asset at 10 m.

    It is ready for ``export_tiles`` and, when the test stubs the per-point
    request, for ``fetch_stats_batch``.
    """
    adapter_attributes = {
        "_dataset_spec": {"image": "users/test/image", "bands": ["B1", "B2"]},
        "_static_image": _StubImage(["B1", "B2"]),
        "_is_collection": False,
        "_timestamps_loaded": True,
        "_needs_per_point_date": False,
        "_collection_time_starts": None,
        "_collection_time_ends": None,
        "crs": "EPSG:4326",
        "scale": 10,
        "max_workers": 2,
    }
    adapter_attributes.update(attributes)
    return _make_offline_adapter(**adapter_attributes)


def _record_tile_exports(adapter: GeeRasterAdapter) -> list[tuple[float, float]]:
    """Replace the per-point tile export of ``adapter`` with a recorder.

    Returns the list that records the ``(lat, lon)`` of each per-point export.
    """
    exported_points: list[tuple[float, float]] = []

    def record_export(lat, lon, window_m, output_path, date=None, resample_m=None):
        exported_points.append((lat, lon))
        return output_path

    adapter._export_single = record_export
    return exported_points


def _export_two_tiles(adapter: GeeRasterAdapter, output_dir, window_m: int, resample_m=None):
    """Call ``export_tiles`` for two points near Uppsala, with the dataset name ``test_dataset``."""
    return adapter.export_tiles(
        [59.85, 59.86],
        [17.63, 17.64],
        window_m,
        output_dir,
        ids=["a", "b"],
        dataset_name="test_dataset",
        resample_m=resample_m,
        disable_progress=True,
    )


class TestTileSizeCheck:
    """``export_tiles`` checks the tile size once, before any per-point export."""

    @pytest.mark.parametrize(
        ("window_m", "resample_m", "scale_text"),
        [
            (50_000, None, "currently 10.0"),  # 5000 x 5000 pixels at the native 10 m
            (2_000, 0.5, "currently 0.5"),  # 4000 x 4000 pixels at resample_m
        ],
    )
    def test_oversized_request_stops_before_any_export(
        self, tmp_path, window_m, resample_m, scale_text
    ):
        """An oversized tile request raises the size error, and no per-point export runs."""
        adapter = _tile_export_adapter()
        exported_points = _record_tile_exports(adapter)

        with pytest.raises(ValueError, match="too large") as error_info:
            _export_two_tiles(adapter, tmp_path, window_m, resample_m)

        # The error uses the export scale (resample_m when set) and names the dataset.
        assert scale_text in str(error_info.value)
        assert "dataset 'test_dataset'" in str(error_info.value)
        assert exported_points == []
        assert list(tmp_path.rglob("*.tif")) == []

    def test_tiles_within_the_limit_are_exported(self, tmp_path):
        """A tile request within the limit runs one export for each point."""
        adapter = _tile_export_adapter()
        exported_points = _record_tile_exports(adapter)

        paths, meta_list = _export_two_tiles(adapter, tmp_path, window_m=200)

        assert sorted(exported_points) == [(59.85, 17.63), (59.86, 17.64)]
        assert paths == [
            tmp_path / "test_dataset" / "a-test_dataset.tif",
            tmp_path / "test_dataset" / "b-test_dataset.tif",
        ]
        assert meta_list == [{}, {}]

    def test_size_check_asks_again_after_a_failed_band_request(self, tmp_path):
        """The size check retries a failed band-name request of the warm-up; the run continues."""
        image = _StubImage(["B1", "B2"], failures=1)
        adapter = _tile_export_adapter(
            _dataset_spec={"image": "users/test/image"}, _static_image=image
        )
        exported_points = _record_tile_exports(adapter)

        _export_two_tiles(adapter, tmp_path, window_m=200)

        assert image.n_band_name_requests == 2
        assert adapter._cached_band_count == 2
        assert len(exported_points) == 2

    def test_band_request_that_fails_again_stops_the_run(self, tmp_path):
        """A band-name request that fails in the warm-up and in the size check stops the run."""
        image = _StubImage(["B1", "B2"], failures=2)
        adapter = _tile_export_adapter(
            _dataset_spec={"image": "users/test/image"}, _static_image=image
        )
        exported_points = _record_tile_exports(adapter)

        with pytest.raises(_StubEarthEngineError, match="stub: band-name request failed"):
            _export_two_tiles(adapter, tmp_path, window_m=200)

        assert image.n_band_name_requests == 2
        assert exported_points == []


# ------------------------------------------------------------------
# _export_single — the pixel grid of one tile
# ------------------------------------------------------------------


def _record_tile_downloads(monkeypatch) -> list[dict]:
    """Replace the tile download with a recorder; return the list of recorded calls."""
    recorded_downloads: list[dict] = []

    def record_download(image, **download_arguments):
        recorded_downloads.append({"image": image, **download_arguments})

    monkeypatch.setattr(adapter_module, "_download_tile_via_url", record_download)
    return recorded_downloads


def _nearest_grid_line(coordinate_m: float, pixel_m: float) -> float:
    """Return the multiple of ``pixel_m`` nearest to ``coordinate_m``."""
    below = math.floor(coordinate_m / pixel_m) * pixel_m
    above = below + pixel_m
    return below if coordinate_m - below < above - coordinate_m else above


class TestExportSingleTileGrid:
    """``_export_single`` requests the exact n × n pixel grid, not a region and a scale."""

    # A point near Uppsala, in UTM zone 33N.
    _LAT, _LON = 59.85, 17.63

    def _point_utm(self) -> tuple[float, float]:
        from pyproj import Transformer

        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        return to_utm.transform(self._LON, self._LAT)

    def test_odd_size_with_resample_m(self, monkeypatch, tmp_path):
        """200 m at resample_m=30: a 7 × 7 grid whose centre pixel contains the point."""
        stub_image = _StubImage(["B1"])
        adapter = _make_offline_adapter(_get_image=lambda date=None, lat=None, lon=None: stub_image)
        recorded_downloads = _record_tile_downloads(monkeypatch)
        output_path = tmp_path / "tiles" / "a-test_dataset.tif"

        returned_path = adapter._export_single(
            self._LAT, self._LON, 200, output_path, None, resample_m=30
        )

        point_x, point_y = self._point_utm()
        # The point's pixel starts at floor(x / 30) · 30 (west edge) and ends at
        # (floor(y / 30) + 1) · 30 (north edge). Three pixels lie on each side.
        expected_west = (math.floor(point_x / 30) - 3) * 30
        expected_north = (math.floor(point_y / 30) + 1 + 3) * 30
        assert returned_path == output_path
        assert output_path.parent.is_dir()
        [download] = recorded_downloads
        assert set(download) == {"image", "crs", "crs_transform", "dimensions", "output_path"}
        assert download["image"] is stub_image
        assert download["crs"] == "EPSG:32633"
        assert download["crs_transform"] == pytest.approx(
            [30.0, 0.0, expected_west, 0.0, -30.0, expected_north]
        )
        assert download["dimensions"] == "7x7"
        assert download["output_path"] == output_path

    def test_even_size_at_the_native_scale(self, monkeypatch, tmp_path):
        """200 m at a native scale of 10 m: a 20 × 20 grid centred on the nearest grid lines."""
        stub_image = _StubImage(["B1"])
        adapter = _make_offline_adapter(
            _get_image=lambda date=None, lat=None, lon=None: stub_image,
            _get_scale_value=lambda image: 10.0,
        )
        recorded_downloads = _record_tile_downloads(monkeypatch)
        output_path = tmp_path / "a-test_dataset.tif"

        adapter._export_single(self._LAT, self._LON, 200, output_path, None, resample_m=None)

        point_x, point_y = self._point_utm()
        expected_west = _nearest_grid_line(point_x, 10.0) - 10 * 10.0
        expected_north = _nearest_grid_line(point_y, 10.0) + 10 * 10.0
        [download] = recorded_downloads
        assert set(download) == {"image", "crs", "crs_transform", "dimensions", "output_path"}
        assert download["image"] is stub_image
        assert download["crs"] == "EPSG:32633"
        assert download["crs_transform"] == pytest.approx(
            [10.0, 0.0, expected_west, 0.0, -10.0, expected_north]
        )
        assert download["dimensions"] == "20x20"


class TestBandNameCache:
    """``_get_band_name`` caches the band information only after a successful request."""

    def test_failed_request_is_not_cached(self):
        """A failed request raises and caches nothing. The next request succeeds and is cached."""
        image = _StubImage(["B1", "B2"], failures=1)
        adapter = _make_offline_adapter(_dataset_spec={"image": "users/test/image"})

        with pytest.raises(_StubEarthEngineError, match="stub: band-name request failed"):
            adapter._get_band_name(image)
        for attribute_name in ("_cached_band_name", "_cached_band_names", "_cached_band_count"):
            assert not hasattr(adapter, attribute_name)

        assert adapter._get_band_name(image) == "B1"
        assert adapter._cached_band_names == ["B1", "B2"]
        assert adapter._cached_band_count == 2

        # Later calls use the cache and send no request.
        assert adapter._get_band_name(image) == "B1"
        assert image.n_band_name_requests == 2


class TestEnsureGeeInit:
    """``_ensure_gee_init`` initializes Earth Engine only when it is not initialized."""

    @pytest.fixture
    def earth_engine_session(self, monkeypatch):
        """Record initializations, session-pool patches, and network calls, in call order.

        Earth Engine starts as not initialized. The stub ``init_gee`` initializes it.
        """
        session = SimpleNamespace(initialized=False, events=[])

        def fake_init_gee(*args, **kwargs):
            session.events.append("init_gee")
            session.initialized = True

        def record_pool_patch(pool_size):
            session.events.append(f"patch_session_pool({pool_size})")

        def record_network_call(*args, **kwargs):
            session.events.append("network call")
            raise AssertionError("An offline test made an Earth Engine network call.")

        monkeypatch.setattr(ee.data, "is_initialized", lambda: session.initialized)
        monkeypatch.setattr("envoi.auth.init_gee", fake_init_gee)
        monkeypatch.setattr(_image, "_patch_ee_session_pool", record_pool_patch)
        # The patch runs once per process: start each test as a new process.
        monkeypatch.setattr(_image, "_session_pool_patched", False)
        for function_name in ("getAlgorithms", "computeValue"):
            monkeypatch.setattr(ee.data, function_name, record_network_call)
        return session

    def test_initialized_session_is_kept(self, earth_engine_session):
        """With Earth Engine initialized: no request, no new initialization, and one pool patch."""
        earth_engine_session.initialized = True

        _image._ensure_gee_init()
        _image._ensure_gee_init()

        assert earth_engine_session.events == ["patch_session_pool(50)"]

    def test_uninitialized_earth_engine_is_initialized_once(self, earth_engine_session):
        """Without an initialized session, ``init_gee`` runs once, then the pool patch runs once."""
        _image._ensure_gee_init()
        _image._ensure_gee_init()

        assert earth_engine_session.events == ["init_gee", "patch_session_pool(50)"]


def _stub_failing_timestamp_request(monkeypatch, request_error: Exception) -> None:
    """Replace ``ee`` in ``_image`` so the timestamp request raises ``request_error``."""

    class FailingDictionary:
        """Stands in for ``ee.Dictionary``. ``getInfo`` raises like a failed request."""

        def __init__(self, values: dict):
            self._values = values

        def getInfo(self) -> dict:
            raise request_error

    def open_collection(collection_id):
        return _StubImageCollection({"system:time_start": [], "system:time_end": []})

    stub_ee = SimpleNamespace(ImageCollection=open_collection, Dictionary=FailingDictionary)
    monkeypatch.setattr(_image, "ee", stub_ee)


class TestCollectionTimeBoundsRequestFailure:
    """A failed timestamp request stops the run. A collection without images does not."""

    def test_failed_request_raises_runtime_error(self, monkeypatch):
        """The error names the collection, includes the original error, and chains it."""
        request_error = _StubEarthEngineError("stub: computation timed out")
        _stub_failing_timestamp_request(monkeypatch, request_error)

        with pytest.raises(RuntimeError, match="collection 'users/test/collection'") as error_info:
            _get_collection_time_bounds("users/test/collection")

        assert "stub: computation timed out" in str(error_info.value)
        assert error_info.value.__cause__ is request_error

    def test_collection_without_images_returns_no_index(self, monkeypatch):
        """A collection with no images at the batch's points gives ``(None, None)`` and no error."""
        _stub_collection_timestamps(monkeypatch, [], [])

        assert _get_collection_time_bounds("users/test/collection") == (None, None)

    def test_adapter_builds_no_fallback_after_a_failed_request(self, monkeypatch):
        """The adapter lets the error propagate and does not use the static-mosaic fallback."""
        request_error = _StubEarthEngineError("stub: computation timed out")
        _stub_failing_timestamp_request(monkeypatch, request_error)
        adapter = _make_offline_adapter(
            _dataset_spec={"collection": "users/test/collection"},
            _is_collection=True,
            _timestamps_loaded=False,
            _needs_per_point_date=True,
            _static_image=None,
            _collection_time_starts=None,
            _collection_time_ends=None,
        )

        with pytest.raises(RuntimeError, match="stub: computation timed out"):
            adapter._ensure_collection_timestamps()

        assert adapter._needs_per_point_date is True
        assert adapter._static_image is None
        assert adapter._timestamps_loaded is False


# ------------------------------------------------------------------
# Failed points — the per-point result loops of fetch_stats_batch and export_tiles
# ------------------------------------------------------------------

# Three points near Uppsala. The stubs below find a point's index by its latitude.
_THREE_LATS = [59.85, 59.86, 59.87]
_THREE_LONS = [17.63, 17.64, 17.65]
_THREE_IDS = ["p0", "p1", "p2"]


@pytest.fixture
def stub_combined_reducer(monkeypatch):
    """Replace the combined-reducer builder, which creates ``ee.Reducer`` objects."""
    monkeypatch.setattr(
        adapter_module,
        "_build_combined_reducer",
        lambda reducer_names: ("stub reducer", [f"_{name}" for name in reducer_names]),
    )


def _stub_two_band_requests(adapter: GeeRasterAdapter, failing_points: set[int] | None) -> dict:
    """Replace the per-point statistics request of ``adapter`` with a two-band stub.

    The request of each point in ``failing_points`` raises ``_StubEarthEngineError``
    (``None`` makes every request fail). The other points get ``B1_mean`` and
    ``B2_mean``, and ``B1_point`` and ``B2_point`` when the point reducer is
    requested, as a two-band image gives them.

    Returns a dict that maps the index of each failed point to its error.
    """
    raised_errors: dict[int, Exception] = {}

    def request(lat, lon, window_m, *args, want_point=False, **kwargs):
        point_index = _THREE_LATS.index(lat)
        if failing_points is None or point_index in failing_points:
            error = _StubEarthEngineError(f"stub: request failed for point {point_index}")
            raised_errors[point_index] = error
            raise error
        stats = {"B1_mean": 100.0 + point_index, "B2_mean": 200.0 + point_index}
        if want_point:
            stats.update({"B1_point": 10.0 + point_index, "B2_point": 20.0 + point_index})
        meta = {
            "in_extent": True,
            "n_pixels": 400,
            "had_nodata": False,
            "coverage_pct": 100.0,
            "window_m": window_m,
        }
        return stats, meta

    adapter._fetch_stats_single = request
    return raised_errors


def _stub_tile_exports(adapter: GeeRasterAdapter, failing_points: set[int] | None) -> dict:
    """Replace the per-point tile export of ``adapter``; the points in ``failing_points`` fail.

    ``None`` makes every export fail. A successful export writes no file and
    returns its path. Returns a dict that maps the index of each failed point
    to its error.
    """
    raised_errors: dict[int, Exception] = {}

    def export(lat, lon, window_m, output_path, date=None, resample_m=None):
        point_index = _THREE_LATS.index(lat)
        if failing_points is None or point_index in failing_points:
            error = _StubEarthEngineError(f"stub: download failed for point {point_index}")
            raised_errors[point_index] = error
            raise error
        return output_path

    adapter._export_single = export
    return raised_errors


class TestFailedPointsInBatch:
    """A failed request gives a failed point: no statistic keys, and the error in its meta."""

    @pytest.mark.parametrize(
        ("statistics", "expected_columns"),
        [
            (["mean"], ["stub_gee_B1_mean_200m", "stub_gee_B2_mean_200m"]),
            (
                ["mean", "point"],
                [
                    "stub_gee_B1_mean_200m",
                    "stub_gee_B2_mean_200m",
                    "stub_gee_B1_point",
                    "stub_gee_B2_point",
                ],
            ),
        ],
    )
    def test_failed_point_adds_no_statistic_column(
        self, stub_combined_reducer, statistics, expected_columns
    ):
        """A two-band batch with a failed point has the same columns as one without failures."""
        tables = []
        for failing_points in (set(), {1}):
            adapter = _tile_export_adapter()
            _stub_two_band_requests(adapter, failing_points)
            results = adapter.fetch_stats_batch(
                _THREE_LATS, _THREE_LONS, 200, statistics, disable_progress=True
            )
            tables.append(
                _append_stat_columns(pd.DataFrame({"id": _THREE_IDS}), "stub_gee", 200, results)
            )
        table_without_failure, table_with_failure = tables

        assert list(table_with_failure.columns) == ["id", *expected_columns]
        assert list(table_without_failure.columns) == ["id", *expected_columns]
        # The failed point is null in every statistic column; the others keep their values.
        assert table_with_failure.loc[1, expected_columns].isna().all()
        assert table_with_failure.loc[[0, 2], "stub_gee_B2_mean_200m"].tolist() == [200.0, 202.0]

    def test_failed_point_meta_has_the_error(self, stub_combined_reducer):
        """The failed point has an empty stats dict, the QC values of a point outside the extent, and its error."""
        adapter = _tile_export_adapter()
        raised_errors = _stub_two_band_requests(adapter, {1})

        results = adapter.fetch_stats_batch(
            _THREE_LATS, _THREE_LONS, 200, ["mean"], disable_progress=True
        )

        failed_stats, failed_meta = results[1]
        assert failed_stats == {}
        assert failed_meta["error"] == "_StubEarthEngineError: stub: request failed for point 1"
        assert failed_meta["exception"] is raised_errors[1]
        assert {key: failed_meta[key] for key in ("in_extent", "n_pixels", "had_nodata")} == {
            "in_extent": False,
            "n_pixels": 0,
            "had_nodata": False,
        }
        assert failed_meta["coverage_pct"] == 0.0
        assert "error" not in results[0][1] and "error" not in results[2][1]

    def test_failed_tile_meta_has_the_error(self, tmp_path):
        """A failed tile export gives no path, and its meta dict has the error."""
        adapter = _tile_export_adapter()
        raised_errors = _stub_tile_exports(adapter, {1})

        paths, meta_list = adapter.export_tiles(
            _THREE_LATS,
            _THREE_LONS,
            200,
            tmp_path,
            ids=_THREE_IDS,
            dataset_name="stub_gee",
            disable_progress=True,
        )

        assert paths[1] is None
        assert paths[0] == tmp_path / "stub_gee" / "p0-stub_gee.tif"
        assert meta_list[1] == {
            "error": "_StubEarthEngineError: stub: download failed for point 1",
            "exception": raised_errors[1],
        }
        assert meta_list[0] == {} and meta_list[2] == {}


@pytest.fixture
def stub_gee_dataset(monkeypatch, stub_combined_reducer) -> GeeRasterAdapter:
    """Register the Earth Engine dataset ``stub_gee``, whose adapter is an offline stub.

    ``extract()`` gets this adapter through its ``get_adapter`` lookup, so the
    test runs the real per-point result loops of the Earth Engine adapter.
    Returns the adapter, so the test can stub its per-point requests.
    """
    update_catalog(
        {
            "datasets": {
                "stub_gee": {
                    "data_source": "earth_engine",
                    "path": "users/test/image",
                    "data_type": "continuous",
                    "bands": ["B1", "B2"],
                }
            }
        }
    )
    adapter = _tile_export_adapter()
    # The package exports the function extract() under the module's name, so
    # get the module itself from the import system.
    extract_module = importlib.import_module("envoi.extract")
    monkeypatch.setattr(extract_module, "get_adapter", lambda data_source: lambda spec: adapter)
    return adapter


def _three_point_frame() -> pd.DataFrame:
    """The three stub points, as an input table without dates."""
    return pd.DataFrame(
        {
            "occurrenceID": _THREE_IDS,
            "decimalLatitude": _THREE_LATS,
            "decimalLongitude": _THREE_LONS,
        }
    )


def _stub_gee_config(batch_id: str, output_type: str) -> dict:
    """Return a run config for ``stub_gee`` with a 200 m window."""
    settings = {"output_type": output_type, "window_size_m": 200}
    if output_type == "tabular":
        settings["statistics"] = ["mean"]
    return {"batch_id": batch_id, "datasets": ["stub_gee"], "settings": settings}


class TestFailedPointsThroughExtract:
    """``extract()`` reports failed Earth Engine points, and stops when every point failed."""

    def test_one_failed_point_is_reported(self, stub_gee_dataset, tmp_path):
        """One failed request of three: same statistic columns, QC error column, sidecar fields."""
        _stub_two_band_requests(stub_gee_dataset, {1})

        with pytest.warns(UserWarning, match="1 of 3 points failed") as recorded_warnings:
            outputs = extract(
                _three_point_frame(), _stub_gee_config("gee_stats", "tabular"), output_dir=tmp_path
            )

        stats_table = pd.read_csv(outputs["gee_stats"])
        assert [column for column in stats_table.columns if column.startswith("stub_gee_")] == [
            "stub_gee_B1_mean_200m",
            "stub_gee_B2_mean_200m",
        ]
        qc_table = pd.read_csv(tmp_path / "gee_stats_qc.csv")
        assert qc_table["stub_gee_error_200m"].isna().tolist() == [True, False, True]
        assert (
            qc_table.loc[1, "stub_gee_error_200m"]
            == "_StubEarthEngineError: stub: request failed for point 1"
        )
        sidecar = json.loads((tmp_path / "gee_stats_metadata.json").read_text())
        quality = sidecar["datasets"]["stub_gee"]["quality"]["200m"]
        assert (quality["n_errors"], quality["ids_errors"]) == (1, ["p1"])
        failed_point_warnings = [
            recorded for recorded in recorded_warnings if "points failed" in str(recorded.message)
        ]
        assert len(failed_point_warnings) == 1

    def test_every_point_failing_raises_with_the_first_error(self, stub_gee_dataset, tmp_path):
        """When every request fails, RuntimeError has the error of point 0 and chains it."""
        raised_errors = _stub_two_band_requests(stub_gee_dataset, None)

        with pytest.raises(RuntimeError, match="every point failed") as error_info:
            extract(
                _three_point_frame(), _stub_gee_config("gee_failed", "tabular"), output_dir=tmp_path
            )

        message = str(error_info.value)
        assert "Dataset 'stub_gee', window 200 m" in message
        assert "First error: _StubEarthEngineError: stub: request failed for point 0" in message
        assert error_info.value.__cause__ is raised_errors[0]
        assert not (tmp_path / "gee_failed.csv").exists()

    def test_one_failed_tile_is_in_the_manifest(self, stub_gee_dataset, tmp_path):
        """One failed tile export of three: manifest error column and tile summary."""
        _stub_tile_exports(stub_gee_dataset, {1})

        with pytest.warns(UserWarning, match="1 of 3 points failed"):
            extract(
                _three_point_frame(), _stub_gee_config("gee_tiles", "raster"), output_dir=tmp_path
            )

        manifest = pd.read_csv(tmp_path / "gee_tiles" / "stub_gee" / "tiles_manifest.csv")
        assert manifest["exported"].tolist() == [True, False, True]
        assert manifest["error"].isna().tolist() == [True, False, True]
        assert (
            manifest.loc[1, "error"] == "_StubEarthEngineError: stub: download failed for point 1"
        )
        sidecar = json.loads((tmp_path / "gee_tiles" / "gee_tiles_metadata.json").read_text())
        tiles_summary = sidecar["datasets"]["stub_gee"]["quality"]["tiles"]
        assert tiles_summary["n_failed"] == 1
        assert (tiles_summary["n_errors"], tiles_summary["ids_errors"]) == (1, ["p1"])

    def test_every_tile_failing_raises_with_the_first_error(self, stub_gee_dataset, tmp_path):
        """When every tile export fails, RuntimeError has the error of point 0 and chains it."""
        raised_errors = _stub_tile_exports(stub_gee_dataset, None)

        with pytest.raises(RuntimeError, match="every point failed") as error_info:
            extract(
                _three_point_frame(),
                _stub_gee_config("gee_no_tiles", "raster"),
                output_dir=tmp_path,
            )

        assert "First error: _StubEarthEngineError: stub: download failed for point 0" in str(
            error_info.value
        )
        assert error_info.value.__cause__ is raised_errors[0]


# ------------------------------------------------------------------
# Statistics of one point — count-only runs and circular aspect statistics
# ------------------------------------------------------------------


class _StubReducer:
    """Stands in for an ``ee.Reducer``. Its name records the reducers combined into it."""

    def __init__(self, name: str):
        self.name = name

    def combine(self, reducer2, sharedInputs=False):
        return _StubReducer(f"{self.name}+{reducer2.name}")


class _StubStatisticsImage:
    """Stands in for the ``ee.Image`` of one point. ``reduceRegion`` returns a record of its call."""

    def __init__(self, label: str):
        self.label = label

    def select(self, band_name: str) -> _StubStatisticsImage:
        return _StubStatisticsImage(f"{self.label}.select({band_name})")

    def reduceRegion(self, **arguments) -> dict:
        return {"image": self.label, **arguments}


def _statistics_adapter(dataset_spec: dict) -> GeeRasterAdapter:
    """Return an offline adapter for an image asset at 10 m, with extra dataset-spec keys."""
    return _make_offline_adapter(
        _dataset_spec={"image": "users/test/dem", **dataset_spec},
        _collection_time_starts=None,
        _collection_time_ends=None,
        crs="EPSG:4326",
        scale=10,
    )


def _stub_statistics_request(monkeypatch, adapter, branch_results: dict[str, dict]) -> list[dict]:
    """Replace the Earth Engine objects that ``_fetch_stats_single`` uses with stubs.

    The point's request returns the entries of ``branch_results`` for the
    branches that the request contains. ``_get_image`` gives the image
    ``main``, or ``circular`` for the circular aspect image. Returns a list
    that records the branches of each request: branch name to the record of
    its ``reduceRegion`` call.
    """
    recorded_requests: list[dict] = []

    class _RecordingDictionary:
        def __init__(self, branches: dict):
            self._branches = branches
            recorded_requests.append(branches)

        def getInfo(self) -> dict:
            return {branch_name: branch_results[branch_name] for branch_name in self._branches}

    stub_ee = SimpleNamespace(
        Dictionary=_RecordingDictionary,
        Reducer=SimpleNamespace(
            mean=lambda: _StubReducer("mean"), count=lambda: _StubReducer("count")
        ),
    )
    monkeypatch.setattr(adapter_module, "ee", stub_ee)
    adapter._make_region = lambda lat, lon, window_m: f"square of {window_m} m"
    adapter._get_image = lambda date=None, lat=None, lon=None, circular_aspect=False: (
        _StubStatisticsImage("circular" if circular_aspect else "main")
    )
    return recorded_requests


def _fetch_point_statistics(adapter, reducer_names: list[str], suffixes: list[str]):
    """Run ``_fetch_stats_single`` for one point near Uppsala with a 200 m window."""
    return adapter._fetch_stats_single(
        59.85, 17.63, 200, _StubReducer("user reducers"), reducer_names, suffixes
    )


def _mean_sin_and_cos(directions_deg: list[float]) -> tuple[float, float]:
    """Return the mean sine and mean cosine of directions in degrees."""
    radians = [math.radians(direction) for direction in directions_deg]
    return (
        sum(math.sin(angle) for angle in radians) / len(radians),
        sum(math.cos(angle) for angle in radians) / len(radians),
    )


# Pixels that face 350° and 10°: circular mean 0° (north), circular standard
# deviation sqrt(-2 ln cos 10°) in degrees. The linear mean would be 180°.
_NORTH_MEAN_SIN, _NORTH_MEAN_COS = _mean_sin_and_cos([350.0, 10.0])
_NORTH_STD_DEG = math.degrees(math.sqrt(-2.0 * math.log(math.cos(math.radians(10.0)))))
_NORTH_CIRCULAR_RESULT = {
    "sin_mean": _NORTH_MEAN_SIN,
    "sin_count": 390,
    "cos_mean": _NORTH_MEAN_COS,
    "cos_count": 390,
}


def _distance_from_north_deg(direction_deg: float) -> float:
    """Return the shortest distance between a direction and north (0°) on the circle."""
    return min(direction_deg % 360.0, 360.0 - direction_deg % 360.0)


class TestCountOnlyStatistics:
    """A run whose only window statistic is ``count`` gives correct counts and QC values."""

    def test_single_band_count_only_gives_qc_counts(self, monkeypatch):
        """Single band: the bare band key is the count, and n_pixels and coverage use it."""
        adapter = _statistics_adapter({"bands": "DEM"})
        _stub_statistics_request(monkeypatch, adapter, {"window": {"DEM": 380}})

        stats, meta = _fetch_point_statistics(adapter, ["count"], ["_count"])

        assert stats == {"count": 380}
        assert (meta["in_extent"], meta["n_pixels"], meta["had_nodata"]) == (True, 380, True)
        # 380 of the 400 pixels of 10 m in a 200 m window.
        assert meta["coverage_pct"] == pytest.approx(95.0)

    def test_two_band_count_only_gives_band_count_columns(self, monkeypatch):
        """Two bands: one {band}_count column per band, and QC uses the smallest count."""
        adapter = _statistics_adapter({"bands": ["B1", "B2"]})
        _stub_statistics_request(monkeypatch, adapter, {"window": {"B1": 400, "B2": 380}})

        stats, meta = _fetch_point_statistics(adapter, ["count"], ["_count"])
        table = _append_stat_columns(pd.DataFrame({"id": ["p0"]}), "stub_gee", 200, [(stats, meta)])

        assert stats == {"B1_count": 400, "B2_count": 380}
        assert table.loc[0, ["stub_gee_B1_count_200m", "stub_gee_B2_count_200m"]].tolist() == [
            400,
            380,
        ]
        assert (meta["in_extent"], meta["n_pixels"]) == (True, 380)
        assert meta["coverage_pct"] == pytest.approx(95.0)
        assert meta["band_coverage_pct"] == {"B1": 100.0, "B2": 95.0}


class TestCircularAspectInPointStatistics:
    """The circular aspect values replace only the requested linear mean, std, and var of aspect."""

    def test_multi_band_replaces_only_requested_aspect_statistics(self, monkeypatch):
        """Multi-band: aspect_mean and aspect_std become circular; other keys keep their values."""
        adapter = _statistics_adapter({"bands": ["DEM"], "derived_bands": ["slope", "aspect"]})
        window_result = {
            "DEM_mean": 120.0,
            "DEM_median": 118.0,
            "DEM_stdDev": 6.0,
            "DEM_count": 400,
            "slope_mean": 12.0,
            "slope_median": 11.0,
            "slope_stdDev": 3.0,
            "slope_count": 400,
            "aspect_mean": 180.0,
            "aspect_median": 355.0,
            "aspect_stdDev": 170.0,
            "aspect_count": 400,
        }
        recorded_requests = _stub_statistics_request(
            monkeypatch,
            adapter,
            {"window": window_result, "aspect_circular": _NORTH_CIRCULAR_RESULT},
        )

        stats, meta = _fetch_point_statistics(
            adapter, ["mean", "median", "std"], ["_mean", "_median", "_stdDev"]
        )

        # Circular values for the requested aspect mean and std.
        assert 0.0 <= stats["aspect_mean"] < 360.0
        assert _distance_from_north_deg(stats["aspect_mean"]) == pytest.approx(0.0, abs=1e-9)
        assert stats["aspect_std"] == pytest.approx(_NORTH_STD_DEG)
        # Linear values everywhere else; no key for the statistic that was not requested.
        assert stats["aspect_median"] == 355.0
        assert {key: stats[key] for key in ("DEM_mean", "DEM_std", "slope_mean", "slope_std")} == {
            "DEM_mean": 120.0,
            "DEM_std": 6.0,
            "slope_mean": 12.0,
            "slope_std": 3.0,
        }
        assert "aspect_var" not in stats
        assert len(stats) == 9
        # QC comes from the window branch.
        assert (meta["n_pixels"], meta["in_extent"]) == (400, True)

        # One request, with the circular branch on the same region and grid as the window.
        assert len(recorded_requests) == 1
        branches = recorded_requests[0]
        assert set(branches) == {"window", "aspect_circular"}
        window_branch, circular_branch = branches["window"], branches["aspect_circular"]
        assert (window_branch["image"], circular_branch["image"]) == ("main", "circular")
        assert circular_branch["reducer"].name == "mean+count"
        for argument_name in ("geometry", "scale", "crs", "bestEffort"):
            assert circular_branch[argument_name] == window_branch[argument_name]
        assert circular_branch["bestEffort"] is True

    def test_single_band_replaces_only_requested_aspect_statistics(self, monkeypatch):
        """Single band (derived aspect only): mean and var become circular; max stays linear."""
        adapter = _statistics_adapter({"derived_bands": ["aspect"]})
        window_result = {
            "aspect_mean": 180.0,
            "aspect_variance": 28000.0,
            "aspect_max": 359.0,
            "aspect_count": 400,
        }
        recorded_requests = _stub_statistics_request(
            monkeypatch,
            adapter,
            {"window": window_result, "aspect_circular": _NORTH_CIRCULAR_RESULT},
        )

        stats, _ = _fetch_point_statistics(
            adapter, ["mean", "var", "max"], ["_mean", "_variance", "_max"]
        )

        assert set(stats) == {"mean", "var", "max"}
        assert _distance_from_north_deg(stats["mean"]) == pytest.approx(0.0, abs=1e-9)
        assert stats["var"] == pytest.approx(_NORTH_STD_DEG**2)
        assert stats["max"] == 359.0
        assert recorded_requests[0]["window"]["image"] == "main.select(aspect)"
        assert recorded_requests[0]["aspect_circular"]["image"] == "circular"

    def test_flat_window_gives_null_aspect_and_keeps_qc(self, monkeypatch):
        """A window with only flat pixels gives a null circular mean, and QC still shows its pixels."""
        adapter = _statistics_adapter({"derived_bands": ["aspect"]})
        flat_circular_result = {"sin_mean": None, "sin_count": 0, "cos_mean": None, "cos_count": 0}
        _stub_statistics_request(
            monkeypatch,
            adapter,
            # Earth Engine gives flat pixels aspect 0, so the linear mean is 0.
            {
                "window": {"aspect_mean": 0.0, "aspect_count": 400},
                "aspect_circular": flat_circular_result,
            },
        )

        stats, meta = _fetch_point_statistics(adapter, ["mean"], ["_mean"])

        assert stats == {"mean": None}
        assert (meta["in_extent"], meta["n_pixels"], meta["coverage_pct"]) == (True, 400, 100.0)

    @pytest.mark.parametrize(
        ("dataset_spec", "reducer_names", "suffixes"),
        [
            # Aspect is reduced, but no mean, std, or var is requested.
            (
                {"bands": ["DEM"], "derived_bands": ["slope", "aspect"]},
                ["median", "max"],
                ["_median", "_max"],
            ),
            # No derived aspect band.
            ({"bands": ["B1", "B2"]}, ["mean"], ["_mean"]),
            # A single named source band: the derived aspect band is not reduced.
            ({"bands": "DEM", "derived_bands": ["aspect"]}, ["mean"], ["_mean"]),
            # A source band named "aspect" is not the derived band.
            ({"bands": ["aspect"]}, ["mean", "std"], ["_mean", "_stdDev"]),
        ],
    )
    def test_no_circular_branch_without_reduced_derived_aspect(
        self, monkeypatch, dataset_spec, reducer_names, suffixes
    ):
        """The request has no circular branch unless derived aspect is reduced with mean, std, or var."""
        adapter = _statistics_adapter(dataset_spec)
        recorded_requests = _stub_statistics_request(monkeypatch, adapter, {"window": {}})

        _fetch_point_statistics(adapter, reducer_names, suffixes)

        assert [set(branches) for branches in recorded_requests] == [{"window"}]


class TestCircularAspectImageSelection:
    """``_get_image`` builds the circular aspect image fresh, and never uses ``_static_image``."""

    @pytest.fixture
    def recorded_builds(self, monkeypatch) -> list[dict]:
        """Replace ``_build_image`` and ``ee`` in the adapter module; record each build."""
        builds: list[dict] = []

        def record_build(dataset_spec, date=None, geometry=None, *args, **keywords):
            builds.append({"date": date, "geometry": geometry, **keywords})
            return f"image {len(builds)}"

        monkeypatch.setattr(adapter_module, "_build_image", record_build)
        stub_geometry = SimpleNamespace(Point=lambda coordinates: f"point {coordinates}")
        monkeypatch.setattr(adapter_module, "ee", SimpleNamespace(Geometry=stub_geometry))
        return builds

    def test_image_asset_does_not_use_the_static_image(self, recorded_builds):
        """For an image asset, the circular image is built without a date; the static image stays."""
        adapter = _make_offline_adapter(
            _dataset_spec={"image": "users/test/dem", "derived_bands": ["aspect"]},
            _static_image="static image",
            _needs_per_point_date=False,
            _collection_time_starts=None,
            _collection_time_ends=None,
        )

        circular_image = adapter._get_image(None, lat=59.85, lon=17.63, circular_aspect=True)

        assert circular_image == "image 1"
        assert recorded_builds[0]["circular_aspect"] is True
        assert recorded_builds[0]["date"] is None
        assert adapter._static_image == "static image"
        assert adapter._get_image(None, lat=59.85, lon=17.63) == "static image"

    def test_most_recent_fallback_is_not_cached(self, recorded_builds):
        """Without coordinates, the circular image of a collection is not cached; the main one is."""
        adapter = _make_offline_adapter(
            _dataset_spec={"collection": "users/test/collection", "derived_bands": ["aspect"]},
            _static_image=None,
            _needs_per_point_date=True,
            _collection_time_starts=pd.DatetimeIndex(["2021-01-01", "2021-02-01"]),
            _collection_time_ends=None,
        )

        assert adapter._get_image(circular_aspect=True) == "image 1"
        assert adapter._static_image is None
        assert adapter._get_image() == "image 2"
        assert adapter._static_image == "image 2"
        assert adapter._get_image(circular_aspect=True) == "image 3"
        assert [build["circular_aspect"] for build in recorded_builds] == [True, False, True]
        assert recorded_builds[0]["date"] == pd.Timestamp("2021-02-01")

    @pytest.mark.parametrize(
        ("dataset_spec", "needs_per_point_date"),
        [
            ({"collection_date_policy": "mosaic"}, False),
            ({}, True),
        ],
    )
    def test_per_point_image_passes_the_flag(
        self, recorded_builds, dataset_spec, needs_per_point_date
    ):
        """Per-point collection images (mosaic or by date) get the flag and the point geometry."""
        adapter = _make_offline_adapter(
            _dataset_spec={
                "collection": "users/test/collection",
                "derived_bands": ["aspect"],
                **dataset_spec,
            },
            _static_image="static image",
            _needs_per_point_date=needs_per_point_date,
            _collection_time_starts=pd.DatetimeIndex(["2021-01-01", "2021-02-01"]),
            _collection_time_ends=None,
        )

        adapter._get_image("2021-01-10", lat=59.85, lon=17.63, circular_aspect=True)

        assert recorded_builds[0]["circular_aspect"] is True
        assert recorded_builds[0]["geometry"] == "point [17.63, 59.85]"
        assert adapter._static_image == "static image"


class _SymbolicImage:
    """Stands in for an ``ee.Image``. Each operation returns an image whose text records it."""

    def __init__(self, expression: str):
        self.expression = expression

    def multiply(self, factor: float) -> _SymbolicImage:
        return _SymbolicImage(f"{self.expression} * {factor:.6f}")

    def sin(self) -> _SymbolicImage:
        return _SymbolicImage(f"sin({self.expression})")

    def cos(self) -> _SymbolicImage:
        return _SymbolicImage(f"cos({self.expression})")

    def rename(self, band_name: str) -> _SymbolicImage:
        return _SymbolicImage(f"{self.expression} as {band_name}")

    def gt(self, threshold) -> _SymbolicImage:
        return _SymbolicImage(f"{self.expression} > {threshold}")

    def updateMask(self, mask: _SymbolicImage) -> _SymbolicImage:
        return _SymbolicImage(f"({self.expression}) masked to {mask.expression}")

    def select(self, band) -> _SymbolicImage:
        return _SymbolicImage(f"{self.expression}.select({band})")

    def projection(self) -> str:
        return f"projection of {self.expression}"

    def setDefaultProjection(self, projection: str) -> _SymbolicImage:
        return _SymbolicImage(f"{self.expression} in {projection}")


class _SymbolicCollection:
    """Stands in for an ``ee.ImageCollection``, with the operations of a mosaic collection."""

    def __init__(self, expression: str):
        self.expression = expression

    def filterBounds(self, geometry: str) -> _SymbolicCollection:
        return _SymbolicCollection(f"{self.expression} at {geometry}")

    def first(self) -> _SymbolicImage:
        return _SymbolicImage(f"first of {self.expression}")

    def mosaic(self) -> _SymbolicImage:
        return _SymbolicImage(f"mosaic of {self.expression}")


def _concatenate_symbolic_images(*images: _SymbolicImage) -> _SymbolicImage:
    """Stands in for ``ee.Image.cat``."""
    return _SymbolicImage("cat(" + ", ".join(image.expression for image in images) + ")")


class TestCircularAspectImage:
    """``_build_image(circular_aspect=True)`` gives sin and cos of aspect, without flat pixels."""

    @pytest.fixture(autouse=True)
    def symbolic_earth_engine(self, monkeypatch):
        """Replace ``ee`` and the derived-band functions in ``_image`` with symbolic stubs."""

        def open_image(asset_id: str) -> _SymbolicImage:
            return _SymbolicImage(asset_id)

        open_image.cat = _concatenate_symbolic_images  # ee.Image.cat
        stub_ee = SimpleNamespace(Image=open_image, ImageCollection=_SymbolicCollection)
        monkeypatch.setattr(_image, "ee", stub_ee)
        monkeypatch.setitem(
            _image._DERIVED_BAND_DISPATCH,
            "aspect",
            lambda image: _SymbolicImage(f"aspect({image.expression})"),
        )
        monkeypatch.setitem(
            _image._DERIVED_BAND_DISPATCH,
            "slope",
            lambda image: _SymbolicImage(f"slope({image.expression})"),
        )

    def test_image_asset(self):
        """Aspect and slope come from the source image; flat pixels (slope 0) are masked."""
        circular_image = _image._build_image(
            {"image": "dem", "bands": ["DEM"], "derived_bands": ["slope", "aspect"]},
            circular_aspect=True,
        )

        assert circular_image.expression == (
            "(cat(sin(aspect(dem) * 0.017453) as sin, cos(aspect(dem) * 0.017453) as cos))"
            " masked to slope(dem) > 0"
        )

    def test_mosaic_collection_keeps_the_native_projection(self):
        """For a space-tiled collection, the circular image uses the projected mosaic."""
        circular_image = _image._build_image(
            {
                "collection": "tiles",
                "collection_date_policy": "mosaic",
                "derived_bands": ["aspect"],
            },
            geometry="point",
            circular_aspect=True,
        )

        mosaic = "mosaic of tiles at point in projection of first of tiles.select(0)"
        assert circular_image.expression == (
            f"(cat(sin(aspect({mosaic}) * 0.017453) as sin, "
            f"cos(aspect({mosaic}) * 0.017453) as cos)) masked to slope({mosaic}) > 0"
        )
