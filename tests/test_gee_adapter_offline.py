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
import requests

from envoi import update_catalog
from envoi._config_parsing import _parse_run_config
from envoi._output_assembly import _append_stat_columns, _resolve_dataset_metadata
from envoi.adapters.earth_engine import _image, _tiles
from envoi.adapters.earth_engine import adapter as adapter_module
from envoi.adapters.earth_engine._image import (
    _get_collection_time_bounds,
    _resolve_date_filter_range,
)
from envoi.adapters.earth_engine.adapter import GeeRasterAdapter
from envoi.catalog import BUILTIN_EE_CATALOG, load_catalogs, load_defaults
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


class _StubPropertyNames:
    """Stands in for the ``ee.List`` of ``propertyNames()``. ``contains`` answers at once."""

    def __init__(self, property_names: list[str]):
        self._property_names = property_names

    def contains(self, property_name: str) -> bool:
        return property_name in self._property_names


class _StubElement:
    """Stands in for one ``ee.Image`` or ``ee.Feature`` of a collection, with its properties."""

    def __init__(self, properties: dict):
        self.properties = properties

    def propertyNames(self) -> _StubPropertyNames:
        return _StubPropertyNames(list(self.properties))

    def get(self, property_name: str):
        return self.properties.get(property_name)


class _StubImageCollection:
    """Stands in for ``ee.ImageCollection``. Each element is a dict of its properties.

    ``map`` calls the function once for each element, as the server does.
    ``aggregate_array`` leaves out the elements without the property, as
    Earth Engine does (seen live on 2026-10-12).
    """

    def __init__(self, element_properties: list[dict]):
        self._element_properties = element_properties

    def map(self, function) -> _StubImageCollection:
        return _StubImageCollection(
            [
                function(_StubElement(properties)).properties
                for properties in self._element_properties
            ]
        )

    def aggregate_array(self, property_name: str) -> list:
        return [
            properties[property_name]
            for properties in self._element_properties
            if properties.get(property_name) is not None
        ]


class _StubDictionary:
    """Stands in for ``ee.Dictionary``. ``getInfo`` returns the wrapped values."""

    def __init__(self, values: dict):
        self._values = values

    def getInfo(self) -> dict:
        return dict(self._values)


def _stub_server_side_if(condition: bool, true_case, false_case):
    """Stands in for ``ee.Algorithms.If``, with a condition that the stubs answer at once."""
    return true_case if condition else false_case


# The ``ee`` names that the timestamp request uses, other than the collection
# and the dictionary.
_STUB_TIMESTAMP_EE_NAMES = {
    "Feature": lambda geometry, properties: _StubElement(dict(properties)),
    "Algorithms": SimpleNamespace(If=_stub_server_side_if),
}


def _stub_collection_timestamps(monkeypatch, starts: list, ends: list) -> list[str]:
    """Replace ``ee`` in ``_image`` so the collection has one image for each start and end.

    ``starts[i]`` and ``ends[i]`` are the ``system:time_start`` and
    ``system:time_end`` of image ``i``. ``None`` means that the image does
    not have the property. Returns a list that records the collection IDs
    that the code under test opened.
    """
    image_properties = [
        {
            property_name: value
            for property_name, value in (("system:time_start", start), ("system:time_end", end))
            if value is not None
        }
        for start, end in zip(starts, ends, strict=True)
    ]
    opened_collection_ids: list[str] = []

    def open_collection(collection_id):
        opened_collection_ids.append(collection_id)
        return _StubImageCollection(image_properties)

    stub_ee = SimpleNamespace(
        ImageCollection=open_collection, Dictionary=_StubDictionary, **_STUB_TIMESTAMP_EE_NAMES
    )
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
        ends = [_epoch_ms("2021-01-09"), None, None]
        _stub_collection_timestamps(monkeypatch, starts, ends)

        start_index, end_index = _get_collection_time_bounds("users/test/collection")

        assert end_index is None
        assert list(start_index) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01")]


class TestCollectionTimeBoundsMissingValues:
    """Each start keeps the end of its own image when some timestamps are missing."""

    def test_probe_collection_gives_a_start_only_index(self, monkeypatch):
        """The collection of the 2026-10-12 probe: start only, start and end, end only."""
        # Two plain aggregate_array lists would be [1000, 2000] and
        # [3000, 5000]: the same length, but 1000 would get the end 3000 of
        # another image, and 2000 the end 5000 of the image without a start.
        _stub_collection_timestamps(monkeypatch, [1000, 2000, None], [None, 3000, 5000])

        start_index, end_index = _get_collection_time_bounds("users/test/collection")

        # The image without a start is dropped, and the image with a start but
        # no end makes the index start-only.
        assert end_index is None
        assert list(start_index) == [
            pd.Timestamp("1970-01-01 00:00:01"),
            pd.Timestamp("1970-01-01 00:00:02"),
        ]

    @pytest.mark.parametrize(
        ("starts", "ends", "expected_starts", "expected_ends"),
        [
            # An image without a start (with an end, as in the probe) is
            # dropped, and the other images keep their own ends.
            (
                [_epoch_ms("1958-01-01"), _epoch_ms("1958-02-01"), None],
                [_epoch_ms("1958-02-01"), _epoch_ms("1958-03-01"), _epoch_ms("1958-04-01")],
                ["1958-01-01", "1958-02-01"],
                ["1958-02-01", "1958-03-01"],
            ),
            # Times before 1970 are negative numbers, and they are times.
            (
                [_epoch_ms("1958-02-01"), _epoch_ms("1958-01-01")],
                [_epoch_ms("1958-03-01"), _epoch_ms("1958-02-01")],
                ["1958-01-01", "1958-02-01"],
                ["1958-02-01", "1958-03-01"],
            ),
            # A negative start without an end makes the index start-only.
            (
                [_epoch_ms("1958-01-01"), _epoch_ms("1958-02-01")],
                [_epoch_ms("1958-02-01"), None],
                ["1958-01-01", "1958-02-01"],
                None,
            ),
        ],
    )
    def test_negative_times_are_paired(
        self, monkeypatch, starts, ends, expected_starts, expected_ends
    ):
        """Times before 1970 (negative, as in TerraClimate) are paired per image, as other times."""
        assert all(start is None or start < 0 for start in starts)
        _stub_collection_timestamps(monkeypatch, starts, ends)

        start_index, end_index = _get_collection_time_bounds("users/test/collection")

        assert list(start_index) == [pd.Timestamp(start) for start in expected_starts]
        if expected_ends is None:
            assert end_index is None
        else:
            assert list(end_index) == [pd.Timestamp(end) for end in expected_ends]

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
    """For ``contains``, a date is clamped only when it is before the first image's
    start day, or at or after the end of the last image interval.
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

    def test_last_start_later_on_its_day_without_end_times(self):
        """Without end times, the last interval ends at the end of the last start's day."""
        time_starts = pd.DatetimeIndex(["2021-06-07 10:00", "2021-06-15 10:00"])
        adapter = _contains_adapter(time_starts, None)

        assert adapter._resolve_date_info("2021-06-15") == {
            "date_clamped": False,
            "date_source": "contains_sample_date",
            "image_time_start": "2021-06-15",
        }
        assert adapter._resolve_date_info("2021-06-16")["date_clamped"] is True

    @pytest.mark.parametrize(
        ("sample_date", "clamped", "image_start", "image_end"),
        [
            ("2021-06-06", True, "2021-06-07", "2021-06-15"),  # before the first start day
            # On the first start day: the first image starts at 10:00 on that day.
            ("2021-06-07", False, "2021-06-07", "2021-06-15"),
            # The image that starts at 10:00 on the sample day, not the one before.
            ("2021-06-15", False, "2021-06-15", "2021-06-23"),
            # The last image ends at 10:00 on this day, after the start of the day.
            ("2021-07-01", False, "2021-06-23", "2021-07-01"),
            ("2021-07-02", True, "2021-06-23", "2021-07-01"),  # after the last interval
        ],
    )
    def test_images_that_start_later_on_the_sample_day(
        self, sample_date, clamped, image_start, image_end
    ):
        """Day rule: an image that starts at 10:00 on the sample day contains that day."""
        time_starts = pd.DatetimeIndex(["2021-06-07 10:00", "2021-06-15 10:00", "2021-06-23 10:00"])
        time_ends = pd.DatetimeIndex(["2021-06-15 10:00", "2021-06-23 10:00", "2021-07-01 10:00"])
        adapter = _contains_adapter(time_starts, time_ends)

        assert adapter._resolve_date_info(sample_date) == {
            "date_clamped": clamped,
            "date_source": "clamped_to_nearest" if clamped else "contains_sample_date",
            "image_time_start": image_start,
            "image_time_end": image_end,
        }
        # The reported image is the image that the filterDate range selects.
        assert _resolve_date_filter_range(
            pd.Timestamp(sample_date), "contains", time_starts, time_ends
        ) == (image_start, image_end)


# ------------------------------------------------------------------
# summarize_date_info — the sidecar date_info summary
# ------------------------------------------------------------------


class TestSummarizeDateInfo:
    """The sidecar ``date_info`` uses every point that has date information and no error."""

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

    def test_failed_tile_is_not_counted(self):
        """A meta dict with an error (a failed tile keeps its date keys) is not counted."""
        successful_meta = {
            "date_clamped": False,
            "date_source": "nearest_to_sample",
            "image_time_start": "2021-01-01",
        }
        failed_meta = {
            "date_clamped": True,
            "date_source": "clamped_to_nearest",
            "image_time_start": "2021-02-01",
            "error": "_StubEarthEngineError: stub: download failed",
        }

        date_info = summarize_date_info([successful_meta, failed_meta])

        assert date_info == {
            "n_nearest_to_sample": 1,
            "n_contains_sample_date": 0,
            "n_clamped_to_nearest": 0,
            "n_most_recent_no_date": 0,
            "image_date_ranges_used": [["2021-01-01", "2021-01-01"]],
        }
        assert summarize_date_info([failed_meta]) is None


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


# ------------------------------------------------------------------
# _download_tile_via_url — the request and the file of one tile
# ------------------------------------------------------------------


class _StubDownloadImage:
    """Stands in for an ``ee.Image``. ``getDownloadURL`` records its parameters."""

    def __init__(self):
        self.download_parameters: list[dict] = []

    def getDownloadURL(self, parameters: dict) -> str:
        self.download_parameters.append(parameters)
        return "https://earthengine.invalid/tile"


class _FailingDownloadImage:
    """Stands in for an ``ee.Image`` whose ``getDownloadURL`` raises an Earth Engine error."""

    def getDownloadURL(self, parameters: dict) -> str:
        raise ee.EEException("stub: image has no bands")


class _StubTileResponse:
    """Stands in for a streamed ``requests`` response.

    ``chunks`` holds the parts of the body. An exception in ``chunks`` is
    raised when the stream reaches it, as when a connection breaks.
    ``iter_content`` gives each part in pieces of at most ``chunk_size`` bytes,
    as ``requests`` does, and ``bytes_read`` counts the bytes that it gave.
    """

    def __init__(self, chunks: list, status_code: int = 200):
        self._chunks = chunks
        self.status_code = status_code
        self.bytes_read = 0

    def close(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            for start in range(0, len(chunk), chunk_size):
                piece = chunk[start : start + chunk_size]
                self.bytes_read += len(piece)
                yield piece


def _broken_stream() -> _StubTileResponse:
    """Return a response whose connection breaks after the first part of the body."""
    broken_connection = requests.exceptions.ChunkedEncodingError("stub: connection broken")
    return _StubTileResponse([b"partial ", broken_connection])


def _stub_tile_responses(monkeypatch, responses: list[_StubTileResponse]) -> SimpleNamespace:
    """Replace ``requests`` and ``time`` in ``_tiles``: GET number N returns ``responses[N - 1]``.

    Returns a record with the URL of each GET (``requested_urls``) and the
    length of each wait in seconds (``waits``). The waits do not sleep.
    """
    record = SimpleNamespace(requested_urls=[], waits=[])
    remaining_responses = list(responses)

    def get(url, stream=False, timeout=None):
        record.requested_urls.append(url)
        return remaining_responses.pop(0)

    stub_requests = SimpleNamespace(
        get=get, RequestException=requests.RequestException, HTTPError=requests.HTTPError
    )
    monkeypatch.setattr(_tiles, "requests", stub_requests)
    monkeypatch.setattr(_tiles, "time", SimpleNamespace(sleep=record.waits.append))
    return record


# The grid of a 7 × 7 tile of 30 m pixels in UTM zone 33N.
_TILE_CRS_TRANSFORM = [30.0, 0.0, 600_000.0, 0.0, -30.0, 6_640_000.0]


class TestDownloadTile:
    """``_download_tile_via_url`` requests the exact pixel grid and writes only complete tiles."""

    def _download(self, output_path, image: _StubDownloadImage | None = None) -> None:
        _tiles._download_tile_via_url(
            image or _StubDownloadImage(),
            crs="EPSG:32633",
            crs_transform=_TILE_CRS_TRANSFORM,
            dimensions="7x7",
            output_path=output_path,
        )

    def test_request_has_the_grid_and_no_region_or_scale(self, monkeypatch, tmp_path):
        """The download parameters are the grid and the format; only the tile is written."""
        record = _stub_tile_responses(monkeypatch, [_StubTileResponse([b"GeoTIFF ", b"bytes"])])
        image = _StubDownloadImage()
        output_path = tmp_path / "a-test_dataset.tif"

        self._download(output_path, image)

        assert image.download_parameters == [
            {
                "crs": "EPSG:32633",
                "crs_transform": _TILE_CRS_TRANSFORM,
                "dimensions": "7x7",
                "format": "GEO_TIFF",
                "filePerBand": False,
            }
        ]
        assert record.requested_urls == ["https://earthengine.invalid/tile"]
        assert record.waits == []
        assert output_path.read_bytes() == b"GeoTIFF bytes"
        # No temporary file stays next to the tile.
        assert list(tmp_path.iterdir()) == [output_path]

    @pytest.mark.parametrize(
        "failed_response",
        [_broken_stream, lambda: _StubTileResponse([], status_code=503)],
        ids=["broken stream", "HTTP 503"],
    )
    def test_good_response_after_a_failed_attempt(self, monkeypatch, tmp_path, failed_response):
        """After a failed attempt, the next good response gives the tile, with nothing else."""
        record = _stub_tile_responses(
            monkeypatch, [failed_response(), _StubTileResponse([b"good ", b"tile"])]
        )
        output_path = tmp_path / "a-test_dataset.tif"

        self._download(output_path)

        assert output_path.read_bytes() == b"good tile"
        assert len(record.requested_urls) == 2
        assert record.waits == [1]
        assert list(tmp_path.iterdir()) == [output_path]

    def test_existing_tile_is_replaced(self, monkeypatch, tmp_path):
        """A tile of an earlier run with the same name is replaced by the new tile."""
        _stub_tile_responses(monkeypatch, [_StubTileResponse([b"new tile"])])
        output_path = tmp_path / "a-test_dataset.tif"
        output_path.write_bytes(b"old tile")

        self._download(output_path)

        assert output_path.read_bytes() == b"new tile"
        assert list(tmp_path.iterdir()) == [output_path]

    def test_broken_stream_is_retried_and_leaves_no_file(self, monkeypatch, tmp_path):
        """When every attempt breaks during the stream, the error is raised and no file stays."""
        record = _stub_tile_responses(monkeypatch, [_broken_stream() for _ in range(3)])

        with pytest.raises(
            RuntimeError, match="after 3 attempts: stub: connection broken"
        ) as error_info:
            self._download(tmp_path / "a-test_dataset.tif")

        assert isinstance(error_info.value.__cause__, requests.exceptions.ChunkedEncodingError)
        assert len(record.requested_urls) == 3
        # A wait before each retry, and none after the last attempt.
        assert record.waits == [1, 2]
        assert list(tmp_path.iterdir()) == []

    def test_existing_tile_is_removed_when_every_attempt_fails(self, monkeypatch, tmp_path):
        """A tile of an earlier run is removed, so the folder matches the manifest (not exported)."""
        _stub_tile_responses(
            monkeypatch, [_StubTileResponse([], status_code=503) for _ in range(3)]
        )
        output_path = tmp_path / "a-test_dataset.tif"
        output_path.write_bytes(b"old tile")

        with pytest.raises(RuntimeError, match="returned HTTP 503 on attempt 3"):
            self._download(output_path)

        assert list(tmp_path.iterdir()) == []

    def test_other_error_in_the_stream_leaves_no_file(self, monkeypatch, tmp_path):
        """An error that is not a request error stops the download at once, and no file stays."""
        disk_error = OSError("stub: no space left on device")
        record = _stub_tile_responses(monkeypatch, [_StubTileResponse([b"partial ", disk_error])])

        with pytest.raises(OSError, match="stub: no space left on device"):
            self._download(tmp_path / "a-test_dataset.tif")

        assert len(record.requested_urls) == 1
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        ("body", "expected_text"),
        [
            (
                json.dumps(
                    {"error": {"code": 400, "message": "stub: invalid crs_transform"}}
                ).encode(),
                "HTTP 400: stub: invalid crs_transform",
            ),
            (b"stub: bad\n request", "HTTP 400: stub: bad request"),
        ],
        ids=["JSON body", "text body"],
    )
    def test_client_error_keeps_the_message(self, monkeypatch, tmp_path, body, expected_text):
        """An HTTP 400 answer raises at once, with Earth Engine's message, and no file stays."""
        record = _stub_tile_responses(monkeypatch, [_StubTileResponse([body], status_code=400)])

        with pytest.raises(requests.HTTPError) as error_info:
            self._download(tmp_path / "a-test_dataset.tif")

        assert str(error_info.value) == f"GEE tile download returned {expected_text}"
        assert len(record.requested_urls) == 1
        assert record.waits == []
        assert list(tmp_path.iterdir()) == []

    def test_existing_tile_is_removed_after_a_client_error(self, monkeypatch, tmp_path):
        """After an HTTP 400 answer, a tile of an earlier run is removed (not exported)."""
        _stub_tile_responses(monkeypatch, [_StubTileResponse([b"stub: bad"], status_code=400)])
        output_path = tmp_path / "a-test_dataset.tif"
        output_path.write_bytes(b"old tile")

        with pytest.raises(requests.HTTPError, match="returned HTTP 400: stub: bad$"):
            self._download(output_path)

        assert list(tmp_path.iterdir()) == []

    def test_existing_tile_is_removed_when_the_download_url_fails(self, monkeypatch, tmp_path):
        """When ``getDownloadURL`` raises, no download starts and an older tile is removed."""
        record = _stub_tile_responses(monkeypatch, [])
        output_path = tmp_path / "a-test_dataset.tif"
        output_path.write_bytes(b"old tile")

        with pytest.raises(ee.EEException, match="stub: image has no bands"):
            self._download(output_path, _FailingDownloadImage())

        assert record.requested_urls == []
        assert list(tmp_path.iterdir()) == []

    def test_success_status_other_than_200_is_not_a_tile(self, monkeypatch, tmp_path):
        """An HTTP 204 answer (no content) raises at once, and no tile is written."""
        record = _stub_tile_responses(monkeypatch, [_StubTileResponse([], status_code=204)])

        with pytest.raises(requests.HTTPError, match="returned HTTP 204$"):
            self._download(tmp_path / "a-test_dataset.tif")

        assert len(record.requested_urls) == 1
        assert list(tmp_path.iterdir()) == []

    def test_rate_limit_is_retried(self, monkeypatch, tmp_path):
        """HTTP 429 answers are retried; after the last attempt, the error names the status."""
        record = _stub_tile_responses(
            monkeypatch, [_StubTileResponse([], status_code=429) for _ in range(3)]
        )

        with pytest.raises(RuntimeError, match="returned HTTP 429 on attempt 3"):
            self._download(tmp_path / "a-test_dataset.tif")

        assert len(record.requested_urls) == 3
        assert record.waits == [1, 2]
        assert list(tmp_path.iterdir()) == []


class TestHttpErrorText:
    """``_read_http_error_text`` reads a bounded part of the body and gives one short line."""

    def test_body_is_read_only_up_to_4_kib(self):
        """Only the first 4096 bytes of a longer body are read; later text is not in the message."""
        # 10 letters, spaces up to byte 4096, then text after the limit.
        body = b"x" * 10 + b" " * (4096 - 10) + b"after-the-limit"
        response = _StubTileResponse([body], status_code=400)

        error_text = _tiles._read_http_error_text(response)

        assert error_text == "x" * 10
        assert response.bytes_read == 4096

    def test_message_is_one_line_of_at_most_300_characters(self):
        """Line breaks and runs of spaces become one space, and the text is cut at 300 characters."""
        body = b"0123456789\n" * 50
        response = _StubTileResponse([body], status_code=400)

        error_text = _tiles._read_http_error_text(response)

        # The one-line text is "0123456789 " repeated; its first 300 characters.
        assert error_text == ("0123456789 " * 28)[:300]
        assert len(error_text) == 300

    def test_json_message_is_cut_at_300_characters(self):
        """A long Earth Engine JSON message is also cut at 300 characters."""
        body = json.dumps({"error": {"code": 400, "message": "y" * 1000}}).encode()

        error_text = _tiles._read_http_error_text(_StubTileResponse([body], status_code=400))

        assert error_text == "y" * 300


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
        return _StubImageCollection([])

    stub_ee = SimpleNamespace(
        ImageCollection=open_collection, Dictionary=FailingDictionary, **_STUB_TIMESTAMP_EE_NAMES
    )
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

    def test_error_after_the_request_is_not_wrapped(self, monkeypatch):
        """An error in the processing of the response propagates as it is, without RuntimeError."""

        class ResponseWithoutEnds:
            """Stands in for ``ee.Dictionary``. ``getInfo`` returns no ``ends`` list."""

            def __init__(self, values: dict):
                self._values = values

            def getInfo(self) -> dict:
                return {"starts": [_epoch_ms("2021-01-01")]}

        def open_collection(collection_id):
            return _StubImageCollection([])

        stub_ee = SimpleNamespace(
            ImageCollection=open_collection,
            Dictionary=ResponseWithoutEnds,
            **_STUB_TIMESTAMP_EE_NAMES,
        )
        monkeypatch.setattr(_image, "ee", stub_ee)

        with pytest.raises(KeyError, match="ends"):
            _get_collection_time_bounds("users/test/collection")

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
            table, _ = _append_stat_columns(
                pd.DataFrame({"id": _THREE_IDS}), "stub_gee", 200, results
            )
            tables.append(table)
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


class TestBatchWithoutPoints:
    """A batch without points sends no Earth Engine request, also no timestamp request."""

    def _collection_adapter(self) -> GeeRasterAdapter:
        """Return an offline adapter for a collection whose timestamps are not loaded yet.

        A timestamp request would reach the blocked Earth Engine network calls.
        """
        return _tile_export_adapter(
            _dataset_spec={"collection": "users/test/collection", "bands": ["B1", "B2"]},
            _static_image=None,
            _is_collection=True,
            _timestamps_loaded=False,
            _needs_per_point_date=True,
        )

    def test_statistics_batch_without_points(self):
        """``fetch_stats_batch`` returns an empty list and reports progress 0 of 0."""
        adapter = self._collection_adapter()
        progress_steps: list[tuple[int, int]] = []

        results = adapter.fetch_stats_batch(
            [],
            [],
            200,
            ["mean"],
            disable_progress=True,
            progress_callback=lambda completed, total: progress_steps.append((completed, total)),
        )

        assert results == []
        assert progress_steps == [(0, 0)]
        assert adapter._timestamps_loaded is False

    def test_tile_batch_without_points(self, tmp_path):
        """``export_tiles`` returns two empty lists and reports progress 0 of 0."""
        adapter = self._collection_adapter()
        progress_steps: list[tuple[int, int]] = []

        paths, meta_list = adapter.export_tiles(
            [],
            [],
            200,
            tmp_path,
            ids=[],
            dataset_name="test_dataset",
            disable_progress=True,
            progress_callback=lambda completed, total: progress_steps.append((completed, total)),
        )

        assert (paths, meta_list) == ([], [])
        assert progress_steps == [(0, 0)]
        assert adapter._timestamps_loaded is False


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

    def test_failed_tile_is_not_in_the_date_info(self, stub_gee_dataset, tmp_path):
        """The sidecar ``date_info`` of a raster run counts only the tiles that were exported."""
        _stub_tile_exports(stub_gee_dataset, {1})
        # A time index with one image a month, so each tile gets date information.
        stub_gee_dataset._collection_time_starts = pd.DatetimeIndex(
            ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"]
        )
        points = _three_point_frame().assign(eventDate=["2021-01-02", "2021-02-02", "2021-03-02"])

        with pytest.warns(UserWarning, match="1 of 3 points failed"):
            extract(points, _stub_gee_config("gee_tiles", "raster"), output_dir=tmp_path)

        sidecar = json.loads((tmp_path / "gee_tiles" / "gee_tiles_metadata.json").read_text())
        assert sidecar["datasets"]["stub_gee"]["date_info"] == {
            "n_nearest_to_sample": 2,
            "n_contains_sample_date": 0,
            "n_clamped_to_nearest": 0,
            "n_most_recent_no_date": 0,
            "image_date_ranges_used": [
                ["2021-01-01", "2021-01-01"],
                ["2021-03-01", "2021-03-01"],
            ],
        }

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
        table, _ = _append_stat_columns(
            pd.DataFrame({"id": ["p0"]}), "stub_gee", 200, [(stats, meta)]
        )

        assert stats == {"B1_count": 400, "B2_count": 380}
        assert table.loc[0, ["stub_gee_B1_count_200m", "stub_gee_B2_count_200m"]].tolist() == [
            400,
            380,
        ]
        assert (meta["in_extent"], meta["n_pixels"]) == (True, 380)
        assert meta["coverage_pct"] == pytest.approx(95.0)
        assert meta["band_coverage_pct"] == {"B1": 100.0, "B2": 95.0}

    @pytest.mark.parametrize(
        ("bands", "window_result", "expected_stats"),
        [
            ("DEM", {"DEM": 0}, {"count": 0}),
            (["B1", "B2"], {"B1": 0, "B2": 0}, {"B1_count": 0, "B2_count": 0}),
        ],
    )
    def test_zero_counts_give_no_values(self, monkeypatch, bands, window_result, expected_stats):
        """Counts of 0 (no valid pixel in the window) give in_extent False and coverage 0."""
        adapter = _statistics_adapter({"bands": bands})
        _stub_statistics_request(monkeypatch, adapter, {"window": window_result})

        stats, meta = _fetch_point_statistics(adapter, ["count"], ["_count"])

        assert stats == expected_stats
        assert (meta["in_extent"], meta["n_pixels"], meta["coverage_pct"]) == (False, 0, 0.0)

    def test_one_band_with_a_count_above_zero_has_values(self, monkeypatch):
        """With two bands, a count above 0 in one band gives in_extent True."""
        adapter = _statistics_adapter({"bands": ["B1", "B2"]})
        _stub_statistics_request(monkeypatch, adapter, {"window": {"B1": 12, "B2": 0}})

        _, meta = _fetch_point_statistics(adapter, ["count"], ["_count"])

        assert meta["in_extent"] is True


class TestWindowWithoutValidPixels:
    """A window without valid pixels has in_extent False and a null sum; count stays 0."""

    @pytest.mark.parametrize(
        ("bands", "reducer_names", "window_result", "expected_stats"),
        [
            (
                "DEM",
                ["mean", "sum"],
                {"DEM_mean": None, "DEM_sum": 0, "DEM_count": 0},
                {"mean": None, "sum": None},
            ),
            (
                "DEM",
                ["mean", "count"],
                {"DEM_mean": None, "DEM_count": 0},
                {"mean": None, "count": 0},
            ),
            (
                ["B1", "B2"],
                ["mean", "sum"],
                {
                    "B1_mean": None,
                    "B1_sum": 0,
                    "B1_count": 0,
                    "B2_mean": None,
                    "B2_sum": 0,
                    "B2_count": 0,
                },
                {"B1_mean": None, "B2_mean": None, "B1_sum": None, "B2_sum": None},
            ),
        ],
    )
    def test_zero_sum_or_count_is_not_data(
        self, monkeypatch, bands, reducer_names, window_result, expected_stats
    ):
        """Earth Engine's 0 sum of an empty window becomes null; the point is not in the extent."""
        adapter = _statistics_adapter({"bands": bands})
        _stub_statistics_request(monkeypatch, adapter, {"window": window_result})
        suffixes = [f"_{reducer_name}" for reducer_name in reducer_names]

        stats, meta = _fetch_point_statistics(adapter, reducer_names, suffixes)

        assert stats == expected_stats
        assert (meta["in_extent"], meta["n_pixels"], meta["coverage_pct"]) == (False, 0, 0.0)

    def test_one_valid_pixel_is_data(self, monkeypatch):
        """A window with one valid pixel has in_extent True and keeps its sum."""
        adapter = _statistics_adapter({"bands": "DEM"})
        window_result = {"DEM_mean": 250.0, "DEM_sum": 250.0, "DEM_count": 1}
        _stub_statistics_request(monkeypatch, adapter, {"window": window_result})

        stats, meta = _fetch_point_statistics(adapter, ["mean", "sum"], ["_mean", "_sum"])

        assert stats == {"mean": 250.0, "sum": 250.0}
        assert (meta["in_extent"], meta["n_pixels"]) == (True, 1)

    def test_only_the_band_without_valid_pixels_has_a_null_sum(self, monkeypatch):
        """Only the sum of the band with count 0 is null; a sum of 0 with valid pixels stays 0."""
        adapter = _statistics_adapter({"bands": ["B1", "B2"]})
        window_result = {
            "B1_sum": 0.0,
            "B1_count": 25,
            "B2_sum": 0,
            "B2_count": 0,
        }
        _stub_statistics_request(monkeypatch, adapter, {"window": window_result})

        stats, meta = _fetch_point_statistics(adapter, ["sum", "count"], ["_sum", "_count"])

        assert stats == {"B1_sum": 0.0, "B2_sum": None, "B1_count": 25, "B2_count": 0}
        assert (meta["in_extent"], meta["n_pixels"]) == (True, 0)


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

    def updateMask(self, mask: _SymbolicImage | int) -> _SymbolicImage:
        mask_expression = mask.expression if isinstance(mask, _SymbolicImage) else mask
        return _SymbolicImage(f"({self.expression}) masked to {mask_expression}")

    def select(self, band) -> _SymbolicImage:
        return _SymbolicImage(f"{self.expression}.select({band})")

    def projection(self) -> str:
        return f"projection of {self.expression}"

    def setDefaultProjection(self, projection: str) -> _SymbolicImage:
        return _SymbolicImage(f"{self.expression} in {projection}")

    def addBands(self, image: _SymbolicImage) -> _SymbolicImage:
        return _SymbolicImage(f"({self.expression} with {image.expression})")


class _SymbolicNumber:
    """Stands in for an ``ee.Number``. A comparison returns the text of the condition."""

    def __init__(self, expression: str):
        self.expression = expression

    def gt(self, threshold) -> str:
        return f"{self.expression} > {threshold}"


class _SymbolicCollection:
    """Stands in for an ``ee.ImageCollection``, with the operations of a mosaic collection."""

    def __init__(self, expression: str):
        self.expression = expression

    def filterBounds(self, geometry: str) -> _SymbolicCollection:
        return _SymbolicCollection(f"{self.expression} at {geometry}")

    def filter(self, condition: str) -> _SymbolicCollection:
        return _SymbolicCollection(f"{self.expression} where {condition}")

    def filterDate(self, start: str, end: str) -> _SymbolicCollection:
        return _SymbolicCollection(f"{self.expression} from {start} to {end}")

    def limit(self, maximum: int) -> _SymbolicCollection:
        return _SymbolicCollection(f"{self.expression} limited to {maximum}")

    def size(self) -> _SymbolicNumber:
        return _SymbolicNumber(f"size of {self.expression}")

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


class TestEmptyDateSelection:
    """``_build_image`` gives a masked image only when the collection has no image at the point.

    The masked image is the first image of the collection that has every
    configured source band.
    """

    @pytest.fixture(autouse=True)
    def symbolic_earth_engine(self, monkeypatch):
        """Replace ``ee`` in ``_image`` with symbolic stubs, including the server-side ``If``."""

        def cast_to_image(value) -> _SymbolicImage:
            # ee.Image() keeps an image as it is; the text of an If is put in brackets.
            return value if isinstance(value, _SymbolicImage) else _SymbolicImage(f"({value})")

        def server_side_if(condition: str, true_case, false_case) -> str:
            return f"if {condition} then {true_case.expression} else {false_case.expression}"

        stub_ee = SimpleNamespace(
            Image=cast_to_image,
            ImageCollection=_SymbolicCollection,
            Algorithms=SimpleNamespace(If=server_side_if),
            Filter=SimpleNamespace(
                eq=lambda property_name, value: f"{property_name} == {value}",
                listContains=lambda property_name, value: f"{value} in {property_name}",
            ),
        )
        monkeypatch.setattr(_image, "ee", stub_ee)

    @pytest.mark.parametrize(
        ("dataset_spec", "time_starts", "time_ends", "selection"),
        [
            # The interval that contains 2021-06-15, in the point's UTM zone.
            (
                {"collection_date_policy": "contains", "use_utm_zone": True},
                pd.DatetimeIndex(["2021-01-01", "2022-01-01"]),
                pd.DatetimeIndex(["2022-01-01", "2023-01-01"]),
                "embeddings at point where UTM_ZONE == 26N from 2021-01-01 to 2022-01-01",
            ),
            # The nearest start.
            (
                {"collection_date_policy": "nearest"},
                pd.DatetimeIndex(["2021-01-01", "2022-01-01"]),
                None,
                "embeddings at point from 2021-01-01 to 2021-01-02",
            ),
            # No timestamp index: one day before and after the sample date.
            ({}, None, None, "embeddings at point from 2021-06-14 to 2021-06-16"),
        ],
    )
    def test_point_without_images_gets_a_masked_image(
        self, dataset_spec, time_starts, time_ends, selection
    ):
        """The check looks for one image at the point; the image is the first selected one."""
        image = _image._build_image(
            {"collection": "embeddings", "bands": ["A00", "A01"], **dataset_spec},
            date=pd.Timestamp("2021-06-15"),
            geometry="point",
            collection_time_starts=time_starts,
            collection_time_ends=time_ends,
            lat=0.0,
            lon=-30.0,
        )

        # The condition looks for an image at the point before the UTM-zone
        # and date filters. When the point has images, but these filters
        # remove all of them, the first selected image is null, and the
        # point's request fails. The masked image is the first image of the
        # whole collection that has both configured bands.
        assert image.expression == (
            f"(if size of embeddings at point limited to 1 > 0 then first of {selection} "
            f"else (first of embeddings where A00 in system:band_names "
            f"where A01 in system:band_names) masked to 0).select(['A00', 'A01'])"
        )

    @pytest.mark.parametrize(
        ("band_setting", "placeholder", "band_selection"),
        [
            # One configured band, as text.
            (
                {"bands": "A00"},
                "first of embeddings where A00 in system:band_names",
                ".select(['A00'])",
            ),
            # No configured band (all bands): the first image of the collection.
            ({}, "first of embeddings", ""),
            # An empty band list (a per-call list of only derived bands).
            ({"bands": []}, "first of embeddings", ""),
        ],
    )
    def test_masked_image_has_the_configured_bands(self, band_setting, placeholder, band_selection):
        """Only configured source bands filter the collection of the masked image."""
        image = _image._build_image(
            {"collection": "embeddings", "collection_date_policy": "nearest", **band_setting},
            date=pd.Timestamp("2021-06-15"),
            geometry="point",
            collection_time_starts=pd.DatetimeIndex(["2021-01-01", "2022-01-01"]),
        )

        assert image.expression == (
            "(if size of embeddings at point limited to 1 > 0 then first of embeddings "
            f"at point from 2021-01-01 to 2021-01-02 else ({placeholder}) masked to 0)"
            f"{band_selection}"
        )

    def test_no_masked_image_without_a_point(self):
        """Without a point geometry, the image is the first selected image, with no condition."""
        image = _image._build_image(
            {"collection": "embeddings", "collection_date_policy": "nearest", "bands": ["A00"]},
            date=pd.Timestamp("2021-06-15"),
            collection_time_starts=pd.DatetimeIndex(["2021-01-01", "2022-01-01"]),
        )

        assert (
            image.expression == "first of embeddings from 2021-01-01 to 2021-01-02.select(['A00'])"
        )


# ------------------------------------------------------------------
# Per-call band lists — selected bands and statistic names
# ------------------------------------------------------------------


@pytest.fixture
def symbolic_adapter_setup(monkeypatch):
    """Let ``GeeRasterAdapter.__post_init__`` run offline, with symbolic Earth Engine objects.

    Every asset is an image collection here. The adapter skips the Earth
    Engine initialization and the scale check, and ``_build_image`` builds a
    symbolic image whose text shows the bands that it selects.
    """
    monkeypatch.setattr(adapter_module, "_ensure_gee_init", lambda: None)
    monkeypatch.setattr(GeeRasterAdapter, "_validate_auto_detected_scale", lambda self: None)
    stub_data = SimpleNamespace(getAsset=lambda asset_id: {"type": "IMAGE_COLLECTION"})
    monkeypatch.setattr(
        adapter_module, "ee", SimpleNamespace(data=stub_data, ImageCollection=_SymbolicCollection)
    )
    monkeypatch.setattr(_image, "ee", SimpleNamespace(ImageCollection=_SymbolicCollection))
    monkeypatch.setitem(
        _image._DERIVED_BAND_DISPATCH,
        "aspect",
        lambda image: _SymbolicImage(f"aspect({image.expression})"),
    )


def _adapter_for_dataset_entry(dataset_entry) -> tuple[GeeRasterAdapter, str, dict]:
    """Parse one ``datasets`` entry with the built-in catalog, and build its adapter.

    The adapter gets the merged config that ``extract()`` gives it: the
    catalog entry with the band overrides of the entry. Returns the adapter,
    the dataset name, and the band overrides.
    """
    catalog_datasets = load_catalogs((BUILTIN_EE_CATALOG,))["datasets"]
    run_config = {
        "batch_id": "bands",
        "datasets": [dataset_entry],
        "settings": {"output_type": "tabular", "statistics": ["mean"], "window_size_m": 200},
    }
    run_settings = _parse_run_config(run_config, load_defaults(), 0, catalog_datasets)
    ((dataset, band_overrides),) = run_settings.datasets
    adapter = GeeRasterAdapter({**catalog_datasets[dataset], **band_overrides})
    # 30 m pixels, so that the point request needs no scale request.
    adapter.scale = 30
    return adapter, dataset, band_overrides


@pytest.mark.usefixtures("symbolic_adapter_setup")
class TestPerCallBandLists:
    """A per-call band list replaces the catalog's bands and gives band-named statistics."""

    @pytest.mark.parametrize(
        ("dataset_entry", "selected_bands"),
        [
            # Only a derived band: the catalog's source band DEM is not selected.
            ({"dem_copernicus_glo30": ["aspect"]}, ["aspect"]),
            ({"dem_copernicus_glo30": ["DEM", "aspect"]}, ["DEM", "aspect"]),
            ("dem_copernicus_glo30", ["DEM"]),
        ],
    )
    def test_selected_bands(self, dataset_entry, selected_bands):
        """The image of the dataset selects exactly the bands of the list, or the catalog's bands."""
        adapter, _, _ = _adapter_for_dataset_entry(dataset_entry)

        assert adapter._static_image.expression.endswith(f".select({selected_bands})")
        assert adapter._get_band_name(adapter._static_image) == selected_bands[0]
        assert adapter._cached_band_names == selected_bands

    def test_derived_only_list_records_no_source_bands(self):
        """The sidecar records the resolved bands of a derived-only list: no source band."""
        catalog_datasets = load_catalogs((BUILTIN_EE_CATALOG,))["datasets"]
        _, dataset, band_overrides = _adapter_for_dataset_entry(
            {"dem_copernicus_glo30": ["aspect"]}
        )

        assert _resolve_dataset_metadata([(dataset, band_overrides)], catalog_datasets) == [
            {"name": "dem_copernicus_glo30", "bands": [], "derived_bands": ["aspect"]}
        ]

    @pytest.mark.parametrize(
        ("dataset_entry", "window_result", "expected_values"),
        [
            # A per-call list of one source band: the column names the band.
            (
                {"sr_landsat_annual": ["red"]},
                {"red_mean": 0.12, "red_count": 400},
                {"sr_landsat_annual_red_mean_200m": 0.12},
            ),
            (
                {"dem_copernicus_glo30": ["DEM"]},
                {"DEM_mean": 120.0, "DEM_count": 400},
                {"dem_copernicus_glo30_DEM_mean_200m": 120.0},
            ),
            # The catalog default with one band keeps the bare name.
            (
                "dem_copernicus_glo30",
                {"DEM_mean": 120.0, "DEM_count": 400},
                {"dem_copernicus_glo30_mean_200m": 120.0},
            ),
        ],
    )
    def test_statistic_columns(self, monkeypatch, dataset_entry, window_result, expected_values):
        """A per-call list gives band-named columns, also for one band; a catalog default does not."""
        adapter, dataset, _ = _adapter_for_dataset_entry(dataset_entry)
        _stub_statistics_request(monkeypatch, adapter, {"window": window_result})

        stats, meta = _fetch_point_statistics(adapter, ["mean"], ["_mean"])
        table, column_names = _append_stat_columns(
            pd.DataFrame({"id": ["p0"]}), dataset, 200, [(stats, meta)]
        )

        assert {column: table.loc[0, column] for column in column_names} == expected_values
        assert (meta["in_extent"], meta["n_pixels"]) == (True, 400)

    def test_one_band_list_names_point_and_count_after_the_band(self, monkeypatch):
        """With a one-band list, the point and count-only keys also name the band."""
        adapter, _, _ = _adapter_for_dataset_entry({"sr_landsat_annual": ["red"]})
        _stub_statistics_request(
            monkeypatch, adapter, {"window": {"red": 380}, "point": {"red": 0.15}}
        )
        # The point branch also needs Reducer.first and Geometry.Point.
        adapter_module.ee.Reducer.first = lambda: _StubReducer("first")
        adapter_module.ee.Geometry = SimpleNamespace(Point=lambda coordinates: "point")

        stats, meta = adapter._fetch_stats_single(
            59.85, 17.63, 200, _StubReducer("count"), ["count"], ["_count"], want_point=True
        )

        assert stats == {"red_count": 380, "red_point": 0.15}
        assert (meta["in_extent"], meta["n_pixels"]) == (True, 380)

    def test_derived_only_list_gives_circular_aspect_columns(self, monkeypatch):
        """``[aspect]`` gives aspect-named columns, and the circular values replace mean and std."""
        adapter, dataset, _ = _adapter_for_dataset_entry({"dem_copernicus_glo30": ["aspect"]})
        window_result = {
            "aspect_mean": 180.0,
            "aspect_stdDev": 170.0,
            "aspect_max": 359.0,
            "aspect_count": 400,
        }
        recorded_requests = _stub_statistics_request(
            monkeypatch,
            adapter,
            {"window": window_result, "aspect_circular": _NORTH_CIRCULAR_RESULT},
        )

        stats, meta = _fetch_point_statistics(
            adapter, ["mean", "std", "max"], ["_mean", "_stdDev", "_max"]
        )
        table, column_names = _append_stat_columns(
            pd.DataFrame({"id": ["p0"]}), dataset, 200, [(stats, meta)]
        )

        assert column_names == [
            "dem_copernicus_glo30_aspect_mean_200m",
            "dem_copernicus_glo30_aspect_std_200m",
            "dem_copernicus_glo30_aspect_max_200m",
        ]
        circular_mean = table.loc[0, "dem_copernicus_glo30_aspect_mean_200m"]
        assert _distance_from_north_deg(circular_mean) == pytest.approx(0.0, abs=1e-9)
        assert table.loc[0, "dem_copernicus_glo30_aspect_std_200m"] == pytest.approx(_NORTH_STD_DEG)
        assert table.loc[0, "dem_copernicus_glo30_aspect_max_200m"] == 359.0
        # The window branch reduces the image of the list (only aspect), not a selection.
        assert recorded_requests[0]["window"]["image"] == "main"
        assert recorded_requests[0]["aspect_circular"]["image"] == "circular"
