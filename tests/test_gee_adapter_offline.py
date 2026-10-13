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

import hashlib
import importlib
import json
import math
import warnings
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

# The package exports the function extract() under the module's name, so get
# the module itself from the import system, for its private helpers.
_extract_module = importlib.import_module("envoi.extract")


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
        assert adapter._resolve_date_info("2021-01-05", _image_times("2021-01-01")) == {
            "date_clamped": False,
            "date_source": "nearest_to_sample",
            "image_time_start": "2021-01-01",
        }
        assert _resolve_date_filter_range(
            pd.Timestamp("2021-01-05"), "nearest", start_index, end_index
        ) == ("2021-01-01", "2021-01-02")

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


def _image_times(start_text: str, end_text: str | None = None) -> dict:
    """Return the ``image_dates`` result of a point's request for an image with these times."""
    return {
        "start": _epoch_ms(start_text),
        "end": _epoch_ms(end_text) if end_text is not None else None,
    }


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

        date_info = adapter._resolve_date_info(sample_date, _image_times(image_start, image_end))

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

        date_info = adapter._resolve_date_info(sample_date, _image_times(image_start))

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

        first_image_times = _image_times("2021-06-01", "2021-06-01")
        last_image_times = _image_times("2021-06-15", "2021-06-15")
        assert adapter._resolve_date_info("2021-06-01", first_image_times) == {
            "date_clamped": False,
            "date_source": "contains_sample_date",
            "image_time_start": "2021-06-01",
            "image_time_end": "2021-06-01",
        }
        # The filterDate range covers the start day of the last image, so this
        # date selects that image, and is not clamped.
        assert adapter._resolve_date_info("2021-06-15", last_image_times) == {
            "date_clamped": False,
            "date_source": "contains_sample_date",
            "image_time_start": "2021-06-15",
            "image_time_end": "2021-06-15",
        }
        assert _resolve_date_filter_range(
            pd.Timestamp("2021-06-15"), "contains", time_starts, time_starts.copy()
        ) == ("2021-06-15", "2021-06-16")
        # The next day is after the last interval.
        assert adapter._resolve_date_info("2021-06-16", last_image_times) == {
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
        last_image_times = _image_times("2021-06-15 00:00", "2021-06-15 06:00")

        assert adapter._resolve_date_info("2021-06-15", last_image_times)["date_clamped"] is False
        assert adapter._resolve_date_info("2021-06-16", last_image_times)["date_clamped"] is True

    def test_one_day_intervals(self):
        """With one-day intervals, a date on the last day is not clamped and the next day is."""
        time_starts = pd.DatetimeIndex(["2021-06-14", "2021-06-15"])
        time_ends = pd.DatetimeIndex(["2021-06-15", "2021-06-16"])
        adapter = _contains_adapter(time_starts, time_ends)
        last_image_times = _image_times("2021-06-15", "2021-06-16")

        assert adapter._resolve_date_info("2021-06-15", last_image_times)["date_clamped"] is False
        assert adapter._resolve_date_info("2021-06-16", last_image_times)["date_clamped"] is True

    def test_last_start_later_on_its_day_without_end_times(self):
        """Without end times, the last interval ends at the end of the last start's day."""
        time_starts = pd.DatetimeIndex(["2021-06-07 10:00", "2021-06-15 10:00"])
        adapter = _contains_adapter(time_starts, None)
        last_image_times = _image_times("2021-06-15 10:00")

        assert adapter._resolve_date_info("2021-06-15", last_image_times) == {
            "date_clamped": False,
            "date_source": "contains_sample_date",
            "image_time_start": "2021-06-15",
        }
        assert adapter._resolve_date_info("2021-06-16", last_image_times)["date_clamped"] is True

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
        image_times = _image_times(f"{image_start} 10:00", f"{image_end} 10:00")

        assert adapter._resolve_date_info(sample_date, image_times) == {
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
        return output_path, {}

    adapter._export_single = record_export
    return exported_points


def _export_two_tiles(
    adapter: GeeRasterAdapter, output_dir, window_m: int, resample_m=None, *, ids=("a", "b")
):
    """Call ``export_tiles`` for two points near Uppsala, with the dataset name ``test_dataset``."""
    return adapter.export_tiles(
        [59.85, 59.86],
        [17.63, 17.64],
        window_m,
        output_dir,
        ids=list(ids),
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


class TestCaseCollisionTileNames:
    """IDs that differ only in letter case get two tile files on Earth Engine, as on local rasters."""

    def test_ids_that_differ_only_in_letter_case_get_two_files(self, tmp_path):
        """``Site1`` and ``site1`` get two hash-suffixed names, and both tiles are written."""
        adapter = _tile_export_adapter()

        def write_tile(lat, lon, window_m, output_path, date=None, resample_m=None):
            output_path.write_bytes(f"tile at {lat}".encode())
            return output_path, {}

        adapter._export_single = write_tile

        paths, meta_list = _export_two_tiles(
            adapter, tmp_path, window_m=200, ids=["Site1", "site1"]
        )

        # The suffix is the first 8 hex characters of the SHA-1 of the ID.
        expected_names = [
            f"{tile_id}-{hashlib.sha1(tile_id.encode('utf-8')).hexdigest()[:8]}-test_dataset.tif"
            for tile_id in ("Site1", "site1")
        ]
        assert paths == [tmp_path / "test_dataset" / name for name in expected_names]
        assert paths[0].name.casefold() != paths[1].name.casefold()
        assert sorted(path.name for path in (tmp_path / "test_dataset").iterdir()) == sorted(
            expected_names
        )
        assert [path.read_bytes() for path in paths] == [b"tile at 59.85", b"tile at 59.86"]
        assert meta_list == [{}, {}]


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
        # An image asset: no per-point date selection, so no date request.
        adapter = _make_offline_adapter(
            _get_image=lambda date=None, lat=None, lon=None: (stub_image, None)
        )
        recorded_downloads = _record_tile_downloads(monkeypatch)
        output_path = tmp_path / "tiles" / "a-test_dataset.tif"

        returned_path, date_info = adapter._export_single(
            self._LAT, self._LON, 200, output_path, None, resample_m=30
        )

        point_x, point_y = self._point_utm()
        # The point's pixel starts at floor(x / 30) · 30 (west edge) and ends at
        # (floor(y / 30) + 1) · 30 (north edge). Three pixels lie on each side.
        expected_west = (math.floor(point_x / 30) - 3) * 30
        expected_north = (math.floor(point_y / 30) + 1 + 3) * 30
        assert returned_path == output_path
        assert date_info == {}
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
            _get_image=lambda date=None, lat=None, lon=None: (stub_image, None),
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
    returns its path and its date fields: the image of a point with a sample
    date starts on that date and has no end. Returns a dict that maps the
    index of each failed point to its error.
    """
    raised_errors: dict[int, Exception] = {}

    def export(lat, lon, window_m, output_path, date=None, resample_m=None):
        point_index = _THREE_LATS.index(lat)
        if failing_points is None or point_index in failing_points:
            error = _StubEarthEngineError(f"stub: download failed for point {point_index}")
            raised_errors[point_index] = error
            raise error
        if date is None:
            return output_path, {}
        return output_path, adapter._resolve_date_info(
            date, {"start": _epoch_ms(date), "end": None}
        )

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

    ``extract()`` gets the class ``PreparedStubAdapter`` through its
    ``get_adapter`` lookup. The class has the capability declarations and the
    pre-flight check of ``GeeRasterAdapter``, and each build gives the prepared
    offline adapter, so the test runs the real per-point result loops of the
    Earth Engine adapter. Returns the adapter, so the test can stub its
    per-point requests.
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

    class PreparedStubAdapter(GeeRasterAdapter):
        """Gives the prepared offline adapter for each build."""

        def __new__(cls, spec):
            # The adapter is not an instance of this subclass, so Python does
            # not call __init__ on it, and __post_init__ sends no request.
            return adapter

    # The package exports the function extract() under the module's name, so
    # get the module itself from the import system.
    extract_module = importlib.import_module("envoi.extract")
    monkeypatch.setattr(extract_module, "get_adapter", lambda data_source: PreparedStubAdapter)
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
                ["2021-01-02", "2021-01-02"],
                ["2021-03-02", "2021-03-02"],
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
# One adapter for every window size of a dataset
# ------------------------------------------------------------------

# The time index that the stub timestamp request returns: one image a month.
_MONTHLY_TIME_STARTS = pd.DatetimeIndex(["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"])

# The (lon, lat) pairs of the three stub points, as the timestamp request gets them.
_THREE_LON_LAT_PAIRS = [[lon, lat] for lat, lon in zip(_THREE_LATS, _THREE_LONS)]


def _stub_window_statistics_requests(adapter: GeeRasterAdapter) -> None:
    """Replace the per-point statistics request of ``adapter`` with a window-dependent stub.

    Point i gets ``B1_mean`` = window_m + i and ``B2_mean`` = 2 × window_m + i,
    the pixel count of a window of 10 m pixels, and the date fields of an
    image that starts on the point's sample date. ``_resolve_date_info``
    computes the date fields from the adapter's timestamp index, so a point
    gets them only when the index is loaded.
    """

    def request(lat, lon, window_m, *args, date=None, **kwargs):
        point_index = _THREE_LATS.index(lat)
        stats = {
            "B1_mean": float(window_m + point_index),
            "B2_mean": float(2 * window_m + point_index),
        }
        meta = {
            "in_extent": True,
            "n_pixels": (window_m // 10) ** 2,
            "had_nodata": False,
            "coverage_pct": 100.0,
            "window_m": window_m,
            **adapter._resolve_date_info(date, {"start": _epoch_ms(date), "end": None}),
        }
        return stats, meta

    adapter._fetch_stats_single = request


@pytest.fixture
def counting_stub_gee_collection(monkeypatch, stub_combined_reducer) -> SimpleNamespace:
    """Register the Earth Engine collection ``stub_gee``, whose adapter class records each build.

    ``extract()`` gets the class ``CountingStubAdapter`` through its
    ``get_adapter`` lookup. Each build runs the constructor of
    ``GeeRasterAdapter`` with a setup that sends no request: the attributes of
    a two-band collection at 10 m whose timestamps are not loaded yet, and the
    stubs of the per-point statistics requests and tile exports. The batch
    methods, the timestamp load on the first batch, and the caches of the
    adapter are the real ones. The timestamp request is a stub that returns
    ``_MONTHLY_TIME_STARTS``.

    Returns the records: ``built_adapters`` (each built adapter, in build
    order) and ``timestamp_requests`` (the collection and the point
    coordinates of each timestamp request).
    """
    update_catalog(
        {
            "datasets": {
                "stub_gee": {
                    "data_source": "earth_engine",
                    "path": "users/test/collection",
                    "data_type": "continuous",
                    "bands": ["B1", "B2"],
                }
            }
        }
    )
    records = SimpleNamespace(built_adapters=[], timestamp_requests=[])

    def request_timestamps(collection_id, bounds_geometry=None):
        records.timestamp_requests.append((collection_id, bounds_geometry))
        return _MONTHLY_TIME_STARTS, None

    # The timestamp load builds the geometry of the batch's points with
    # ee.Geometry.MultiPoint, which would load the API from the server. The
    # stub gives the coordinate pairs themselves.
    monkeypatch.setattr(adapter_module, "_get_collection_time_bounds", request_timestamps)
    monkeypatch.setattr(
        adapter_module,
        "ee",
        SimpleNamespace(Geometry=SimpleNamespace(MultiPoint=lambda point_pairs: point_pairs)),
    )

    class CountingStubAdapter(GeeRasterAdapter):
        """Records each build, and sets up a collection adapter without a request."""

        def __post_init__(self):
            records.built_adapters.append(self)
            prepared_adapter = _tile_export_adapter(
                _dataset_spec={"collection": "users/test/collection", "bands": ["B1", "B2"]},
                _is_collection=True,
                _timestamps_loaded=False,
                _needs_per_point_date=True,
            )
            # Take the attributes of the prepared offline adapter. The
            # constructor has already set `spec`, which the prepared adapter
            # does not have.
            vars(self).update(vars(prepared_adapter))
            _stub_window_statistics_requests(self)
            _stub_tile_exports(self, set())

    monkeypatch.setattr(_extract_module, "get_adapter", lambda data_source: CountingStubAdapter)
    return records


def _dated_three_point_frame() -> pd.DataFrame:
    """The three stub points, with a sample date in each month of the stub time index."""
    return _three_point_frame().assign(eventDate=["2021-01-02", "2021-02-02", "2021-03-02"])


def _stub_gee_window_config(batch_id: str, output_type: str, window_size_m) -> dict:
    """Return a run config for ``stub_gee`` with the window size or sizes ``window_size_m``."""
    run_config = _stub_gee_config(batch_id, output_type)
    run_config["settings"]["window_size_m"] = window_size_m
    return run_config


def _dataset_entry_without_quality(sidecar_path) -> dict:
    """Return the sidecar entry of ``stub_gee`` without its ``quality`` key."""
    dataset_entry = json.loads(sidecar_path.read_text())["datasets"]["stub_gee"]
    return {key: value for key, value in dataset_entry.items() if key != "quality"}


class TestOneAdapterForEveryWindow:
    """``extract()`` builds one Earth Engine adapter for every window size of a dataset.

    The second window uses the timestamp index that the first window loaded,
    and the results of each window equal those of a run with only that window.
    """

    def test_gee_multi_window_builds_one_adapter(self, counting_stub_gee_collection, tmp_path):
        """Tabular: one build and one timestamp request; each window equals a single-window run."""
        records = counting_stub_gee_collection
        points = _dated_three_point_frame()

        multi_window_outputs = extract(
            points,
            _stub_gee_window_config("multi", "tabular", [100, 200]),
            output_dir=tmp_path,
            quiet=True,
        )

        assert len(records.built_adapters) == 1
        assert records.timestamp_requests == [("users/test/collection", _THREE_LON_LAT_PAIRS)]
        multi_window_stats = pd.read_csv(multi_window_outputs["multi"])
        multi_window_qc = pd.read_csv(tmp_path / "multi_qc.csv")
        multi_window_quality = json.loads((tmp_path / "multi_metadata.json").read_text())[
            "datasets"
        ]["stub_gee"]["quality"]
        # The stub values depend on the window: B1_mean = window + point index.
        assert multi_window_stats["stub_gee_B1_mean_100m"].tolist() == [100.0, 101.0, 102.0]
        assert multi_window_stats["stub_gee_B1_mean_200m"].tolist() == [200.0, 201.0, 202.0]
        # The second window has the date fields, so it used the loaded index.
        assert multi_window_qc["stub_gee_date_source_200m"].tolist() == ["nearest_to_sample"] * 3

        for window_size in (100, 200):
            batch_id = f"single_{window_size}"
            single_window_outputs = extract(
                points,
                _stub_gee_window_config(batch_id, "tabular", window_size),
                output_dir=tmp_path,
                quiet=True,
            )

            # The statistic and QC columns of the window, and its quality
            # summary, equal those of the single-window run.
            single_window_stats = pd.read_csv(single_window_outputs[batch_id])
            single_window_qc = pd.read_csv(tmp_path / f"{batch_id}_qc.csv")
            pd.testing.assert_frame_equal(
                multi_window_stats[single_window_stats.columns], single_window_stats
            )
            pd.testing.assert_frame_equal(
                multi_window_qc[single_window_qc.columns], single_window_qc
            )
            single_window_quality = json.loads(
                (tmp_path / f"{batch_id}_metadata.json").read_text()
            )["datasets"]["stub_gee"]["quality"]
            window_key = f"{window_size}m"
            assert multi_window_quality[window_key] == single_window_quality[window_key]
            assert _dataset_entry_without_quality(
                tmp_path / "multi_metadata.json"
            ) == _dataset_entry_without_quality(tmp_path / f"{batch_id}_metadata.json")

        # Each single-window run builds its own adapter.
        assert len(records.built_adapters) == 3

    def test_gee_multi_window_tiles_build_one_adapter(self, counting_stub_gee_collection, tmp_path):
        """Raster: one build and one timestamp request; each window equals a single-window run."""
        records = counting_stub_gee_collection
        points = _dated_three_point_frame()

        extract(
            points,
            _stub_gee_window_config("multi", "raster", [100, 200]),
            output_dir=tmp_path,
            quiet=True,
        )

        assert len(records.built_adapters) == 1
        assert records.timestamp_requests == [("users/test/collection", _THREE_LON_LAT_PAIRS)]
        multi_window_sidecar_path = tmp_path / "multi" / "multi_metadata.json"
        multi_window_quality = json.loads(multi_window_sidecar_path.read_text())["datasets"][
            "stub_gee"
        ]["quality"]

        for window_size in (100, 200):
            batch_id = f"single_{window_size}"
            extract(
                points,
                _stub_gee_window_config(batch_id, "raster", window_size),
                output_dir=tmp_path,
                quiet=True,
            )

            # The manifest of the window names tiles with the window suffix,
            # and otherwise equals the manifest of the single-window run.
            multi_window_manifest = pd.read_csv(
                tmp_path / "multi" / "stub_gee" / f"tiles_manifest-{window_size}m.csv"
            )
            single_window_manifest = pd.read_csv(
                tmp_path / batch_id / "stub_gee" / "tiles_manifest.csv"
            )
            assert multi_window_manifest["tile_filename"].tolist() == [
                f"{sample_id}-stub_gee-{window_size}m.tif" for sample_id in _THREE_IDS
            ]
            pd.testing.assert_frame_equal(
                multi_window_manifest.drop(columns="tile_filename"),
                single_window_manifest.drop(columns="tile_filename"),
            )

            # The tile summary of the window, and the other fields of the
            # dataset entry (with date_info), equal those of the single-window run.
            single_window_sidecar_path = tmp_path / batch_id / f"{batch_id}_metadata.json"
            single_window_quality = json.loads(single_window_sidecar_path.read_text())["datasets"][
                "stub_gee"
            ]["quality"]
            assert multi_window_quality[f"{window_size}m"] == single_window_quality
            assert _dataset_entry_without_quality(
                multi_window_sidecar_path
            ) == _dataset_entry_without_quality(single_window_sidecar_path)

        # Each single-window run builds its own adapter.
        assert len(records.built_adapters) == 3


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
    ``main``, or ``circular`` for the circular aspect image, without a date
    request (``image_dates`` is None, as for an image asset). Returns a list
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
        _StubStatisticsImage("circular" if circular_aspect else "main"),
        None,
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
            return f"image {len(builds)}", None

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

        circular_image, _ = adapter._get_image(None, lat=59.85, lon=17.63, circular_aspect=True)

        assert circular_image == "image 1"
        assert recorded_builds[0]["circular_aspect"] is True
        assert recorded_builds[0]["date"] is None
        assert adapter._static_image == "static image"
        assert adapter._get_image(None, lat=59.85, lon=17.63) == ("static image", None)

    def test_most_recent_fallback_is_not_cached(self, recorded_builds):
        """Without coordinates, the circular image of a collection is not cached; the main one is."""
        adapter = _make_offline_adapter(
            _dataset_spec={"collection": "users/test/collection", "derived_bands": ["aspect"]},
            _static_image=None,
            _needs_per_point_date=True,
            _collection_time_starts=pd.DatetimeIndex(["2021-01-01", "2021-02-01"]),
            _collection_time_ends=None,
        )

        assert adapter._get_image(circular_aspect=True) == ("image 1", None)
        assert adapter._static_image is None
        assert adapter._get_image() == ("image 2", None)
        assert adapter._static_image == "image 2"
        assert adapter._get_image(circular_aspect=True) == ("image 3", None)
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

    def get(self, property_name: str) -> str:
        return f"{self.expression}.{property_name}"


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
        circular_image, image_dates = _image._build_image(
            {"image": "dem", "bands": ["DEM"], "derived_bands": ["slope", "aspect"]},
            circular_aspect=True,
        )

        assert circular_image.expression == (
            "(cat(sin(aspect(dem) * 0.017453) as sin, cos(aspect(dem) * 0.017453) as cos))"
            " masked to slope(dem) > 0"
        )
        assert image_dates is None

    def test_mosaic_collection_keeps_the_native_projection(self):
        """For a space-tiled collection, the circular image uses the projected mosaic."""
        circular_image, image_dates = _image._build_image(
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
        assert image_dates is None


def _symbolic_text(value) -> str:
    """Return the text of a symbolic value: an image, a dictionary, null (None), or text."""
    if isinstance(value, _SymbolicImage):
        return value.expression
    if value is None:
        return "null"
    return str(value)


class TestEmptyDateSelection:
    """``_build_image`` gives a masked image only when the collection has no image at the point.

    The masked image is the first image of the collection that has every
    configured source band. The date request reads the selected image before
    this step, and is null for the masked image.
    """

    @pytest.fixture(autouse=True)
    def symbolic_earth_engine(self, monkeypatch):
        """Replace ``ee`` in ``_image`` with symbolic stubs, including the server-side ``If``.

        The per-point fallback is a symbolic image whose text shows its
        arguments; ``TestPerPointFallback`` tests its rules.
        """

        def cast_to_image(value) -> _SymbolicImage:
            # ee.Image() keeps an image as it is; the text of an If is put in brackets.
            return value if isinstance(value, _SymbolicImage) else _SymbolicImage(f"({value})")

        def server_side_if(condition: str, true_case, false_case) -> str:
            return (
                f"if {condition} then {_symbolic_text(true_case)} "
                f"else {_symbolic_text(false_case)}"
            )

        def symbolic_dictionary(values: dict) -> str:
            return "{" + ", ".join(f"{key}: {value}" for key, value in values.items()) + "}"

        def symbolic_fallback(own_images, centre_day, policy) -> _SymbolicImage:
            return _SymbolicImage(
                f"fallback({own_images.expression}, {centre_day:%Y-%m-%d}, {policy})"
            )

        stub_ee = SimpleNamespace(
            Image=cast_to_image,
            ImageCollection=_SymbolicCollection,
            Algorithms=SimpleNamespace(If=server_side_if),
            Dictionary=symbolic_dictionary,
            Filter=SimpleNamespace(
                eq=lambda property_name, value: f"{property_name} == {value}",
                listContains=lambda property_name, value: f"{value} in {property_name}",
            ),
        )
        monkeypatch.setattr(_image, "ee", stub_ee)
        monkeypatch.setattr(_image, "_build_fallback_image", symbolic_fallback)

    @pytest.mark.parametrize(
        ("dataset_spec", "time_starts", "time_ends", "fast_path", "fallback"),
        [
            # The interval that contains 2021-06-15, in the point's UTM zone.
            # The range ends at the end of the centre day, not at the interval
            # end (2022-01-01). Without an own image in it: the fallback.
            (
                {"collection_date_policy": "contains", "use_utm_zone": True},
                pd.DatetimeIndex(["2021-01-01", "2022-01-01"]),
                pd.DatetimeIndex(["2022-01-01", "2023-01-01"]),
                "embeddings at point where UTM_ZONE == 26N from 2021-01-01 to 2021-06-16",
                "fallback(embeddings at point where UTM_ZONE == 26N, 2021-06-15, contains)",
            ),
            # The nearest start. Without an own image on that day: the fallback.
            (
                {"collection_date_policy": "nearest"},
                pd.DatetimeIndex(["2021-01-01", "2022-01-01"]),
                None,
                "embeddings at point from 2021-01-01 to 2021-01-02",
                "fallback(embeddings at point, 2021-06-15, nearest)",
            ),
            # No timestamp index: one day before and after the sample date, no
            # fallback, and no date request.
            ({}, None, None, "embeddings at point from 2021-06-14 to 2021-06-16", None),
        ],
    )
    def test_point_without_images_gets_a_masked_image(
        self, dataset_spec, time_starts, time_ends, fast_path, fallback
    ):
        """The check looks for one image at the point; the image is the selected one."""
        image, image_dates = _image._build_image(
            {"collection": "embeddings", "bands": ["A00", "A01"], **dataset_spec},
            date=pd.Timestamp("2021-06-15"),
            geometry="point",
            collection_time_starts=time_starts,
            collection_time_ends=time_ends,
            lat=0.0,
            lon=-30.0,
        )

        # The fast path: the first own image in the batch-selected range. When
        # the range has no own image: the per-point fallback.
        fast_path_has_image = f"size of {fast_path} limited to 1 > 0"
        if fallback is None:
            selection = f"first of {fast_path}"
        else:
            selection = f"(if {fast_path_has_image} then first of {fast_path} else {fallback})"

        # The condition looks for an image at the point before the UTM-zone
        # and date filters. When the point has images, but no own image is
        # selected, the selected image is null, and the point's request fails.
        # The masked image is the first image of the whole collection that has
        # both configured bands.
        point_has_images = "size of embeddings at point limited to 1 > 0"
        assert image.expression == (
            f"(if {point_has_images} then {selection} "
            f"else (first of embeddings where A00 in system:band_names "
            f"where A01 in system:band_names) masked to 0).select(['A00', 'A01'])"
        )
        # The date request reads the selected image, before the masked image,
        # with the same condition: null for a point without images. Its
        # fallback flag uses the condition of the fast path: True only when
        # the fast-path range has no own image, so the fallback selected it.
        if time_starts is None:
            assert image_dates is None
        else:
            assert image_dates == (
                f"if {point_has_images} then {{start: {selection}.system:time_start, "
                f"end: {selection}.system:time_end, "
                f"fallback: if {fast_path_has_image} then False else True}} else null"
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
        image, _ = _image._build_image(
            {"collection": "embeddings", "collection_date_policy": "nearest", **band_setting},
            date=pd.Timestamp("2021-06-15"),
            geometry="point",
            collection_time_starts=pd.DatetimeIndex(["2021-01-01", "2022-01-01"]),
        )

        fast_path = "embeddings at point from 2021-01-01 to 2021-01-02"
        assert image.expression == (
            f"(if size of embeddings at point limited to 1 > 0 then (if size of {fast_path} "
            f"limited to 1 > 0 then first of {fast_path} else fallback(embeddings at point, "
            f"2021-06-15, nearest)) else ({placeholder}) masked to 0){band_selection}"
        )

    def test_no_masked_image_without_a_point(self):
        """Without a point geometry: the first selected image, no condition, no fallback, no dates."""
        image, image_dates = _image._build_image(
            {"collection": "embeddings", "collection_date_policy": "nearest", "bands": ["A00"]},
            date=pd.Timestamp("2021-06-15"),
            collection_time_starts=pd.DatetimeIndex(["2021-01-01", "2022-01-01"]),
        )

        assert (
            image.expression == "first of embeddings from 2021-01-01 to 2021-01-02.select(['A00'])"
        )
        assert image_dates is None


# ------------------------------------------------------------------
# Per-point fallback — the selection rules, evaluated by a simulated server
# ------------------------------------------------------------------


class _NullOnServer:
    """Stands in for a null value on the Earth Engine server (a null image).

    Every operation on null gives null again. Earth Engine evaluates only the
    branch of ``If`` that it needs, but the stubs evaluate both branches, so
    null fails only when a result is read (``_read_server_value``), as a
    request that reads a null image fails.
    """

    def get(self, property_name: str) -> _NullOnServer:
        return self

    def set(self, property_name: str, value) -> _NullOnServer:
        return self

    def select(self, bands) -> _NullOnServer:
        return self

    def updateMask(self, mask) -> _NullOnServer:
        return self


_SERVER_NULL = _NullOnServer()


class _SimulatedNumber:
    """Stands in for an ``ee.Number`` and computes the value at once."""

    def __init__(self, value):
        self.value = value.value if isinstance(value, _SimulatedNumber) else value

    def gt(self, other) -> bool:
        return self.value > other

    def divide(self, other) -> _SimulatedNumber:
        return _SimulatedNumber(self.value / other)

    def floor(self) -> _SimulatedNumber:
        return _SimulatedNumber(math.floor(self.value))

    def subtract(self, other) -> _SimulatedNumber:
        return _SimulatedNumber(self.value - _SimulatedNumber(other).value)

    def abs(self) -> _SimulatedNumber:
        return _SimulatedNumber(abs(self.value))

    def multiply(self, other) -> _SimulatedNumber:
        return _SimulatedNumber(self.value * other)

    def add(self, other) -> _SimulatedNumber:
        return _SimulatedNumber(self.value + _SimulatedNumber(other).value)


class _SimulatedImage:
    """Stands in for one ``ee.Image``: a name, the points in its footprint, and its properties."""

    def __init__(self, name: str, points: set[str], properties: dict, masked: bool = False):
        self.name = name
        self.points = points
        self.properties = properties
        self.masked = masked

    def get(self, property_name: str):
        return self.properties.get(property_name)

    def set(self, property_name: str, value) -> _SimulatedImage:
        properties = {**self.properties, property_name: _SimulatedNumber(value).value}
        return _SimulatedImage(self.name, self.points, properties, self.masked)

    def updateMask(self, mask) -> _SimulatedImage:
        return _SimulatedImage(self.name, self.points, self.properties, masked=True)

    def select(self, bands) -> _SimulatedImage:
        return self


def _server_milliseconds(value) -> int:
    """Return a ``filterDate`` argument (milliseconds, or ``YYYY-MM-DD`` text) in milliseconds."""
    return value if isinstance(value, int) else _epoch_ms(value)


class _SimulatedCollection:
    """Stands in for an ``ee.ImageCollection`` of simulated images, and filters them at once."""

    def __init__(self, images):
        self.images = list(images)

    def filterBounds(self, point: str) -> _SimulatedCollection:
        return _SimulatedCollection(image for image in self.images if point in image.points)

    def filter(self, condition) -> _SimulatedCollection:
        return _SimulatedCollection(image for image in self.images if condition(image))

    def filterDate(self, start, end) -> _SimulatedCollection:
        start_ms, end_ms = _server_milliseconds(start), _server_milliseconds(end)
        return self.filter(
            lambda image: image.get("system:time_start") is not None
            and start_ms <= image.get("system:time_start") < end_ms
        )

    def limit(self, maximum: int) -> _SimulatedCollection:
        return _SimulatedCollection(self.images[:maximum])

    def size(self) -> _SimulatedNumber:
        return _SimulatedNumber(len(self.images))

    def first(self):
        return self.images[0] if self.images else _SERVER_NULL

    def map(self, function) -> _SimulatedCollection:
        return _SimulatedCollection(function(image) for image in self.images)

    def sort(self, property_name: str, ascending: bool = True) -> _SimulatedCollection:
        # Not stable on purpose: images with equal values come in the reverse
        # order of the collection. A rule that relies on the order before the
        # sort then gives the wrong image.
        direction = 1 if ascending else -1
        positioned_images = list(enumerate(self.images))
        positioned_images.sort(key=lambda pair: (direction * pair[1].get(property_name), -pair[0]))
        return _SimulatedCollection(image for _, image in positioned_images)


def _simulated_property_filter(compare):
    """Return a stand-in for ``ee.Filter.lt`` and similar; a missing property never matches."""

    def property_filter(property_name: str, value):
        return lambda image: image.get(property_name) is not None and compare(
            image.get(property_name), value
        )

    return property_filter


def _read_server_value(value):
    """Return what ``getInfo()`` gives for a simulated value; raise for a null image.

    An image reads as its name, with `` (masked)`` for the masked image.
    """
    if value is _SERVER_NULL:
        raise _StubEarthEngineError("Parameter 'image' is required and may not be null.")
    if isinstance(value, dict):
        return {key: _read_server_value(item) for key, item in value.items()}
    if isinstance(value, _SimulatedImage):
        return value.name + (" (masked)" if value.masked else "")
    return value


def _simulated_image(
    name: str, start_text: str, end_text: str | None = None, *, points: str, utm_zone=None
) -> _SimulatedImage:
    """Return a simulated image with band B4 at the points named by the letters of ``points``."""
    properties = {"system:time_start": _epoch_ms(start_text), "system:band_names": ["B4"]}
    if end_text is not None:
        properties["system:time_end"] = _epoch_ms(end_text)
    if utm_zone is not None:
        properties["UTM_ZONE"] = utm_zone
    return _SimulatedImage(name, set(points), properties)


def _batch_index(images: list[_SimulatedImage]):
    """Return the batch index (starts, ends) of the images, as ``_get_collection_time_bounds`` does."""
    starts = [image.get("system:time_start") for image in images]
    ends = [image.get("system:time_end") for image in images]
    if any(end is None for end in ends):
        unique_starts = sorted(set(starts))
        return pd.DatetimeIndex(pd.to_datetime(unique_starts, unit="ms")), None
    times = pd.DataFrame({"start": starts, "end": ends}).groupby("start", as_index=False).max()
    return (
        pd.DatetimeIndex(pd.to_datetime(times["start"], unit="ms")),
        pd.DatetimeIndex(pd.to_datetime(times["end"], unit="ms")),
    )


@pytest.fixture
def simulated_server(monkeypatch):
    """Replace ``ee`` in ``_image`` with a simulated server; return a function that sets its images.

    The aspect and slope functions record the image they read, so a test can
    see which image the circular aspect image comes from.
    """
    collection_images: list[_SimulatedImage] = []

    def cast_to_image(value):
        return _SERVER_NULL if value is None else value

    cast_to_image.cat = _concatenate_symbolic_images  # ee.Image.cat
    simulated_ee = SimpleNamespace(
        ImageCollection=lambda collection_id: _SimulatedCollection(collection_images),
        Image=cast_to_image,
        Number=_SimulatedNumber,
        Dictionary=dict,
        Algorithms=SimpleNamespace(
            If=lambda condition, true_case, false_case: true_case if condition else false_case
        ),
        Filter=SimpleNamespace(
            eq=_simulated_property_filter(lambda left, right: left == right),
            lt=_simulated_property_filter(lambda left, right: left < right),
            gt=_simulated_property_filter(lambda left, right: left > right),
            gte=_simulated_property_filter(lambda left, right: left >= right),
            And=lambda *conditions: lambda image: all(match(image) for match in conditions),
            Or=lambda *conditions: lambda image: any(match(image) for match in conditions),
            listContains=lambda property_name, value: lambda image: value
            in (image.get(property_name) or []),
        ),
    )
    monkeypatch.setattr(_image, "ee", simulated_ee)
    for derived_band in ("aspect", "slope"):
        monkeypatch.setitem(
            _image._DERIVED_BAND_DISPATCH,
            derived_band,
            lambda image, derived_band=derived_band: _SymbolicImage(
                f"{derived_band}({_read_server_value(image)})"
            ),
        )

    def set_images(images: list[_SimulatedImage]) -> None:
        collection_images[:] = images

    return set_images


def _select_point_image(
    images: list[_SimulatedImage],
    point: str,
    sample_date: str | None,
    policy: str,
    *,
    batch_points: str,
    **build_arguments,
):
    """Run ``_build_image`` for one point of a batch on the simulated server.

    The batch index has the images of the ``batch_points`` (one letter for
    each point). Without a sample date, the date is the latest start of the
    index, as ``_get_image`` passes it. Returns ``(image, image_dates)`` as
    the server values; read them with ``_read_server_value``.
    """
    batch_images = [image for image in images if image.points & set(batch_points)]
    time_starts, time_ends = _batch_index(batch_images)
    date = time_starts.max() if sample_date is None else pd.Timestamp(sample_date)
    return _image._build_image(
        {"collection": "simulated", "bands": ["B4"], "collection_date_policy": policy},
        date=date,
        geometry=point,
        collection_time_starts=time_starts,
        collection_time_ends=time_ends,
        **build_arguments,
    )


def _image_dates_of(image: _SimulatedImage, *, fallback: bool) -> dict:
    """Return the ``image_dates`` result that Earth Engine gives for ``image``.

    ``fallback`` is True when the per-point fallback selected the image.
    """
    return {
        "start": image.get("system:time_start"),
        "end": image.get("system:time_end"),
        "fallback": fallback,
    }


# Uppsala and Göteborg in June 2021 (live 2026-10-12): Sentinel-2 images on
# alternating days, with no common day.
_UPPSALA_IMAGES = [
    _simulated_image(f"U {day}", f"{day} 10:30", f"{day} 10:31", points="U")
    for day in ("2021-06-12", "2021-06-15", "2021-06-17", "2021-06-20")
]
_GOTEBORG_IMAGES = [
    _simulated_image(f"G {day}", f"{day} 10:40", f"{day} 10:41", points="G")
    for day in ("2021-06-13", "2021-06-16", "2021-06-18")
]


@pytest.mark.usefixtures("simulated_server")
class TestPerPointFallbackNearest:
    """``nearest``: a point without an own image on the batch-selected day gets its nearest image."""

    def test_points_with_images_on_different_days(self, simulated_server):
        """Göteborg keeps the batch-selected image; Uppsala gets its own nearest image."""
        images = _UPPSALA_IMAGES + _GOTEBORG_IMAGES
        simulated_server(images)

        results = {
            point: _select_point_image(images, point, "2021-06-16", "nearest", batch_points="UG")
            for point in "UG"
        }

        # The batch-selected day is 2021-06-16 (Göteborg). Uppsala's images
        # on 06-15 and 06-17 are one day away: the earlier one wins. Only
        # Uppsala's image comes from the fallback.
        assert {point: _read_server_value(image) for point, (image, _) in results.items()} == {
            "U": "U 2021-06-15",
            "G": "G 2021-06-16",
        }
        assert _read_server_value(results["U"][1]) == _image_dates_of(
            _UPPSALA_IMAGES[1], fallback=True
        )
        assert _read_server_value(results["G"][1]) == _image_dates_of(
            _GOTEBORG_IMAGES[1], fallback=False
        )

    def test_point_alone_gets_the_same_image(self, simulated_server):
        """A point alone selects the same image as in the batch with the other point."""
        simulated_server(_UPPSALA_IMAGES + _GOTEBORG_IMAGES)

        for point, expected_name in (("U", "U 2021-06-15"), ("G", "G 2021-06-16")):
            image, _ = _select_point_image(
                _UPPSALA_IMAGES + _GOTEBORG_IMAGES,
                point,
                "2021-06-16",
                "nearest",
                batch_points=point,
            )
            assert _read_server_value(image) == expected_name

    def test_equal_gaps_select_the_earlier_start_without_a_stable_sort(self, simulated_server):
        """Equal whole-day gaps select the earlier start, whatever the order of the collection."""
        # All three images are one day away. The simulated sort puts equal
        # keys in the reverse order of the collection, so a sort by the day
        # gap alone would select the last image (06-17). Only the start time
        # in the sort key selects the earliest start, 06-15 09:00.
        images = [
            _simulated_image("A 06-15 09:00", "2021-06-15 09:00", points="A"),
            _simulated_image("A 06-15 10:00", "2021-06-15 10:00", points="A"),
            _simulated_image("A 06-17", "2021-06-17 09:00", points="A"),
            _simulated_image("B 06-16", "2021-06-16 10:00", points="B"),
        ]
        simulated_server(images)

        image, image_dates = _select_point_image(
            images, "A", "2021-06-16", "nearest", batch_points="AB"
        )

        assert _read_server_value(image) == "A 06-15 09:00"
        assert _read_server_value(image_dates) == {
            "start": _epoch_ms("2021-06-15 09:00"),
            "end": None,
            "fallback": True,
        }

    def test_gap_counts_whole_days(self, simulated_server):
        """The gap is in whole days between start days, not in hours."""
        # 2021-06-14 23:00 is 25 hours before the centre day (2 days), and
        # 2021-06-17 23:00 is 47 hours after it (1 day).
        images = [
            _simulated_image("A two days before", "2021-06-14 23:00", points="A"),
            _simulated_image("A one day after", "2021-06-17 23:00", points="A"),
            _simulated_image("B on the day", "2021-06-16 10:00", points="B"),
        ]
        simulated_server(images)

        image, _ = _select_point_image(images, "A", "2021-06-16", "nearest", batch_points="AB")

        assert _read_server_value(image) == "A one day after"

    @pytest.mark.parametrize(
        ("own_starts", "expected_name"),
        [
            # Outside the 32-day window: the 366-day window finds the nearest.
            (["2021-04-27", "2021-07-26"], "A 2021-07-26"),
            # The edges of the 366-day window: 366 days before and after.
            (["2020-06-15", "2022-06-17"], "A 2020-06-15"),
            (["2022-06-17"], "A 2022-06-17"),
        ],
    )
    def test_windows(self, simulated_server, own_starts, expected_name):
        """The 366-day window is used when the 32-day window has no own image."""
        images = [_simulated_image(f"A {start}", start, points="A") for start in own_starts]
        images.append(_simulated_image("B 06-16", "2021-06-16 10:00", points="B"))
        simulated_server(images)

        image, _ = _select_point_image(images, "A", "2021-06-16", "nearest", batch_points="AB")

        assert _read_server_value(image) == expected_name

    @pytest.mark.parametrize("own_start", ["2020-06-14", "2022-06-18"])
    def test_no_own_image_in_the_windows_is_a_failed_point(self, simulated_server, own_start):
        """An own image 367 days away is outside both windows: the point's request fails."""
        images = [
            _simulated_image("A far away", own_start, points="A"),
            _simulated_image("B 06-16", "2021-06-16 10:00", points="B"),
        ]
        simulated_server(images)

        image, image_dates = _select_point_image(
            images, "A", "2021-06-16", "nearest", batch_points="AB"
        )

        with pytest.raises(_StubEarthEngineError, match="may not be null"):
            _read_server_value(image)
        with pytest.raises(_StubEarthEngineError, match="may not be null"):
            _read_server_value(image_dates)

    def test_centre_day_is_limited_to_the_batch_index(self, simulated_server):
        """A sample date long after the latest start searches around the latest start."""
        images = [
            _simulated_image("A 06-12", "2021-06-12", points="A"),
            _simulated_image("B 06-16", "2021-06-16 10:00", points="B"),
        ]
        simulated_server(images)

        # The sample date is far after the latest start of the index (B
        # 06-16), so the centre day is 2021-06-16. Windows around the sample
        # date itself would have no image.
        image, _ = _select_point_image(images, "A", "2099-12-31", "nearest", batch_points="AB")

        assert _read_server_value(image) == "A 06-12"

    def test_no_sample_date_searches_around_the_latest_start(self, simulated_server):
        """Without a sample date, a point without an image on the latest start day gets its nearest."""
        images = _UPPSALA_IMAGES + _GOTEBORG_IMAGES
        simulated_server(images)

        image, _ = _select_point_image(images, "G", None, "nearest", batch_points="UG")

        # The latest start of the batch is Uppsala's 06-20; Göteborg's
        # nearest image is 06-18.
        assert _read_server_value(image) == "G 2021-06-18"


@pytest.mark.usefixtures("simulated_server")
class TestPerPointFallbackContains:
    """``contains``: the fast path never selects a later image; the fallback follows the day rule."""

    def test_fast_path_does_not_select_an_image_after_the_centre_day(self, simulated_server):
        """A's images start on the 1st, B's on the 15th: on June 20, A gets its June image."""
        a_images = [
            _simulated_image("A 06-01", "2021-06-01", "2021-07-01", points="A"),
            _simulated_image("A 07-01", "2021-07-01", "2021-08-01", points="A"),
        ]
        b_images = [
            _simulated_image("B 05-15", "2021-05-15", "2021-06-15", points="B"),
            _simulated_image("B 06-15", "2021-06-15", "2021-07-15", points="B"),
            _simulated_image("B 07-15", "2021-07-15", "2021-08-15", points="B"),
        ]
        images = a_images + b_images
        simulated_server(images)

        # The batch-selected image is B 06-15, with the range [06-15, 07-15).
        # A's July image starts in that range, but after the centre day.
        a_image, a_image_dates = _select_point_image(
            images, "A", "2021-06-20", "contains", batch_points="AB"
        )
        b_image, _ = _select_point_image(images, "B", "2021-06-20", "contains", batch_points="AB")

        assert _read_server_value(a_image) == "A 06-01"
        assert _read_server_value(a_image_dates) == _image_dates_of(a_images[0], fallback=True)
        assert _read_server_value(b_image) == "B 06-15"

    @pytest.mark.parametrize(
        ("own_images", "expected_name"),
        [
            # 1. An image that contains the centre day, the latest start.
            (
                [("A long", "2021-05-01", "2021-08-01"), ("A short", "2021-06-10", "2021-06-21")],
                "A short",
            ),
            # An image without an end that starts on the centre day contains it.
            (
                [("A no end", "2021-06-20 10:00", None), ("A before", "2021-06-13", None)],
                "A no end",
            ),
            # 2. No image contains the day: the latest start before it. An
            # image without an end contains only its start day.
            ([("A 06-13", "2021-06-13", None), ("A 06-23", "2021-06-23", None)], "A 06-13"),
            # An image that ends at the start of the day does not contain it;
            # one that ends later on the day does.
            (
                [("A ends at 00:00", "2021-06-10", "2021-06-20"), ("A 06-15", "2021-06-15", None)],
                "A 06-15",
            ),
            (
                [
                    ("A ends at 06:00", "2021-06-10", "2021-06-20 06:00"),
                    ("A 06-15", "2021-06-15", None),
                ],
                "A ends at 06:00",
            ),
            # 3. No image starts before it in 32 days: the earliest start after it.
            ([("A 07-10", "2021-07-10", None), ("A 07-01", "2021-07-01", None)], "A 07-01"),
            # 366-day window: before the day first, also when an image after
            # it is nearer.
            ([("A 03-12", "2021-03-12", None), ("A 08-09", "2021-08-09", None)], "A 03-12"),
            ([("A 2022-06-20", "2022-06-20", None)], "A 2022-06-20"),
        ],
    )
    def test_fallback_rules(self, simulated_server, own_images, expected_name):
        """The fallback order: containing image, latest before, earliest after; 32 then 366 days."""
        images = [_simulated_image(name, start, end, points="A") for name, start, end in own_images]
        simulated_server(images)

        image = _image._build_fallback_image(
            _SimulatedCollection(images), pd.Timestamp("2021-06-20"), "contains"
        )

        assert _read_server_value(image) == expected_name

    def test_no_own_image_in_the_windows_is_null(self, simulated_server):
        """An own image 367 days after the centre day is outside every window."""
        images = [_simulated_image("A far away", "2022-06-22", None, points="A")]
        simulated_server(images)

        image = _image._build_fallback_image(
            _SimulatedCollection(images), pd.Timestamp("2021-06-20"), "contains"
        )

        assert image is None

    def test_no_sample_date_searches_around_the_latest_start(self, simulated_server):
        """Without a sample date, the centre day is the latest start; its containing image wins."""
        # G's long image contains the latest start day of the batch (U
        # 06-20). G 06-19 is nearer to that day, but it ends on 06-19, so
        # it does not contain 06-20: "nearest" would select it.
        g_images = [
            _simulated_image("G long", "2021-06-01", "2021-07-01", points="G"),
            _simulated_image("G 06-19", "2021-06-19 10:40", "2021-06-19 10:41", points="G"),
        ]
        images = _UPPSALA_IMAGES + g_images
        simulated_server(images)

        g_image, g_image_dates = _select_point_image(
            images, "G", None, "contains", batch_points="UG"
        )
        u_image, _ = _select_point_image(images, "U", None, "contains", batch_points="UG")

        assert _read_server_value(g_image) == "G long"
        assert _read_server_value(g_image_dates) == _image_dates_of(g_images[0], fallback=True)
        assert _read_server_value(u_image) == "U 2021-06-20"


@pytest.mark.usefixtures("simulated_server")
class TestPerPointImageDates:
    """The date request names the selected image; the masked image has no dates."""

    def test_point_without_images_gets_the_masked_image_and_no_dates(self, simulated_server):
        """A point where the collection has no image: the masked image, and null dates."""
        images = _UPPSALA_IMAGES + _GOTEBORG_IMAGES
        simulated_server(images)

        image, image_dates = _select_point_image(
            images, "X", "2021-06-16", "nearest", batch_points="UGX"
        )

        assert _read_server_value(image) == "U 2021-06-12 (masked)"
        assert _read_server_value(image_dates) is None

    def test_point_whose_images_fail_the_utm_filter_is_a_failed_point(self, simulated_server):
        """Point images exist but none is in the point's UTM zone: a failed point, not no data."""
        # The point at 0° N, 30° W is in zone 26N; its images are labelled 27N.
        images = [
            _simulated_image("A 06-15", "2021-06-15", points="A", utm_zone="27N"),
            _simulated_image("A 06-16", "2021-06-16", points="A", utm_zone="27N"),
        ]
        simulated_server(images)
        time_starts, time_ends = _batch_index(images)

        image, image_dates = _image._build_image(
            {"collection": "simulated", "bands": ["B4"], "use_utm_zone": True},
            date=pd.Timestamp("2021-06-16"),
            geometry="A",
            collection_time_starts=time_starts,
            collection_time_ends=time_ends,
            lat=0.0,
            lon=-30.0,
        )

        with pytest.raises(_StubEarthEngineError, match="may not be null"):
            _read_server_value(image)
        with pytest.raises(_StubEarthEngineError, match="may not be null"):
            _read_server_value(image_dates)

    def test_circular_aspect_image_uses_the_same_selection(self, simulated_server):
        """The circular aspect image of a fallback point is built from the point's selected image."""
        images = _UPPSALA_IMAGES + _GOTEBORG_IMAGES
        simulated_server(images)

        main_image, main_dates = _select_point_image(
            images, "U", "2021-06-16", "nearest", batch_points="UG"
        )
        circular_image, circular_dates = _select_point_image(
            images, "U", "2021-06-16", "nearest", batch_points="UG", circular_aspect=True
        )

        assert _read_server_value(main_image) == "U 2021-06-15"
        assert circular_image.expression == (
            "(cat(sin(aspect(U 2021-06-15) * 0.017453) as sin, "
            "cos(aspect(U 2021-06-15) * 0.017453) as cos)) masked to slope(U 2021-06-15) > 0"
        )
        assert _read_server_value(circular_dates) == _read_server_value(main_dates)


# ------------------------------------------------------------------
# The dates of the selected image in the statistics and the tiles
# ------------------------------------------------------------------


def _nearest_index_adapter(**attributes) -> GeeRasterAdapter:
    """Return an offline adapter of a ``nearest`` collection whose batch index has 2021-06-16."""
    adapter_attributes = {
        "_dataset_spec": {"collection": "users/test/collection", "bands": ["B4"]},
        "_collection_time_starts": pd.DatetimeIndex(["2021-06-13", "2021-06-16"]),
        "_collection_time_ends": None,
    }
    adapter_attributes.update(attributes)
    return _make_offline_adapter(**adapter_attributes)


class TestResolveDateInfoWithImageDates:
    """``_resolve_date_info`` reports the selected image; the batch index gives the date source."""

    def test_dates_of_the_selected_image(self):
        """The fallback image (06-15) is reported, not the batch-selected image (06-16)."""
        adapter = _nearest_index_adapter()

        date_info = adapter._resolve_date_info(
            "2021-06-16", _image_times("2021-06-15 10:30", "2021-06-15 10:31")
        )

        assert date_info == {
            "date_clamped": False,
            "date_source": "nearest_to_sample",
            "image_time_start": "2021-06-15",
            "image_time_end": "2021-06-15",
        }

    def test_image_without_an_end_has_no_end_field(self):
        """An image without ``system:time_end`` gives no ``image_time_end``."""
        adapter = _nearest_index_adapter()

        assert adapter._resolve_date_info("2021-06-20", _image_times("2021-06-16")) == {
            "date_clamped": True,
            "date_source": "clamped_to_nearest",
            "image_time_start": "2021-06-16",
        }

    def test_point_without_images_has_no_date_fields(self):
        """A null date result (the masked image) gives no date field at all."""
        adapter = _nearest_index_adapter()

        assert adapter._resolve_date_info("2021-06-16", None) == {}
        assert adapter._resolve_date_info(None, None) == {}

    def test_no_batch_index_gives_no_date_fields(self):
        """Without a batch index (image assets, mosaics), there are no date fields."""
        adapter = _nearest_index_adapter(_collection_time_starts=None)

        assert adapter._resolve_date_info("2021-06-16", _image_times("2021-06-16")) == {}


class TestImageDatesInPointStatistics:
    """The statistics request carries the date branch; the meta dict names the selected image."""

    @pytest.mark.parametrize(
        ("image_dates_result", "expected_date_fields"),
        [
            (
                _image_times("2021-06-15 10:30", "2021-06-15 10:31"),
                {
                    "date_clamped": False,
                    "date_source": "nearest_to_sample",
                    "image_time_start": "2021-06-15",
                    "image_time_end": "2021-06-15",
                },
            ),
            # The masked image of a point without images: no date fields.
            (None, {}),
        ],
    )
    def test_date_branch(self, monkeypatch, image_dates_result, expected_date_fields):
        """The date branch is in the point's one request, and its result gives the date fields."""
        adapter = _nearest_index_adapter(crs="EPSG:4326", scale=10)
        recorded_requests = _stub_statistics_request(
            monkeypatch,
            adapter,
            {"window": {"B4_mean": 0.1, "B4_count": 4}, "image_dates": image_dates_result},
        )
        image_requests: list[tuple] = []

        def get_image(date=None, lat=None, lon=None, circular_aspect=False):
            image_requests.append((date, lat, lon, circular_aspect))
            return _StubStatisticsImage("main"), "dates of the selected image"

        adapter._get_image = get_image

        stats, meta = adapter._fetch_stats_single(
            59.85, 17.63, 200, _StubReducer("mean"), ["mean"], ["_mean"], date="2021-06-16"
        )

        assert stats == {"mean": 0.1}
        assert [set(branches) for branches in recorded_requests] == [{"window", "image_dates"}]
        assert recorded_requests[0]["image_dates"] == "dates of the selected image"
        assert {key: meta[key] for key in expected_date_fields} == expected_date_fields
        assert not {"date_clamped", "image_time_start"} & (set(meta) - set(expected_date_fields))
        assert image_requests == [("2021-06-16", 59.85, 17.63, False)]

    def test_circular_aspect_image_has_the_same_arguments(self, monkeypatch):
        """The circular aspect image is requested with the date and point of the main image."""
        adapter = _statistics_adapter({"bands": [], "derived_bands": ["aspect"]})
        _stub_statistics_request(
            monkeypatch,
            adapter,
            {"window": {"aspect_mean": 180.0, "aspect_count": 4}, "aspect_circular": {}},
        )
        image_requests: list[tuple] = []

        def get_image(date=None, lat=None, lon=None, circular_aspect=False):
            image_requests.append((date, lat, lon, circular_aspect))
            return _StubStatisticsImage("circular" if circular_aspect else "main"), None

        adapter._get_image = get_image
        adapter._band_named_keys = True

        adapter._fetch_stats_single(
            59.85, 17.63, 200, _StubReducer("mean"), ["mean"], ["_mean"], date="2021-06-16"
        )

        assert image_requests == [
            ("2021-06-16", 59.85, 17.63, False),
            ("2021-06-16", 59.85, 17.63, True),
        ]


class _StubDateRequest:
    """Stands in for the server-side ``image_dates`` value of one tile. ``getInfo`` counts calls."""

    def __init__(self, result: dict | None = None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.n_requests = 0

    def getInfo(self):
        self.n_requests += 1
        if self._error is not None:
            raise self._error
        return self._result


class TestImageDatesOfTiles:
    """Each tile with per-point date selection sends one date request before its download."""

    def test_tile_gets_the_dates_of_its_image(self, monkeypatch, tmp_path):
        """One date request, then the download; the date fields name the selected image."""
        date_request = _StubDateRequest(_image_times("2021-06-15 10:30", "2021-06-15 10:31"))
        adapter = _nearest_index_adapter(
            _get_image=lambda date=None, lat=None, lon=None: (_StubImage(["B4"]), date_request)
        )
        recorded_downloads = _record_tile_downloads(monkeypatch)
        output_path = tmp_path / "a-test_dataset.tif"

        returned_path, date_info = adapter._export_single(
            59.85, 17.63, 200, output_path, "2021-06-16", resample_m=20
        )

        assert returned_path == output_path
        assert date_request.n_requests == 1
        assert len(recorded_downloads) == 1
        assert date_info == {
            "date_clamped": False,
            "date_source": "nearest_to_sample",
            "image_time_start": "2021-06-15",
            "image_time_end": "2021-06-15",
        }

    def test_failed_date_request_removes_the_old_tile(self, monkeypatch, tmp_path):
        """A failed date request raises before the download and removes a tile of an earlier run."""
        error = _StubEarthEngineError("stub: date request failed")
        adapter = _nearest_index_adapter(
            _get_image=lambda date=None, lat=None, lon=None: (
                _StubImage(["B4"]),
                _StubDateRequest(error=error),
            )
        )
        recorded_downloads = _record_tile_downloads(monkeypatch)
        output_path = tmp_path / "a-test_dataset.tif"
        output_path.write_bytes(b"tile of an earlier run")

        with pytest.raises(_StubEarthEngineError, match="date request failed"):
            adapter._export_single(59.85, 17.63, 200, output_path, "2021-06-16", resample_m=20)

        assert not output_path.exists()
        assert recorded_downloads == []

    def test_failed_date_request_is_a_failed_tile(self, monkeypatch, tmp_path):
        """In a batch, a failed date request makes a failed tile; the other tiles get their dates."""
        date_requests = {
            lat: _StubDateRequest(_image_times(f"2021-06-{15 + index}"))
            for index, lat in enumerate(_THREE_LATS)
        }
        failed_request_error = _StubEarthEngineError("stub: date request failed")
        date_requests[_THREE_LATS[1]] = _StubDateRequest(error=failed_request_error)
        static_image = _StubImage(["B4"])

        def get_image(date=None, lat=None, lon=None):
            if lat is None:
                return static_image, None
            return _StubImage(["B4"]), date_requests[lat]

        adapter = _tile_export_adapter(
            _dataset_spec={"collection": "users/test/collection", "bands": ["B4"]},
            _collection_time_starts=pd.DatetimeIndex(["2021-06-13", "2021-06-16"]),
            _get_image=get_image,
        )
        recorded_downloads = _record_tile_downloads(monkeypatch)
        old_tile = tmp_path / "stub_gee" / "p1-stub_gee.tif"
        old_tile.parent.mkdir(parents=True)
        old_tile.write_bytes(b"tile of an earlier run")

        paths, meta_list = adapter.export_tiles(
            _THREE_LATS,
            _THREE_LONS,
            200,
            tmp_path,
            ids=_THREE_IDS,
            dates=["2021-06-16"] * 3,
            dataset_name="stub_gee",
            disable_progress=True,
        )

        assert [request.n_requests for request in date_requests.values()] == [1, 1, 1]
        assert len(recorded_downloads) == 2
        assert paths == [
            tmp_path / "stub_gee" / "p0-stub_gee.tif",
            None,
            tmp_path / "stub_gee" / "p2-stub_gee.tif",
        ]
        assert meta_list[1] == {
            "error": "_StubEarthEngineError: stub: date request failed",
            "exception": failed_request_error,
        }
        assert not old_tile.exists()
        assert [meta_list[index]["image_time_start"] for index in (0, 2)] == [
            "2021-06-15",
            "2021-06-17",
        ]


# ------------------------------------------------------------------
# Warning for points that used a fallback image far from their sample date
# ------------------------------------------------------------------


class TestCountGapDays:
    """``_count_gap_days`` counts whole UTC days between a sample date and an image (day rule)."""

    @pytest.mark.parametrize(
        ("image_start", "image_end", "expected_gap_days"),
        [
            ("2021-07-26 10:30", None, 40),
            # Whole days: 23:00 on 2021-05-07 is 39 days and 1 hour before
            # the sample day, but 40 days by start day. The end does not count.
            ("2021-05-07 23:00", "2021-06-30", 40),
            ("2021-06-16 23:59", None, 0),
        ],
    )
    def test_nearest_counts_the_days_to_the_start_day(
        self, image_start, image_end, expected_gap_days
    ):
        """``nearest``: the whole days between the sample day and the image's start day."""
        gap_days = _image._count_gap_days(
            "2021-06-16",
            _epoch_ms(image_start),
            _epoch_ms(image_end) if image_end is not None else None,
            "nearest",
        )

        assert gap_days == expected_gap_days

    @pytest.mark.parametrize(
        ("image_start", "image_end", "expected_gap_days"),
        [
            # The image contains the sample day.
            ("2021-06-01", "2021-07-01", 0),
            # An image without an end contains its start day, also when it
            # starts later on that day.
            ("2021-06-20 10:00", None, 0),
            # An image after the sample day: the days to its start day.
            ("2021-07-30 10:00", "2021-08-30", 40),
            # An image before the sample day: the days from its last day. An
            # end at 00:00 does not contain the day that starts then, so the
            # last day is 2021-05-11, not the day of the end (2021-05-12).
            ("2021-05-01", "2021-05-12", 40),
            # An end later on its day contains that day.
            ("2021-05-01", "2021-05-11 06:00", 40),
            # An image without an end, or with an end on its start day,
            # contains only its start day.
            ("2021-05-11 10:00", None, 40),
            ("2021-05-11 10:00", "2021-05-11 10:01", 40),
            # An image that ends at the start of the sample day does not contain it.
            ("2021-06-10", "2021-06-20", 1),
        ],
    )
    def test_contains_counts_the_days_to_the_nearest_day_of_the_image(
        self, image_start, image_end, expected_gap_days
    ):
        """``contains``: 0 when the image contains the sample day, else the days to its nearest day."""
        gap_days = _image._count_gap_days(
            "2021-06-20",
            _epoch_ms(image_start),
            _epoch_ms(image_end) if image_end is not None else None,
            "contains",
        )

        assert gap_days == expected_gap_days


def _fallback_image_times(start_text: str, end_text: str | None = None) -> dict:
    """Return the ``image_dates`` result of a point whose image the per-point fallback selected."""
    return {**_image_times(start_text, end_text), "fallback": True}


class TestFallbackImageGapInDateInfo:
    """``_resolve_date_info`` adds the internal gap only for the fallback image of a dated point."""

    @pytest.mark.parametrize(
        ("sample_date", "image_times", "expected_gap_days"),
        [
            ("2021-06-16", _fallback_image_times("2021-07-26 10:30"), 40),
            ("2021-06-16", _fallback_image_times("2021-06-06 10:30"), 10),
            # The fast path selected the image: no gap.
            ("2021-06-16", {**_image_times("2021-07-26 10:30"), "fallback": False}, None),
            # A clamped date (before the first start of the batch index): no gap.
            ("2020-03-01", _fallback_image_times("2021-06-13"), None),
            # No sample date: no gap.
            (None, _fallback_image_times("2021-03-01"), None),
        ],
    )
    def test_nearest(self, sample_date, image_times, expected_gap_days):
        """The gap of a ``nearest`` collection counts the days to the image's start day."""
        adapter = _nearest_index_adapter()

        date_info = adapter._resolve_date_info(sample_date, image_times)

        assert date_info.get("fallback_image_gap_days") == expected_gap_days
        assert ("fallback_image_gap_days" in date_info) == (expected_gap_days is not None)
        assert date_info["image_time_start"] == pd.to_datetime(
            image_times["start"], unit="ms"
        ).strftime("%Y-%m-%d")

    def test_contains_uses_the_last_day_of_an_earlier_image(self):
        """The gap of a ``contains`` collection counts from the last day of an image before the date."""
        adapter = _contains_adapter(_EIGHT_DAY_STARTS, _EIGHT_DAY_ENDS)

        # 2021-01-20 is in the last interval of the index, so it is not
        # clamped. The image's last day is 2020-12-09: 42 days before the
        # sample day (its start day is 50 days before it).
        date_info = adapter._resolve_date_info(
            "2021-01-20", _fallback_image_times("2020-12-01", "2020-12-10")
        )

        assert date_info["date_clamped"] is False
        assert date_info["fallback_image_gap_days"] == 42


# Four points of one batch. Each point's request returns the image times in
# _FAR_IMAGE_TIMES_BY_LATITUDE, and the adapter turns them into the point's
# date fields. Only p_far used a fallback image more than 32 days from its
# sample date and is not clamped.
_FAR_IMAGE_POINTS = pd.DataFrame(
    {
        "occurrenceID": ["p_far", "p_near", "p_clamped", "p_fast"],
        "decimalLatitude": [59.85, 59.86, 59.87, 59.88],
        "decimalLongitude": [17.63, 17.64, 17.65, 17.66],
        "eventDate": ["2021-06-16", "2021-06-16", "2020-03-01", "2021-06-16"],
    }
)
_FAR_IMAGE_TIMES_BY_LATITUDE = {
    # A fallback image 40 days after the sample date.
    59.85: _fallback_image_times("2021-07-26 10:30"),
    # A fallback image 10 days before the sample date.
    59.86: _fallback_image_times("2021-06-06 10:30"),
    # A clamped date (before the first image of the batch), and a fallback
    # image 306 days later.
    59.87: _fallback_image_times("2021-01-01 10:30"),
    # The fast path: the image of the batch-selected range, 40 days away.
    59.88: {**_image_times("2021-07-26 10:30"), "fallback": False},
}


def _stub_far_image_requests(adapter: GeeRasterAdapter) -> None:
    """Replace the per-point statistics request and tile export of ``adapter`` with stubs.

    The batch index starts on 2021-01-01 and ends on 2021-12-01. Each point
    gets the image times of ``_FAR_IMAGE_TIMES_BY_LATITUDE``, and
    ``_resolve_date_info`` turns them into its date fields, as for a real
    request.
    """
    adapter._collection_time_starts = pd.DatetimeIndex(["2021-01-01 10:30", "2021-12-01 10:30"])

    def request(lat, lon, window_m, *args, date=None, **kwargs):
        meta = {
            "in_extent": True,
            "n_pixels": 400,
            "had_nodata": False,
            "coverage_pct": 100.0,
            "window_m": window_m,
            **adapter._resolve_date_info(date, _FAR_IMAGE_TIMES_BY_LATITUDE[lat]),
        }
        return {"B1_mean": 1.0, "B2_mean": 2.0}, meta

    def export(lat, lon, window_m, output_path, date=None, resample_m=None):
        return output_path, adapter._resolve_date_info(date, _FAR_IMAGE_TIMES_BY_LATITUDE[lat])

    adapter._fetch_stats_single = request
    adapter._export_single = export


def _far_image_warnings(recorded_warnings) -> list:
    """Return the recorded warnings about images far from the sample date."""
    return [
        recorded
        for recorded in recorded_warnings
        if "more than 32 days from their sample date" in str(recorded.message)
    ]


class TestFarImageWarningThroughExtract:
    """``extract()`` warns once per dataset and window about fallback images far from the date."""

    def test_tabular_run_warns_only_for_the_far_fallback_image(self, stub_gee_dataset, tmp_path):
        """Only p_far gives a warning; the sidecar records it and has no internal key."""
        _stub_far_image_requests(stub_gee_dataset)

        with pytest.warns(UserWarning, match="more than 32 days") as recorded_warnings:
            extract(_FAR_IMAGE_POINTS, _stub_gee_config("gee_far", "tabular"), output_dir=tmp_path)

        [far_image_warning] = _far_image_warnings(recorded_warnings)
        # The warning points at the caller of extract().
        assert far_image_warning.filename == __file__
        message = str(far_image_warning.message)
        assert message.startswith(
            "Dataset 'stub_gee', window 200 m: 1 of 4 points used an image more than 32 days "
            "from their sample date (largest gap: 40 days; IDs: 'p_far')."
        )
        assert message.endswith(
            "The QC columns 'stub_gee_image_time_start_200m' and "
            "'stub_gee_image_time_end_200m' show the dates of each point's image."
        )
        sidecar_text = (tmp_path / "gee_far_metadata.json").read_text()
        sidecar = json.loads(sidecar_text)
        assert {
            key: value for key, value in sidecar["warnings"].items() if key.startswith("far_image")
        } == {"far_image_stub_gee_200m": message}
        assert "fallback_image_gap_days" not in sidecar_text
        qc_table = pd.read_csv(tmp_path / "gee_far_qc.csv")
        assert not [column for column in qc_table.columns if "gap" in column]
        assert qc_table["stub_gee_image_time_start_200m"].tolist() == [
            "2021-07-26",
            "2021-06-06",
            "2021-01-01",
            "2021-07-26",
        ]
        assert qc_table["stub_gee_date_clamped_200m"].tolist() == [False, False, True, False]

    def test_tabular_run_without_metadata_says_that_the_qc_table_is_not_written(
        self, stub_gee_dataset, tmp_path
    ):
        """With ``write_metadata=False``, the warning says that the QC table is not written."""
        _stub_far_image_requests(stub_gee_dataset)

        with pytest.warns(UserWarning, match="more than 32 days") as recorded_warnings:
            extract(
                _FAR_IMAGE_POINTS,
                _stub_gee_config("gee_far", "tabular"),
                output_dir=tmp_path,
                write_metadata=False,
            )

        [far_image_warning] = _far_image_warnings(recorded_warnings)
        assert str(far_image_warning.message).endswith(
            "is not written because write_metadata is False."
        )

    def test_raster_run_warns_only_for_the_far_fallback_image(self, stub_gee_dataset, tmp_path):
        """A raster run gives the same warning; the sidecar and the manifest have no internal key."""
        _stub_far_image_requests(stub_gee_dataset)

        with pytest.warns(UserWarning, match="more than 32 days") as recorded_warnings:
            extract(
                _FAR_IMAGE_POINTS, _stub_gee_config("gee_far_tiles", "raster"), output_dir=tmp_path
            )

        [far_image_warning] = _far_image_warnings(recorded_warnings)
        assert far_image_warning.filename == __file__
        message = str(far_image_warning.message)
        assert message.startswith(
            "Dataset 'stub_gee', window 200 m: 1 of 4 points used an image more than 32 days "
            "from their sample date (largest gap: 40 days; IDs: 'p_far')."
        )
        assert "The sidecar 'date_info' lists the dates of the images" in message
        batch_dir = tmp_path / "gee_far_tiles"
        sidecar_text = (batch_dir / "gee_far_tiles_metadata.json").read_text()
        sidecar = json.loads(sidecar_text)
        assert {
            key: value for key, value in sidecar["warnings"].items() if key.startswith("far_image")
        } == {"far_image_stub_gee_200m": message}
        assert "fallback_image_gap_days" not in sidecar_text
        manifest = pd.read_csv(batch_dir / "stub_gee" / "tiles_manifest.csv")
        assert list(manifest.columns) == ["occurrenceID", "tile_filename", "exported"]

    def test_run_without_dates_gives_no_warning(self, stub_gee_dataset, tmp_path):
        """Without a sample date, a fallback image far from the latest start gives no warning."""
        _stub_far_image_requests(stub_gee_dataset)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            extract(
                _FAR_IMAGE_POINTS.drop(columns="eventDate"),
                _stub_gee_config("gee_no_dates", "tabular"),
                output_dir=tmp_path,
            )

        assert _far_image_warnings(recorded_warnings) == []
        sidecar = json.loads((tmp_path / "gee_no_dates_metadata.json").read_text())
        assert not [key for key in sidecar.get("warnings", {}) if key.startswith("far_image")]
        assert sidecar["datasets"]["stub_gee"]["date_info"]["n_most_recent_no_date"] == 4


class TestFarImageWarningText:
    """The warning counts the points with a gap above 32 days and lists up to 10 IDs."""

    def test_threshold_is_the_first_fallback_window(self):
        """The warning threshold equals the first search window of the fallback (32 days)."""
        assert _extract_module._FAR_IMAGE_GAP_DAYS == _image._FALLBACK_WINDOWS_DAYS[0] == 32

    @pytest.mark.parametrize(("gap_days", "expected_warning"), [(32, False), (33, True)])
    def test_warning_starts_above_32_days(self, gap_days, expected_warning):
        """A gap of 32 days gives no warning; a gap of 33 days gives one."""
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            message = _extract_module._warn_about_far_images(
                [{"fallback_image_gap_days": gap_days}, {}],
                ["a", "b"],
                dataset="d",
                batch_label="window 200 m",
                date_location="Dates are in the QC table.",
            )

        assert (message is not None) == expected_warning
        assert len(_far_image_warnings(recorded_warnings)) == int(expected_warning)

    def test_message_lists_ten_ids_and_the_number_of_points(self):
        """Twelve far points: the count, the largest gap, ten IDs, and the number not listed."""
        meta_list = [{"fallback_image_gap_days": 33 + index} for index in range(12)]
        meta_list.append({"fallback_image_gap_days": 5})

        with pytest.warns(UserWarning, match="more than 32 days"):
            message = _extract_module._warn_about_far_images(
                meta_list,
                [f"p{index}" for index in range(13)],
                dataset="d",
                batch_label="point values",
                date_location="Dates are in the QC table.",
            )

        ids_text = ", ".join(f"'p{index}'" for index in range(10))
        assert message.startswith(
            f"Dataset 'd', point values: 12 of 13 points used an image more than 32 days from "
            f"their sample date (largest gap: 44 days; IDs: {ids_text}, ... and 2 more)."
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
