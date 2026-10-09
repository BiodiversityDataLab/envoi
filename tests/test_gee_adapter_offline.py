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

from types import SimpleNamespace

import ee
import pandas as pd
import pytest

from envoi.adapters.earth_engine import _image
from envoi.adapters.earth_engine._image import (
    _get_collection_time_bounds,
    _resolve_date_filter_range,
)
from envoi.adapters.earth_engine.adapter import GeeRasterAdapter
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
