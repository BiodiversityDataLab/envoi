"""Tests of the provider contract with a test-only fake provider.

``FakeStatsAdapter`` is a third adapter that only this module registers. It
runs through ``envoi.extract()`` with no change to a generic module, so a new
data provider needs only its adapter module. The module also tests the checks
of ``envoi.adapters.register()``. No test uses the network or a raster file.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest

from envoi import adapters, update_catalog
from envoi.adapters import register
from envoi.adapters.base import BaseAdapter
from envoi.adapters.local_adapter import LocalRasterAdapter
from envoi.extract import extract
from envoi.progress import ProgressStepCallback

# The bands of the fake provider, in the provider's own order, and the base
# value of the statistics of each band.
_FAKE_BAND_BASE_VALUES = {"temp": 0.0, "precip": 100.0}


class FakeStatsAdapter(BaseAdapter):
    """A tabular-only provider with two bands and fixed statistics.

    For a point at ``(lat, lon)``, the ``mean`` of a band is the band's base
    value plus ``lat``, and the ``max`` is the ``mean`` plus ``lon``. The
    statistic keys are ``{band}_{reducer}``. Every point is in the extent and
    has full coverage. The class records the merged spec of each instance in
    ``built_specs``.
    """

    output_types = frozenset({"tabular"})
    supported_reducers = frozenset({"mean", "max"})

    # The merged spec of each instance, in build order. The fixture clears it.
    built_specs: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, spec: dict[str, Any]) -> None:
        # A per-call band list sets `bands`. Without one, the provider uses all its bands.
        self.bands = list(spec.get("bands") or _FAKE_BAND_BASE_VALUES)
        FakeStatsAdapter.built_specs.append(dict(spec))

    def fetch_stats_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        reducer_names: Sequence[str],
        *,
        dates: Sequence | None = None,
        progress_desc: str | None = None,
        disable_progress: bool = False,
        progress_callback: ProgressStepCallback | None = None,
    ) -> list[tuple[dict, dict]]:
        """Return one ``(stats_dict, meta_dict)`` pair for each point, in input order."""
        results = []
        for lat, lon in zip(lats, lons):
            stats = {}
            for band in self.bands:
                band_mean = _FAKE_BAND_BASE_VALUES[band] + lat
                band_values = {"mean": band_mean, "max": band_mean + lon}
                for reducer_name in reducer_names:
                    stats[f"{band}_{reducer_name}"] = band_values[reducer_name]
            meta = {"in_extent": True, "n_pixels": 4, "had_nodata": False, "coverage_pct": 100.0}
            results.append((stats, meta))
        return results

    def build_dataset_meta(
        self,
        spec: dict,
        meta_list: list | None = None,
        exported_paths: list | None = None,
        quality: dict | None = None,
        lats: Sequence[float] | None = None,
        lons: Sequence[float] | None = None,
    ) -> dict:
        """Return the data source, the path, the bands, and the quality summary."""
        dataset_meta: dict[str, Any] = {
            "data_source": spec["data_source"],
            "path": spec["path"],
            "bands": list(self.bands),
        }
        if quality:
            dataset_meta["quality"] = dict(quality)
        return dataset_meta


def _fake_adapter_with(**class_attributes: Any) -> type[BaseAdapter]:
    """Return a subclass of ``FakeStatsAdapter`` with other class attributes."""
    return type("ChangedFakeAdapter", (FakeStatsAdapter,), class_attributes)


@pytest.fixture
def isolated_registry(monkeypatch) -> dict[str, type[BaseAdapter]]:
    """Give ``envoi.adapters`` a copy of the registry for one test.

    ``register()``, ``get_adapter()``, and the catalog checks read the module
    attribute ``_REG``, so they all use the copy. monkeypatch restores the
    original registry after the test.
    """
    registry_copy = dict(adapters._REG)
    monkeypatch.setattr(adapters, "_REG", registry_copy)
    return registry_copy


@pytest.fixture
def fake_provider(isolated_registry) -> type[FakeStatsAdapter]:
    """Register ``FakeStatsAdapter`` as the data source ``fake`` through ``register()``."""
    FakeStatsAdapter.built_specs.clear()
    register("fake", FakeStatsAdapter)
    return FakeStatsAdapter


@pytest.fixture
def three_points() -> pd.DataFrame:
    """Three dated points with simple coordinates, so the fake statistics are exact numbers."""
    return pd.DataFrame(
        {
            "occurrenceID": ["A", "B", "C"],
            "decimalLatitude": [10.0, 20.0, 30.0],
            "decimalLongitude": [1.0, 2.0, 3.0],
            "eventDate": ["2020-01-15", "2020-06-15", "2021-03-31"],
        }
    )


def _read_sidecar(path: Path) -> dict:
    """Read a metadata sidecar with ``json``. This fails if the file is not valid JSON."""
    return json.loads(path.read_text())


class TestFakeProviderThroughExtract:
    """A provider that is only an adapter class runs through ``envoi.extract()``."""

    def test_fake_adapter_keeps_base_adapter_signatures(self):
        """The fake adapter calls have the signatures of ``BaseAdapter``, annotations included."""
        for method_name in ("fetch_stats_batch", "build_dataset_meta"):
            fake_signature = inspect.signature(getattr(FakeStatsAdapter, method_name))
            base_signature = inspect.signature(getattr(BaseAdapter, method_name))
            assert fake_signature == base_signature, method_name

    def test_tabular_run_with_per_call_band_list(self, fake_provider, three_points, tmp_path):
        """A per-call list of two bands gives the exact statistic, QC, and sidecar outputs."""
        update_catalog(
            {
                "datasets": {
                    "fake_climate": {
                        "data_source": "fake",
                        "path": "fake://climate",
                        "data_type": "continuous",
                    }
                }
            }
        )
        # The per-call list reverses the provider's band order, so the column
        # order shows that the adapter used the per-call list.
        run_config = {
            "batch_id": "fake_run",
            "datasets": [{"fake_climate": ["precip", "temp"]}],
            "settings": {
                "output_type": "tabular",
                "statistics": ["mean", "max"],
                "window_size_m": 100,
            },
        }

        outputs = extract(three_points, run_config, output_dir=tmp_path, write_metadata=True)

        # The adapter receives the catalog entry merged with the per-call band list.
        assert fake_provider.built_specs == [
            {
                "data_source": "fake",
                "path": "fake://climate",
                "data_type": "continuous",
                "bands": ["precip", "temp"],
                "band_naming": "per_band",
            }
        ]

        # Statistic keys "{band}_{reducer}" become "<dataset>_{band}_{reducer}_<N>m" columns.
        stats_table = pd.read_csv(outputs["fake_run"])
        assert stats_table.columns.tolist() == [
            "occurrenceID",
            "decimalLatitude",
            "decimalLongitude",
            "eventDate",
            "fake_climate_precip_mean_100m",
            "fake_climate_precip_max_100m",
            "fake_climate_temp_mean_100m",
            "fake_climate_temp_max_100m",
        ]
        assert stats_table["occurrenceID"].tolist() == ["A", "B", "C"]
        # mean = band base value + lat; max = mean + lon.
        assert stats_table["fake_climate_precip_mean_100m"].tolist() == [110.0, 120.0, 130.0]
        assert stats_table["fake_climate_precip_max_100m"].tolist() == [111.0, 122.0, 133.0]
        assert stats_table["fake_climate_temp_mean_100m"].tolist() == [10.0, 20.0, 30.0]
        assert stats_table["fake_climate_temp_max_100m"].tolist() == [11.0, 22.0, 33.0]

        # The four required meta keys become the QC columns of the dataset and window.
        qc_table = pd.read_csv(tmp_path / "fake_run_qc.csv")
        assert qc_table.columns.tolist() == [
            "occurrenceID",
            "decimalLatitude",
            "decimalLongitude",
            "eventDate",
            "fake_climate_in_extent_100m",
            "fake_climate_n_pixels_100m",
            "fake_climate_had_nodata_100m",
            "fake_climate_coverage_pct_100m",
        ]
        assert qc_table["fake_climate_in_extent_100m"].tolist() == [True, True, True]
        assert qc_table["fake_climate_n_pixels_100m"].tolist() == [4, 4, 4]
        assert qc_table["fake_climate_had_nodata_100m"].tolist() == [False, False, False]
        assert qc_table["fake_climate_coverage_pct_100m"].tolist() == [100.0, 100.0, 100.0]

        # The sidecar has the adapter's dataset entry with the quality summary
        # that extract() computed, and the bands of the per-call list.
        sidecar = _read_sidecar(tmp_path / "fake_run_metadata.json")
        assert sidecar["datasets"] == {
            "fake_climate": {
                "data_source": "fake",
                "path": "fake://climate",
                "bands": ["precip", "temp"],
                "quality": {
                    "100m": {
                        "n_zero": 0,
                        "n_partial": 0,
                        "n_full": 3,
                        "total": 3,
                        "ids_no_data": [],
                        "n_errors": 0,
                        "ids_errors": [],
                    }
                },
            }
        }
        assert sidecar["config"]["datasets"] == [
            {"name": "fake_climate", "bands": ["precip", "temp"], "derived_bands": None}
        ]
        assert "warnings" not in sidecar


class TestRegister:
    """``register()`` accepts a valid adapter and rejects each invalid registration."""

    @pytest.mark.parametrize("name", ["Fake", "1fake", "fake-source", "", "fake\n", 3])
    def test_name_that_does_not_match_the_pattern_is_rejected(self, isolated_registry, name):
        """A name that does not match ``^[a-z][a-z0-9_]*$`` raises ``ValueError``."""
        with pytest.raises(ValueError, match=r"the name must match \^\[a-z\]\[a-z0-9_\]\*\$"):
            register(name, FakeStatsAdapter)
        assert name not in isolated_registry

    @pytest.mark.parametrize("adapter_class", [object, "FakeStatsAdapter"])
    def test_class_that_is_not_a_base_adapter_is_rejected(self, isolated_registry, adapter_class):
        """A value that is not a subclass of ``BaseAdapter`` raises ``ValueError``."""
        with pytest.raises(
            ValueError, match="Cannot register adapter 'fake': .* is not a subclass of BaseAdapter"
        ):
            register("fake", adapter_class)
        assert "fake" not in isolated_registry

    def test_different_class_under_a_registered_name_is_rejected(self, isolated_registry):
        """A second class under a registered name raises ``ValueError``; the first one stays."""
        with pytest.raises(
            ValueError,
            match=(
                "Cannot register adapter 'local' as FakeStatsAdapter: the class "
                "envoi.adapters.local_adapter.LocalRasterAdapter is already registered"
            ),
        ):
            register("local", FakeStatsAdapter)
        assert isolated_registry["local"] is LocalRasterAdapter

    @pytest.mark.parametrize(
        ("name", "adapter_class"),
        [("fake", FakeStatsAdapter), ("local", LocalRasterAdapter)],
    )
    def test_same_class_again_is_accepted(self, isolated_registry, name, adapter_class):
        """Registering the registered class again under its name changes nothing."""
        register(name, adapter_class)
        register(name, adapter_class)
        assert isolated_registry[name] is adapter_class

    def test_class_of_a_reloaded_module_replaces_the_registered_class(self, isolated_registry):
        """A class with the same module and qualified name (a reloaded module) replaces the old one."""
        register("fake", FakeStatsAdapter)
        reloaded_class = _fake_adapter_with(
            __module__=FakeStatsAdapter.__module__, __qualname__=FakeStatsAdapter.__qualname__
        )
        register("fake", reloaded_class)
        assert isolated_registry["fake"] is reloaded_class

    @pytest.mark.parametrize(
        ("class_attributes", "message"),
        [
            ({"output_types": frozenset()}, "output_types must be a non-empty subset"),
            (
                {"output_types": frozenset({"tabular", "vector"})},
                "output_types must be a non-empty subset",
            ),
            ({"output_types": {"tabular"}}, "output_types must be a frozenset of str"),
            (
                {"supported_reducers": frozenset({"mean", "p99"})},
                r"supported_reducers has unknown reducer\(s\) \['p99'\]",
            ),
            (
                {"supported_reducers": frozenset()},
                "supported_reducers must not be empty, because output_types has 'tabular'",
            ),
            (
                {"supported_reducers": ("mean", "max")},
                "supported_reducers must be a frozenset of str",
            ),
            (
                {"computable_derived_bands": frozenset({1})},
                "computable_derived_bands must be a frozenset of str",
            ),
            ({"catalog_keys": ["units"]}, "catalog_keys must be a frozenset of str"),
        ],
    )
    def test_invalid_capability_declaration_is_rejected(
        self, isolated_registry, class_attributes, message
    ):
        """An invalid capability declaration raises ``ValueError`` that names the adapter."""
        with pytest.raises(ValueError, match=f"Cannot register adapter 'fake_changed': {message}"):
            register("fake_changed", _fake_adapter_with(**class_attributes))
        assert "fake_changed" not in isolated_registry

    def test_raster_only_adapter_without_reducers_is_accepted(self, isolated_registry):
        """An adapter without ``"tabular"`` can declare no reducers."""
        raster_only_class = _fake_adapter_with(
            output_types=frozenset({"raster"}), supported_reducers=frozenset()
        )
        register("fake_raster", raster_only_class)
        assert isolated_registry["fake_raster"] is raster_only_class
