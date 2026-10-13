"""Tests of the provider contract with a test-only fake provider.

``FakeStatsAdapter`` is a third adapter that only this module registers. It
runs through ``envoi.extract()`` with no change to a generic module, so a new
data provider needs only its adapter module. The module also tests the checks
of ``envoi.adapters.register()``, the pre-flight checks that ``extract()``
runs with the adapter declarations and ``check_dataset`` before it builds any
adapter, the catalog checks with the adapter's ``catalog_keys`` and
catalog hooks, the checks of the adapter results with variants of the
fake adapter that break the result contract, and the lifetime of the adapter:
one instance for each run config and dataset, used for every window size. No
test uses the network or a raster file.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest
import yaml

from envoi import adapters, update_catalog
from envoi._filenames import build_tile_filenames
from envoi.adapters import register
from envoi.adapters.base import BaseAdapter
from envoi.adapters.local_adapter import LocalRasterAdapter
from envoi.catalog import CatalogError, load_catalog, load_catalogs
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
    ``built_specs``. Its pre-flight check accepts only a ``path`` that starts
    with ``fake://``, and records its arguments in ``checked_specs``. Its
    catalog entries can have the key ``fake_product``, which must be text.
    """

    output_types = frozenset({"tabular"})
    supported_reducers = frozenset({"mean", "max"})
    catalog_keys = frozenset({"fake_product"})

    # The merged spec of each instance, in build order. The fixture clears it.
    built_specs: ClassVar[list[dict[str, Any]]] = []
    # The (merged spec, context) of each check_dataset call, in call order.
    # The fixture clears it.
    checked_specs: ClassVar[list[tuple[dict[str, Any], str]]] = []

    @classmethod
    def check_dataset(cls, spec: dict, *, context: str) -> None:
        """Raise ValueError when the ``path`` of the dataset does not start with ``fake://``."""
        FakeStatsAdapter.checked_specs.append((dict(spec), context))
        if not str(spec["path"]).startswith("fake://"):
            raise ValueError(
                f"{context}: the path {spec['path']!r} is not an address of the fake "
                f"provider. Set 'path' in the catalog entry to an address that starts "
                f"with 'fake://'."
            )

    @classmethod
    def validate_catalog_entry(cls, name: str, entry: dict) -> None:
        """Raise ValueError when the entry sets ``fake_product`` to a value that is not text."""
        fake_product = entry.get("fake_product")
        if fake_product is not None and not isinstance(fake_product, str):
            raise ValueError(
                f"fake_product must be the name of a product as text, got {fake_product!r}."
            )

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


class FakeRasterAdapter(FakeStatsAdapter):
    """The fake provider, with the ``"raster"`` output type as well.

    ``export_tiles`` writes no file. It returns the tile path of each point
    with a latitude below 25, and ``None`` (no tile, as for a point outside
    the extent) for the other points. Each meta dict is empty.
    """

    output_types = frozenset({"tabular", "raster"})

    def export_tiles(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        output_dir: str | Path,
        *,
        ids: Sequence[str] | None = None,
        dates: Sequence | None = None,
        dataset_name: str = "dataset",
        resample_m: float | None = None,
        filename_suffix: str | None = None,
        progress_desc: str | None = None,
        disable_progress: bool = False,
        progress_callback: ProgressStepCallback | None = None,
    ) -> tuple[list[Path | None], list[dict]]:
        """Return ``(paths, meta_list)``, in input order, without writing any file."""
        tiles_dir = Path(output_dir) / dataset_name
        tile_filenames = build_tile_filenames(ids, dataset_name, filename_suffix)
        paths = [
            tiles_dir / tile_filename if lat < 25 else None
            for lat, tile_filename in zip(lats, tile_filenames)
        ]
        return paths, [{} for _ in paths]


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
    FakeStatsAdapter.checked_specs.clear()
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


def _register_fake_datasets() -> None:
    """Register the datasets ``fake_climate`` and ``fake_remote`` of the data source ``fake``.

    The path of ``fake_remote`` does not start with ``fake://``, so it fails
    the pre-flight check of the fake provider.
    """
    update_catalog(
        {
            "datasets": {
                "fake_climate": {
                    "data_source": "fake",
                    "path": "fake://climate",
                    "data_type": "continuous",
                },
                "fake_remote": {
                    "data_source": "fake",
                    "path": "https://example.org/climate.tif",
                    "data_type": "continuous",
                },
            }
        }
    )


def _fake_tabular_config(
    batch_id: str,
    datasets: list,
    statistics: list | dict | None = None,
    window_size_m: int | list[int] = 100,
) -> dict[str, Any]:
    """Return a tabular run config, with ``["mean"]`` without ``statistics``."""
    return {
        "batch_id": batch_id,
        "datasets": datasets,
        "settings": {
            "output_type": "tabular",
            "statistics": statistics or ["mean"],
            "window_size_m": window_size_m,
        },
    }


def _fake_raster_config(batch_id: str) -> dict[str, Any]:
    """Return a raster run config of ``fake_climate`` with a 100 m window."""
    return {
        "batch_id": batch_id,
        "datasets": ["fake_climate"],
        "settings": {"output_type": "raster", "window_size_m": 100},
    }


def _register_fake_variant(adapter_class: type[BaseAdapter]) -> None:
    """Register ``adapter_class`` as the data source ``fake_variant``.

    Also register the dataset ``fake_climate`` of that data source. Use it
    with the ``isolated_registry`` fixture.
    """
    register("fake_variant", adapter_class)
    update_catalog({"datasets": {"fake_climate": _fake_climate_entry(data_source="fake_variant")}})


class TestFakeProviderThroughExtract:
    """A provider that is only an adapter class runs through ``envoi.extract()``."""

    def test_fake_adapter_keeps_base_adapter_signatures(self):
        """The fake adapter calls have the signatures of ``BaseAdapter``, annotations included."""
        for method_name in (
            "fetch_stats_batch",
            "build_dataset_meta",
            "check_dataset",
            "validate_catalog_entry",
        ):
            fake_signature = inspect.signature(getattr(FakeStatsAdapter, method_name))
            base_signature = inspect.signature(getattr(BaseAdapter, method_name))
            assert fake_signature == base_signature, method_name
        assert inspect.signature(FakeRasterAdapter.export_tiles) == inspect.signature(
            BaseAdapter.export_tiles
        )

    def test_tabular_run_with_per_call_band_list(self, fake_provider, three_points, tmp_path):
        """A per-call list of two bands gives the exact statistic, QC, and sidecar outputs.

        The catalog entry has the fake provider's own catalog key, which
        reaches the pre-flight check and the adapter.
        """
        update_catalog(
            {
                "datasets": {
                    "fake_climate": {
                        "data_source": "fake",
                        "path": "fake://climate",
                        "data_type": "continuous",
                        "fake_product": "reanalysis",
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

        # The pre-flight check and the adapter receive the catalog entry merged
        # with the per-call band list.
        merged_spec = {
            "data_source": "fake",
            "path": "fake://climate",
            "data_type": "continuous",
            "fake_product": "reanalysis",
            "bands": ["precip", "temp"],
            "band_naming": "per_band",
        }
        assert fake_provider.checked_specs == [
            (merged_spec, "Output 'fake_run': dataset 'fake_climate'")
        ]
        assert fake_provider.built_specs == [merged_spec]

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

    def test_raster_run_gets_the_tile_summary_from_extract(
        self, isolated_registry, three_points, tmp_path
    ):
        """``extract()`` adds the tile summary as the last key of the adapter's dataset entry.

        The fake adapter computes no tile summary. Point C gets no tile.
        """
        _register_fake_variant(FakeRasterAdapter)

        extract(three_points, _fake_raster_config("fake_tiles"), output_dir=tmp_path)

        sidecar = _read_sidecar(tmp_path / "fake_tiles" / "fake_tiles_metadata.json")
        dataset_entry = sidecar["datasets"]["fake_climate"]
        assert dataset_entry == {
            "data_source": "fake_variant",
            "path": "fake://climate",
            "bands": ["temp", "precip"],
            "quality": {
                "tiles": {
                    "n_exported": 2,
                    "n_failed": 1,
                    "total": 3,
                    "n_errors": 0,
                    "ids_errors": [],
                }
            },
        }
        assert list(dataset_entry)[-1] == "quality"
        manifest = pd.read_csv(tmp_path / "fake_tiles" / "fake_climate" / "tiles_manifest.csv")
        assert manifest["tile_filename"].tolist()[:2] == [
            "A-fake_climate.tif",
            "B-fake_climate.tif",
        ]
        assert manifest["exported"].tolist() == [True, True, False]


class TestPreFlightChecks:
    """A run config that the fake provider does not support raises before any adapter is built.

    The capability declarations and ``check_dataset`` of the fake adapter
    decide. Each error is a ``ValueError`` that names the output and the
    dataset, and the call writes no output.
    """

    def test_raster_output_type_is_rejected(self, fake_provider, three_points, tmp_path):
        """A raster run config raises, because the fake adapter declares only ``"tabular"``."""
        _register_fake_datasets()

        with pytest.raises(ValueError) as error_info:
            extract(three_points, _fake_raster_config("fake_tiles"), output_dir=tmp_path)

        message = str(error_info.value)
        assert message.startswith(
            "Output 'fake_tiles': dataset 'fake_climate' has data_source 'fake', whose adapter "
            "does not support output_type 'raster'. Supported output types: ['tabular']."
        )
        assert fake_provider.built_specs == []
        assert not any(tmp_path.iterdir())

    def test_unsupported_reducer_is_rejected(self, fake_provider, three_points, tmp_path):
        """A reducer outside the fake adapter's ``supported_reducers`` raises and lists them."""
        _register_fake_datasets()
        run_config = _fake_tabular_config("fake_run", ["fake_climate"], ["mean", "median"])

        with pytest.raises(ValueError) as error_info:
            extract(three_points, run_config, output_dir=tmp_path)

        message = str(error_info.value)
        assert message.startswith(
            "Output 'fake_run': dataset 'fake_climate' has data_source 'fake', whose adapter "
            "does not support the reducer(s) ['median']. Supported reducers: ['max', 'mean']."
        )
        assert "Remove ['median'] from 'statistics'" in message
        assert fake_provider.built_specs == []
        assert not any(tmp_path.iterdir())

    def test_only_the_reducers_of_the_dataset_data_type_are_checked(
        self, fake_provider, three_points, tmp_path
    ):
        """An unsupported reducer of the other data_type does not raise or reach the adapter."""
        _register_fake_datasets()
        statistics = {"continuous": ["max"], "categorical": ["mode"]}
        run_config = _fake_tabular_config("fake_run", ["fake_climate"], statistics)

        outputs = extract(three_points, run_config, output_dir=tmp_path)

        # The continuous dataset gets the continuous list only.
        stats_table = pd.read_csv(outputs["fake_run"])
        assert [column for column in stats_table.columns if column.startswith("fake_")] == [
            "fake_climate_temp_max_100m",
            "fake_climate_precip_max_100m",
        ]
        assert stats_table["fake_climate_temp_max_100m"].tolist() == [11.0, 22.0, 33.0]
        assert len(fake_provider.built_specs) == 1

    def test_derived_band_is_rejected(self, fake_provider, three_points, tmp_path):
        """A derived band in a per-call band list raises, because the fake adapter computes none."""
        _register_fake_datasets()
        run_config = _fake_tabular_config("fake_run", [{"fake_climate": ["temp", "slope"]}])

        with pytest.raises(ValueError) as error_info:
            extract(three_points, run_config, output_dir=tmp_path)

        message = str(error_info.value)
        assert message.startswith(
            "Output 'fake_run': dataset 'fake_climate' has data_source 'fake', whose adapter "
            "cannot compute the derived band(s) ['slope']."
        )
        assert "Derived bands that this adapter computes: none." in message
        assert "Remove ['slope'] from the band list of 'fake_climate'." in message
        assert fake_provider.built_specs == []
        assert not any(tmp_path.iterdir())

    def test_failing_check_dataset_stops_the_call(self, fake_provider, three_points, tmp_path):
        """The ``ValueError`` of the fake adapter's ``check_dataset`` reaches the caller."""
        _register_fake_datasets()

        with pytest.raises(ValueError) as error_info:
            extract(
                three_points,
                _fake_tabular_config("fake_run", ["fake_remote"]),
                output_dir=tmp_path,
            )

        assert str(error_info.value).startswith(
            "Output 'fake_run': dataset 'fake_remote': the path "
            "'https://example.org/climate.tif' is not an address of the fake provider."
        )
        assert [context for _, context in fake_provider.checked_specs] == [
            "Output 'fake_run': dataset 'fake_remote'"
        ]
        assert fake_provider.built_specs == []
        assert not any(tmp_path.iterdir())

    @pytest.mark.parametrize(
        "invalid_run_config",
        [
            pytest.param(_fake_raster_config("second"), id="raster"),
            pytest.param(
                _fake_tabular_config("second", ["fake_climate"], ["median"]),
                id="unsupported_reducer",
            ),
            pytest.param(
                _fake_tabular_config("second", [{"fake_climate": ["slope"]}]),
                id="derived_band",
            ),
            pytest.param(_fake_tabular_config("second", ["fake_remote"]), id="failing_check"),
        ],
    )
    def test_invalid_second_run_config_stops_the_call_before_any_build(
        self, fake_provider, three_points, tmp_path, invalid_run_config
    ):
        """An invalid second run config raises before the adapter of the valid first one is built."""
        _register_fake_datasets()
        run_configs = [_fake_tabular_config("first", ["fake_climate"]), invalid_run_config]

        with pytest.raises(ValueError, match="^Output 'second': dataset 'fake_"):
            extract(three_points, run_configs, output_dir=tmp_path)

        assert fake_provider.built_specs == []
        assert not any(tmp_path.iterdir())


def _fake_climate_entry(**keys: Any) -> dict[str, Any]:
    """Return the catalog entry of ``fake_climate``, with more ``keys``."""
    return {
        "data_source": "fake",
        "path": "fake://climate",
        "data_type": "continuous",
        **keys,
    }


def _write_catalog_file(directory: Path, datasets: dict[str, Any]) -> Path:
    """Write ``datasets`` as a catalog file in ``directory`` and return its path."""
    catalog_path = directory / "catalog.yml"
    catalog_path.write_text(yaml.safe_dump({"datasets": datasets}), encoding="utf-8")
    return catalog_path


class TestCatalogRules:
    """The fake provider's ``catalog_keys`` and catalog hooks decide its catalog rules.

    ``envoi.catalog`` checks an entry when it is loaded or registered, so an
    invalid entry stops before ``extract()`` and before any adapter is built.
    """

    def test_own_catalog_key_is_accepted(self, fake_provider):
        """The key that the fake adapter declares in ``catalog_keys`` registers unchanged."""
        update_catalog({"datasets": {"fake_climate": _fake_climate_entry(fake_product="era")}})

        stored_entry = load_catalogs()["datasets"]["fake_climate"]
        assert stored_entry == _fake_climate_entry(fake_product="era")

    def test_unknown_catalog_key_is_rejected(self, fake_provider):
        """A key that is not generic and not in ``catalog_keys`` raises and lists allowed keys."""
        with pytest.raises(CatalogError) as error_info:
            update_catalog({"datasets": {"fake_climate": _fake_climate_entry(fake_prodcut="era")}})

        message = str(error_info.value)
        assert message.startswith(
            "datasets.fake_climate: unknown key(s) ['fake_prodcut'] for data_source 'fake'. "
            "Allowed keys: ["
        )
        assert "'fake_product'" in message
        assert "Put free text, for example notes about the dataset, under " in message
        assert "fake_climate" not in load_catalogs()["datasets"]
        assert fake_provider.checked_specs == []
        assert fake_provider.built_specs == []

    def test_catalog_key_of_another_adapter_is_rejected(self, fake_provider):
        """A key of the local adapter (``nodata``) is an unknown key for the fake provider."""
        with pytest.raises(CatalogError, match=r"unknown key\(s\) \['nodata'\]"):
            update_catalog({"datasets": {"fake_climate": _fake_climate_entry(nodata=-9999)}})

    def test_value_error_of_the_hook_becomes_a_catalog_error(self, fake_provider, tmp_path):
        """A ``ValueError`` of the hook becomes a ``CatalogError`` that names the dataset.

        A catalog file and a catalog dict give the same message.
        """
        entry = _fake_climate_entry(fake_product=3)
        expected_message = (
            "datasets.fake_climate: fake_product must be the name of a product as text, got 3."
        )

        with pytest.raises(CatalogError) as dict_error_info:
            update_catalog({"datasets": {"fake_climate": entry}})
        with pytest.raises(CatalogError) as file_error_info:
            load_catalog(_write_catalog_file(tmp_path, {"fake_climate": entry}))

        for error_info in (dict_error_info, file_error_info):
            assert str(error_info.value) == expected_message
            assert type(error_info.value.__cause__) is ValueError
        assert "fake_climate" not in load_catalogs()["datasets"]

    def test_catalog_error_of_the_hook_is_not_wrapped_again(self, isolated_registry):
        """A ``CatalogError`` of a hook reaches the caller unchanged, without a second prefix."""

        def reject_every_entry(cls, name: str, entry: dict) -> None:
            raise CatalogError(f"datasets.{name}: the fake provider rejects every entry.")

        register(
            "fake_strict",
            _fake_adapter_with(validate_catalog_entry=classmethod(reject_every_entry)),
        )

        with pytest.raises(CatalogError) as error_info:
            update_catalog(
                {"datasets": {"fake_climate": _fake_climate_entry(data_source="fake_strict")}}
            )

        assert str(error_info.value) == (
            "datasets.fake_climate: the fake provider rejects every entry."
        )
        assert error_info.value.__cause__ is None

    def test_inspection_runs_for_catalog_files_only(self, isolated_registry, tmp_path):
        """``inspect_catalog_entry`` fills a key for a catalog file, not for a catalog dict.

        The hook runs once, for the file, and the stored entry is the dict
        that it returns.
        """
        inspected_inputs: list[dict[str, Any]] = []

        def fill_fake_product(cls, name: str, entry: dict) -> dict:
            inspected_inputs.append(entry)
            return {**entry, "fake_product": f"read from {entry['path']}"}

        register(
            "fake_inspected",
            _fake_adapter_with(inspect_catalog_entry=classmethod(fill_fake_product)),
        )
        entry = _fake_climate_entry(data_source="fake_inspected")

        file_entry = load_catalog(_write_catalog_file(tmp_path, {"fake_climate": entry}))[
            "datasets"
        ]["fake_climate"]
        update_catalog({"datasets": {"fake_climate": entry}})
        dict_entry = load_catalogs()["datasets"]["fake_climate"]

        assert file_entry == {**entry, "fake_product": "read from fake://climate"}
        assert dict_entry == entry
        assert inspected_inputs == [entry]


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


# Variants of the fake adapter that break one rule of the result contract
# ("Result contract" in the BaseAdapter docstring). Each one changes the
# valid result of the fake adapter.


class TooFewResultsAdapter(FakeStatsAdapter):
    """Returns no result for the last point."""

    def fetch_stats_batch(self, *args: Any, **kwargs: Any) -> Any:
        return super().fetch_stats_batch(*args, **kwargs)[:-1]


class NonPairItemAdapter(FakeStatsAdapter):
    """Returns the result of the second point as a tuple of three items."""

    def fetch_stats_batch(self, *args: Any, **kwargs: Any) -> Any:
        results = super().fetch_stats_batch(*args, **kwargs)
        stats, meta = results[1]
        results[1] = (stats, meta, {})
        return results


class MissingMetaKeyAdapter(FakeStatsAdapter):
    """Returns meta dicts without the required key ``coverage_pct``."""

    def fetch_stats_batch(self, *args: Any, **kwargs: Any) -> Any:
        results = super().fetch_stats_batch(*args, **kwargs)
        for _, meta in results:
            del meta["coverage_pct"]
        return results


class NonDictDatasetMetaAdapter(FakeStatsAdapter):
    """Returns the dataset entry as a list of (key, value) pairs."""

    def build_dataset_meta(self, *args: Any, **kwargs: Any) -> Any:
        return list(super().build_dataset_meta(*args, **kwargs).items())


class ShortPathsAdapter(FakeRasterAdapter):
    """Returns no tile path for the last point."""

    def export_tiles(self, *args: Any, **kwargs: Any) -> Any:
        paths, meta_list = super().export_tiles(*args, **kwargs)
        return paths[:-1], meta_list


class TextPathAdapter(FakeRasterAdapter):
    """Returns each tile path as text, not as a ``pathlib.Path``."""

    def export_tiles(self, *args: Any, **kwargs: Any) -> Any:
        paths, meta_list = super().export_tiles(*args, **kwargs)
        return [None if path is None else str(path) for path in paths], meta_list


def _contract_error_message(adapter_class: type[BaseAdapter], method_name: str, rule: str) -> str:
    """Return the ``RuntimeError`` message for a result of ``fake_climate`` that breaks ``rule``."""
    return (
        f"Dataset 'fake_climate': {method_name} of the adapter {adapter_class.__qualname__} "
        f"returned a result that breaks the adapter result contract: {rule}. This is a bug "
        f"in the adapter, not a problem with your input table or run config. Please report "
        f"it to the envoi developers."
    )


class TestAdapterResultChecks:
    """``extract()`` checks each adapter result before it uses it.

    A result that breaks the result contract raises ``RuntimeError`` that
    names the dataset, the adapter class, the call, and the broken rule, and
    the call writes no output.
    """

    @pytest.mark.parametrize(
        ("adapter_class", "method_name", "rule"),
        [
            pytest.param(
                TooFewResultsAdapter,
                "fetch_stats_batch",
                "it must return one (stats, meta) pair for each point, but returned 2 item(s) "
                "for 3 point(s)",
                id="too_few_results",
            ),
            pytest.param(
                NonPairItemAdapter,
                "fetch_stats_batch",
                "each item must be a (stats, meta) pair, but item 1 is a tuple of 3 item(s)",
                id="non_pair_item",
            ),
            pytest.param(
                MissingMetaKeyAdapter,
                "fetch_stats_batch",
                "each meta dict must have the keys ['in_extent', 'n_pixels', 'had_nodata', "
                "'coverage_pct'], but the meta dict of item 0 has no ['coverage_pct']",
                id="missing_meta_key",
            ),
            pytest.param(
                NonDictDatasetMetaAdapter,
                "build_dataset_meta",
                "it must return a dict, but returned a list",
                id="non_dict_dataset_meta",
            ),
        ],
    )
    def test_broken_tabular_result_raises(
        self, isolated_registry, three_points, tmp_path, adapter_class, method_name, rule
    ):
        """A tabular result that breaks one rule raises ``RuntimeError`` that names the adapter."""
        _register_fake_variant(adapter_class)

        with pytest.raises(RuntimeError) as error_info:
            extract(
                three_points,
                _fake_tabular_config("broken", ["fake_climate"]),
                output_dir=tmp_path,
            )

        assert str(error_info.value) == _contract_error_message(adapter_class, method_name, rule)
        assert not any(tmp_path.iterdir())

    @pytest.mark.parametrize(
        ("adapter_class", "rule"),
        [
            pytest.param(
                ShortPathsAdapter,
                "paths must have one entry for each point, but has 2 for 3 point(s)",
                id="short_paths",
            ),
            pytest.param(
                TextPathAdapter,
                "each path must be a pathlib.Path or None, but path 0 is a str",
                id="text_path",
            ),
        ],
    )
    def test_broken_raster_result_raises(
        self, isolated_registry, three_points, tmp_path, adapter_class, rule
    ):
        """A raster result that breaks one rule raises ``RuntimeError`` that names the adapter."""
        _register_fake_variant(adapter_class)

        with pytest.raises(RuntimeError) as error_info:
            extract(three_points, _fake_raster_config("broken_tiles"), output_dir=tmp_path)

        assert str(error_info.value) == _contract_error_message(adapter_class, "export_tiles", rule)
        assert not any(tmp_path.iterdir())


# Variants of the fake adapter that record their lifetime: the build, the
# window of each statistics request, and each close() call.


class WindowRecordingAdapter(FakeStatsAdapter):
    """Records ``"build"``, ``"fetch <window> m"``, and ``"close"`` in ``events``, in call order."""

    # The events of every instance of this class and its subclasses. Each test clears it.
    events: ClassVar[list[str]] = []

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__(spec)
        WindowRecordingAdapter.events.append("build")

    def fetch_stats_batch(
        self, lats: Any, lons: Any, window_m: int, reducer_names: Any, **kwargs: Any
    ) -> Any:
        WindowRecordingAdapter.events.append(f"fetch {window_m} m")
        return super().fetch_stats_batch(lats, lons, window_m, reducer_names, **kwargs)

    def close(self) -> None:
        WindowRecordingAdapter.events.append("close")


class SecondWindowFailsAdapter(WindowRecordingAdapter):
    """Raises ``RuntimeError`` in ``fetch_stats_batch`` for the 200 m window."""

    def fetch_stats_batch(
        self, lats: Any, lons: Any, window_m: int, reducer_names: Any, **kwargs: Any
    ) -> Any:
        results = super().fetch_stats_batch(lats, lons, window_m, reducer_names, **kwargs)
        if window_m == 200:
            raise RuntimeError("stub: the request of the 200 m window failed")
        return results


class TestAdapterLifetime:
    """``extract()`` builds one adapter for each (run config, dataset) pair.

    The adapter serves every window size of the pair, and its ``close()``
    runs once, after the last window, also when a window raises.
    """

    def test_three_window_sizes_build_one_adapter(self, fake_provider, three_points, tmp_path):
        """Three window sizes build the fake adapter once, and each window has its own columns."""
        _register_fake_datasets()
        run_config = _fake_tabular_config(
            "fake_windows", ["fake_climate"], ["mean", "max"], window_size_m=[100, 200, 300]
        )

        outputs = extract(three_points, run_config, output_dir=tmp_path)

        assert fake_provider.built_specs == [_fake_climate_entry()]

        # One statistic column for each window, band, and reducer, in this order.
        stats_table = pd.read_csv(outputs["fake_windows"])
        assert [column for column in stats_table.columns if column.startswith("fake_climate_")] == [
            f"fake_climate_{band}_{reducer}_{window_size}m"
            for window_size in (100, 200, 300)
            for band in ("temp", "precip")
            for reducer in ("mean", "max")
        ]
        # The fake statistics do not depend on the window: mean = band base
        # value + lat; max = mean + lon.
        expected_values = {
            "temp_mean": [10.0, 20.0, 30.0],
            "temp_max": [11.0, 22.0, 33.0],
            "precip_mean": [110.0, 120.0, 130.0],
            "precip_max": [111.0, 122.0, 133.0],
        }
        for window_size in (100, 200, 300):
            for statistic_key, values in expected_values.items():
                column = f"fake_climate_{statistic_key}_{window_size}m"
                assert stats_table[column].tolist() == values, column

        # The QC columns and the quality summary of each window.
        qc_table = pd.read_csv(tmp_path / "fake_windows_qc.csv")
        assert [column for column in qc_table.columns if column.startswith("fake_climate_")] == [
            f"fake_climate_{qc_name}_{window_size}m"
            for window_size in (100, 200, 300)
            for qc_name in ("in_extent", "n_pixels", "had_nodata", "coverage_pct")
        ]
        sidecar = _read_sidecar(tmp_path / "fake_windows_metadata.json")
        quality = sidecar["datasets"]["fake_climate"]["quality"]
        assert list(quality) == ["100m", "200m", "300m"]
        assert [quality[window_key]["n_full"] for window_key in quality] == [3, 3, 3]

    def test_two_run_configs_build_one_adapter_each(self, fake_provider, three_points, tmp_path):
        """Two run configs with the same dataset build one adapter each, with their own bands."""
        _register_fake_datasets()
        run_configs = [
            _fake_tabular_config("all_bands", ["fake_climate"], window_size_m=[100, 200]),
            _fake_tabular_config(
                "precip_only", [{"fake_climate": ["precip"]}], window_size_m=[100, 200]
            ),
        ]

        outputs = extract(three_points, run_configs, output_dir=tmp_path)

        assert fake_provider.built_specs == [
            _fake_climate_entry(),
            _fake_climate_entry(bands=["precip"], band_naming="per_band"),
        ]
        all_bands_table = pd.read_csv(outputs["all_bands"])
        assert [
            column for column in all_bands_table.columns if column.startswith("fake_climate_")
        ] == [
            "fake_climate_temp_mean_100m",
            "fake_climate_precip_mean_100m",
            "fake_climate_temp_mean_200m",
            "fake_climate_precip_mean_200m",
        ]
        precip_table = pd.read_csv(outputs["precip_only"])
        assert [
            column for column in precip_table.columns if column.startswith("fake_climate_")
        ] == ["fake_climate_precip_mean_100m", "fake_climate_precip_mean_200m"]
        assert precip_table["fake_climate_precip_mean_200m"].tolist() == [110.0, 120.0, 130.0]

    def test_close_runs_once_after_the_last_window(self, isolated_registry, three_points, tmp_path):
        """``close()`` runs once, after the statistics requests of all three windows."""
        WindowRecordingAdapter.events.clear()
        _register_fake_variant(WindowRecordingAdapter)
        run_config = _fake_tabular_config("closed", ["fake_climate"], window_size_m=[100, 200, 300])

        extract(three_points, run_config, output_dir=tmp_path)

        assert WindowRecordingAdapter.events == [
            "build",
            "fetch 100 m",
            "fetch 200 m",
            "fetch 300 m",
            "close",
        ]

    def test_close_runs_once_when_a_window_raises(self, isolated_registry, three_points, tmp_path):
        """When the 200 m window raises, ``close()`` runs once, and the error reaches the caller."""
        WindowRecordingAdapter.events.clear()
        _register_fake_variant(SecondWindowFailsAdapter)
        run_config = _fake_tabular_config("failed", ["fake_climate"], window_size_m=[100, 200, 300])

        with pytest.raises(RuntimeError, match="^stub: the request of the 200 m window failed$"):
            extract(three_points, run_config, output_dir=tmp_path)

        assert WindowRecordingAdapter.events == ["build", "fetch 100 m", "fetch 200 m", "close"]
        assert not any(tmp_path.iterdir())
