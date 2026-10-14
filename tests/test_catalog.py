"""Unit tests for envoi.catalog — catalog validation and user-catalog state.

These cover the error paths that turn invalid YAML / dicts into a clear
CatalogError, plus the accumulate/reset semantics of update_catalog() and
reset_catalog(). Hitting them directly is far faster than exercising them
through the full extract() pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from envoi import adapters as adapter_registry
from envoi import list_datasets, reset_catalog, update_catalog
from envoi.catalog import (
    BUILTIN_EE_CATALOG,
    CatalogError,
    _load_catalog_any,
    load_catalog,
    load_catalogs,
)

# ------------------------------------------------------------------
# Shared cleanup: every test starts with an empty user catalog. Tests that
# call update_catalog() leak state at the module level, so the autouse
# fixture resets after each one to keep them independent.
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_user_catalog_after_test():
    # Setup runs nothing — yield immediately so the test sees a clean slate
    # (assuming previous test cleaned up). Teardown clears anything this test
    # added so the next test isn't polluted.
    yield
    reset_catalog()


# ------------------------------------------------------------------
# _load_catalog_any — dict input validation.
# ------------------------------------------------------------------


class TestCatalogDictValidation:
    """Error paths when a dict is passed in (the most common entry point)."""

    def test_missing_data_source_raises(self):
        # Every dataset spec needs data_source; the error must name the field.
        with pytest.raises(CatalogError, match="data_source"):
            _load_catalog_any({"datasets": {"x": {"path": "data/file.tif"}}})

    def test_missing_data_source_lists_the_registered_data_sources(self, monkeypatch):
        """The message lists the data sources of the adapter registry, not a fixed list."""
        monkeypatch.setitem(adapter_registry._REG, "test_source", object)

        with pytest.raises(
            CatalogError,
            match="Valid values are: earth_engine, local, test_source.",
        ):
            _load_catalog_any({"datasets": {"x": {"path": "data/file.tif"}}})

    def test_missing_path_raises(self):
        # Every dataset spec also needs path. Same shape as the data_source check.
        with pytest.raises(CatalogError, match="path"):
            _load_catalog_any({"datasets": {"x": {"data_source": "local"}}})

    def test_empty_path_string_raises(self):
        # Empty string is treated the same as missing — a path that points
        # nowhere is useless and we'd rather fail loudly than silently later.
        with pytest.raises(CatalogError, match="path"):
            _load_catalog_any({"datasets": {"x": {"data_source": "local", "path": ""}}})

    def test_earth_engine_requires_data_type(self):
        # EE catalog entries must declare data_type up front so the right
        # reducer set (continuous vs categorical) is selected. Local rasters
        # may omit it because the type can be inferred from the raster file.
        with pytest.raises(CatalogError, match="data_type"):
            _load_catalog_any(
                {"datasets": {"x": {"data_source": "earth_engine", "path": "USGS/GTOPO30"}}}
            )

    def test_local_dataset_does_not_require_data_type(self):
        # Confirm the looser local-source rule — no data_type, no error.
        # The path is fake so _inspect_raster will silently skip auto-detect,
        # but the validator itself must accept the entry.
        catalog = _load_catalog_any(
            {"datasets": {"x": {"data_source": "local", "path": "/tmp/nonexistent.tif"}}}
        )
        assert "x" in catalog["datasets"]

    def test_non_dict_spec_raises(self):
        # Each value under "datasets" must itself be a mapping; a list/string/
        # number is rejected with a clear "must be a mapping" message.
        with pytest.raises(CatalogError, match="must be a mapping"):
            _load_catalog_any({"datasets": {"x": ["not", "a", "dict"]}})

    def test_none_input_returns_empty_catalog(self):
        # Internal helpers pass None when a catalog source is omitted; this
        # must not raise — just return an empty normalised structure.
        result = _load_catalog_any(None)
        assert result == {"datasets": {}}

    def test_dict_without_datasets_key_normalised(self):
        # A user dict that forgets the top-level "datasets" key is still
        # accepted; the helper adds an empty datasets dict so downstream
        # code can iterate without a key-check.
        result = _load_catalog_any({"other_key": "ignored"})
        assert result["datasets"] == {}


# ------------------------------------------------------------------
# load_catalog — file-system entry point.
# ------------------------------------------------------------------


class TestLoadCatalogFile:
    """Error paths when a catalog YAML is loaded from disk."""

    def test_missing_file_raises(self, tmp_path):
        # A nonexistent path produces a CatalogError, not FileNotFoundError —
        # so callers only have to catch one exception type.
        with pytest.raises(CatalogError, match="not found"):
            load_catalog(tmp_path / "does_not_exist.yml")

    def test_invalid_yaml_raises(self, tmp_path):
        # Tab indentation under a key — pyyaml rejects this with a
        # ScannerError, which load_catalog wraps as CatalogError.
        bad_yaml_path = tmp_path / "bad.yml"
        bad_yaml_path.write_text("datasets:\n\tfoo: bar\n")
        with pytest.raises(CatalogError, match="parse error"):
            load_catalog(bad_yaml_path)

    def test_top_level_list_raises(self, tmp_path):
        # YAML where the root is a list (not a mapping) — caught by the
        # _validate_catalog isinstance(dict) check, not the yaml parser.
        list_root_path = tmp_path / "list_root.yml"
        list_root_path.write_text("- foo\n- bar\n")
        with pytest.raises(CatalogError, match="must be a mapping"):
            load_catalog(list_root_path)

    def test_missing_datasets_key_raises(self, tmp_path):
        # Root is a dict but does not contain the required "datasets" key.
        no_datasets_path = tmp_path / "no_datasets.yml"
        no_datasets_path.write_text("something_else: value\n")
        with pytest.raises(CatalogError, match="datasets"):
            load_catalog(no_datasets_path)

    def test_empty_datasets_mapping_raises(self, tmp_path):
        # A "datasets:" key with no children — the validator requires at
        # least one entry so callers can rely on a non-empty mapping.
        empty_datasets_path = tmp_path / "empty_datasets.yml"
        empty_datasets_path.write_text("datasets: {}\n")
        with pytest.raises(CatalogError, match="non-empty mapping"):
            load_catalog(empty_datasets_path)


# ------------------------------------------------------------------
# update_catalog / reset_catalog — session-wide user catalog.
# ------------------------------------------------------------------


class TestUpdateAndResetCatalog:
    """Cover the accumulate-then-clear contract of the user catalog."""

    def test_update_then_reset_round_trip(self):
        # After update_catalog adds a dataset it must appear in load_catalogs;
        # after reset_catalog the same query must no longer find it.
        update_catalog({"datasets": {"my_local": {"data_source": "local", "path": "/tmp/x.tif"}}})
        merged = load_catalogs()
        assert "my_local" in merged["datasets"]

        reset_catalog()
        merged_after_reset = load_catalogs()
        assert "my_local" not in merged_after_reset["datasets"]

    def test_update_is_cumulative_across_calls(self):
        # Two separate update_catalog calls should accumulate, not replace.
        # This is what allows users to register datasets piecemeal during
        # an interactive session.
        update_catalog({"datasets": {"a": {"data_source": "local", "path": "/tmp/a.tif"}}})
        update_catalog({"datasets": {"b": {"data_source": "local", "path": "/tmp/b.tif"}}})
        merged = load_catalogs()
        assert {"a", "b"} <= set(merged["datasets"].keys())

    def test_update_with_same_key_overwrites(self):
        # When two updates share a dataset name, the later one wins — same
        # semantics as dict.update. Documented behaviour, but worth pinning.
        update_catalog({"datasets": {"x": {"data_source": "local", "path": "/tmp/old.tif"}}})
        update_catalog({"datasets": {"x": {"data_source": "local", "path": "/tmp/new.tif"}}})
        merged = load_catalogs()
        assert merged["datasets"]["x"]["path"] == "/tmp/new.tif"

    def test_update_propagates_validation_errors(self):
        # update_catalog goes through _load_catalog_any, so a bad spec must
        # raise up to the caller — not silently get accepted and break later.
        with pytest.raises(CatalogError, match="data_source"):
            update_catalog({"datasets": {"bad": {"path": "/tmp/missing_source.tif"}}})

    def test_update_from_yaml_file(self, tmp_path):
        # update_catalog accepts a file path as well as a dict argument; the
        # YAML route is the typical workflow for sharing catalog snippets.
        yml_path = tmp_path / "extra.yml"
        yml_path.write_text(
            "datasets:\n" "  yml_local:\n" "    data_source: local\n" "    path: /tmp/yml.tif\n"
        )
        update_catalog(yml_path)
        merged = load_catalogs()
        assert "yml_local" in merged["datasets"]


# ------------------------------------------------------------------
# Unknown data_source — rejected when a catalog is registered or loaded.
# ------------------------------------------------------------------


class TestUnknownDataSource:
    """A ``data_source`` without a registered adapter fails before ``extract()`` runs."""

    def test_update_catalog_dict_raises_and_registers_nothing(self):
        """A dict entry with an unknown ``data_source`` raises and is not registered."""
        dataset_entry = {"data_source": "gee", "path": "USGS/GTOPO30", "data_type": "continuous"}

        expected_message = "datasets.x: unknown data_source 'gee'"
        with pytest.raises(CatalogError, match=expected_message) as error_info:
            update_catalog({"datasets": {"x": dataset_entry}})

        assert "earth_engine, local" in str(error_info.value)
        assert "x" not in load_catalogs()["datasets"]

    def test_load_catalog_file_raises(self, tmp_path):
        """A YAML entry with an unknown ``data_source`` raises ``CatalogError``."""
        catalog_path = tmp_path / "catalog.yml"
        catalog_path.write_text(
            yaml.safe_dump({"datasets": {"my_dem": {"data_source": "gdal", "path": "dem.tif"}}}),
            encoding="utf-8",
        )

        expected_message = "datasets.my_dem: unknown data_source 'gdal'"
        with pytest.raises(CatalogError, match=expected_message) as error_info:
            load_catalog(catalog_path)

        assert "earth_engine, local" in str(error_info.value)


# ------------------------------------------------------------------
# Unknown collection_date_policy — rejected when a catalog is registered or loaded.
# ------------------------------------------------------------------


def _earth_engine_entry(collection_date_policy) -> dict:
    """Return an Earth Engine catalog entry with ``dataset_spec.collection_date_policy``."""
    return {
        "data_source": "earth_engine",
        "path": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "data_type": "continuous",
        "dataset_spec": {"collection_date_policy": collection_date_policy},
    }


class TestUnknownCollectionDatePolicy:
    """A ``collection_date_policy`` other than nearest, contains, or mosaic fails at registration."""

    @pytest.mark.parametrize("policy", ["nearst", "interval", "", None, 1])
    def test_update_catalog_dict_raises_and_registers_nothing(self, policy):
        """A dict entry with an unknown policy raises, names the value, and is not registered."""
        with pytest.raises(CatalogError) as error_info:
            update_catalog({"datasets": {"era5": _earth_engine_entry(policy)}})

        message = str(error_info.value)
        assert message.startswith(
            f"datasets.era5: unknown dataset_spec.collection_date_policy {policy!r}."
        )
        assert message.endswith("Valid values are: contains, mosaic, nearest.")
        assert "era5" not in load_catalogs()["datasets"]

    def test_load_catalog_file_raises(self, tmp_path):
        """A YAML entry with an unknown policy raises ``CatalogError``."""
        catalog_path = tmp_path / "catalog.yml"
        catalog_path.write_text(
            yaml.safe_dump({"datasets": {"era5": _earth_engine_entry("closest")}}),
            encoding="utf-8",
        )

        expected_message = "datasets.era5: unknown dataset_spec.collection_date_policy 'closest'"
        with pytest.raises(CatalogError, match=expected_message):
            load_catalog(catalog_path)

    @pytest.mark.parametrize("policy", ["nearest", "contains", "mosaic", "Contains", "MOSAIC"])
    def test_known_policies_are_accepted_in_any_letter_case(self, policy):
        """The three policies register, also in another letter case, as the adapter reads them."""
        update_catalog({"datasets": {"era5": _earth_engine_entry(policy)}})

        registered_spec = load_catalogs()["datasets"]["era5"]["dataset_spec"]
        assert registered_spec["collection_date_policy"] == policy

    def test_builtin_catalog_loads_with_the_check(self):
        """The built-in catalog loads, and each of its entries uses a known policy."""
        builtin_datasets = load_catalogs(BUILTIN_EE_CATALOG)["datasets"]

        for dataset_name, dataset_entry in builtin_datasets.items():
            dataset_spec = dataset_entry.get("dataset_spec") or {}
            policy = dataset_spec.get("collection_date_policy", "nearest")
            assert policy in {"nearest", "contains", "mosaic"}, dataset_name


# ------------------------------------------------------------------
# load_catalogs — merge precedence and source-flattening.
# ------------------------------------------------------------------


class TestLoadCatalogsMerge:
    def test_user_catalog_overrides_supplied_source(self):
        # User-registered entries are applied as the *final* merge layer, so
        # they always beat any caller-supplied catalog passed positionally.
        # This is the rule that makes interactive workflows predictable.
        update_catalog({"datasets": {"x": {"data_source": "local", "path": "/tmp/user.tif"}}})
        supplied_catalog = {
            "datasets": {"x": {"data_source": "local", "path": "/tmp/supplied.tif"}}
        }
        merged = load_catalogs(supplied_catalog)
        assert merged["datasets"]["x"]["path"] == "/tmp/user.tif"

    def test_later_supplied_source_overrides_earlier(self):
        # Among positional sources, later args override earlier ones on the
        # per-dataset key. (Both are then beaten by the user catalog — which
        # is empty in this test so the rule is observable.)
        merged = load_catalogs(
            {"datasets": {"x": {"data_source": "local", "path": "/tmp/first.tif"}}},
            {"datasets": {"x": {"data_source": "local", "path": "/tmp/second.tif"}}},
        )
        assert merged["datasets"]["x"]["path"] == "/tmp/second.tif"

    def test_list_of_sources_is_flattened(self):
        # A list/tuple passed as one positional argument is flattened, so the
        # caller can build up a list of sources programmatically and pass it
        # in a single call.
        merged = load_catalogs(
            [
                {"datasets": {"a": {"data_source": "local", "path": "/tmp/a.tif"}}},
                {"datasets": {"b": {"data_source": "local", "path": "/tmp/b.tif"}}},
            ]
        )
        assert {"a", "b"} <= set(merged["datasets"].keys())

    def test_builtin_sentinel_loads_packaged_ee_catalog(self):
        # The BUILTIN_EE_CATALOG sentinel triggers loading the
        # configs/ee_catalog.yml file that ships with the package. A non-empty
        # datasets dict is the minimum signal that the bundled file was found
        # and parsed.
        merged = load_catalogs(BUILTIN_EE_CATALOG)
        assert merged["datasets"], "builtin EE catalog should not be empty"


# ------------------------------------------------------------------
# list_datasets — verbosity levels, merged-catalog view, error handling.
# ------------------------------------------------------------------


class TestListDatasets:
    """list_datasets() surfaces the merged catalog at three verbosity levels.

    These tests use capsys to assert both the printed output and the returned
    object — list_datasets() is documented to do both, so each test checks
    that both halves of the contract hold.
    """

    def test_names_returns_sorted_list_of_strings(self):
        # Default verbosity is "names" — should return a sorted list of
        # dataset keys.
        result = list_datasets()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)
        # Sorted is part of the contract so output is deterministic.
        assert result == sorted(result)

    def test_names_includes_a_known_builtin_dataset(self):
        # Smoke check that the merged view actually exposes built-in
        # datasets (not just an empty list). dem_copernicus_glo30 is a
        # stable entry in the bundled catalog used elsewhere in the
        # README/tests.
        result = list_datasets("names")
        assert "dem_copernicus_glo30" in result

    def test_info_returns_records_with_expected_fields(self):
        # "info" verbosity returns a list of dicts with the human-readable
        # metadata fields. Missing fields are stored as None so callers can
        # filter consistently without KeyError surprises.
        records = list_datasets("info")
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)
        expected_fields = {
            "name",
            "display_name",
            "data_source",
            "data_type",
            "category",
            "description",
            "citation",
            "ee_source_url",
            "source_url",
        }
        # Every record must expose the full info-level field set, even if
        # some values are None — the shape is part of the contract.
        for record in records:
            assert expected_fields <= set(record.keys())

    def test_info_display_name_falls_back_to_the_dataset_name(self):
        # display_name is optional in a catalog entry, but the info listing
        # always has to hand callers something printable — menus and docs
        # would otherwise show a blank label.
        update_catalog(
            {
                "datasets": {
                    "unlabelled_dataset": {
                        "data_source": "local",
                        "path": "/tmp/nonexistent.tif",
                    }
                }
            }
        )
        records = {record["name"]: record for record in list_datasets("info")}
        assert records["unlabelled_dataset"]["display_name"] == "unlabelled_dataset"
        # Built-in entries carry a real label instead.
        assert records["dem_copernicus_glo30"]["display_name"] == "Copernicus DEM GLO-30"

    def test_full_returns_complete_entries(self):
        # "full" verbosity returns the entire catalog entry plus a "name"
        # field. For at least one known dataset we expect to see the
        # top-level keys present in the YAML (path, data_source, etc.).
        records = list_datasets("full")
        by_name = {r["name"]: r for r in records}
        dem_entry = by_name["dem_copernicus_glo30"]
        assert dem_entry["data_source"] == "earth_engine"
        assert "path" in dem_entry

    def test_includes_user_registered_datasets(self):
        # User-registered datasets should show up alongside built-ins —
        # this matches what extract() sees, so the listing is honest.
        update_catalog(
            {
                "datasets": {
                    "my_test_dataset": {
                        "data_source": "local",
                        "path": "/tmp/nonexistent.tif",
                    }
                }
            }
        )
        names = list_datasets("names")
        assert "my_test_dataset" in names

    def test_invalid_verbosity_raises_value_error(self):
        # A typo in verbosity should fail loudly, not silently default —
        # the error message must surface the supported levels so the
        # caller can fix the typo without reading the source.
        with pytest.raises(ValueError, match="verbosity must be one of"):
            list_datasets("verbose")


# ------------------------------------------------------------------
# Local raster inspection — resolution_m in metres
# ------------------------------------------------------------------


def _write_one_band_raster(path, crs: str, pixel_width: float) -> None:
    """Write a 10 × 10 float32 raster with square pixels of ``pixel_width`` CRS units."""
    random_generator = np.random.default_rng(seed=21)
    pixel_values = random_generator.uniform(0.0, 100.0, (10, 10)).astype(np.float32)
    # Origins inside the valid range of each CRS; the values do not matter.
    origin_x, origin_y = (18.0, 63.0) if crs == "EPSG:4326" else (500_000.0, 1_000_000.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(origin_x, origin_y, pixel_width, pixel_width),
    ) as dst:
        dst.write(pixel_values, 1)


def _load_local_catalog_file(tmp_path, dataset_spec: dict) -> dict:
    """Write a catalog YAML with one local dataset, load it, and return the dataset's entry.

    A catalog file runs the local raster inspection (``load_catalog``).
    """
    catalog_path = tmp_path / "catalog.yml"
    catalog_path.write_text(
        yaml.safe_dump({"datasets": {"inspected": dataset_spec}}), encoding="utf-8"
    )
    return load_catalog(catalog_path)["datasets"]["inspected"]


class TestLocalRasterResolution:
    """``resolution_m`` that local raster inspection fills in is in metres for every CRS."""

    @pytest.mark.parametrize(
        ("crs", "pixel_width", "expected_resolution_m"),
        [
            # Geographic: 0.001° × 111 319.49 m per degree (one degree at the equator).
            ("EPSG:4326", 0.001, 111.31949),
            # Projected in metres: unchanged.
            ("EPSG:32634", 30.0, 30.0),
            # Projected in US survey feet: 1 ft = 1200 / 3937 m (about 0.3048 m).
            ("EPSG:2263", 100.0, 100.0 * 1200 / 3937),
        ],
    )
    def test_resolution_m_is_in_metres(self, tmp_path, crs, pixel_width, expected_resolution_m):
        """The auto-detected ``resolution_m`` is the pixel width converted to metres."""
        raster_path = tmp_path / "raster.tif"
        _write_one_band_raster(raster_path, crs, pixel_width)

        dataset_entry = _load_local_catalog_file(
            tmp_path, {"data_source": "local", "path": str(raster_path)}
        )

        assert dataset_entry["resolution_m"] == pytest.approx(expected_resolution_m)

    def test_resolution_m_from_the_catalog_is_kept(self, tmp_path):
        """A ``resolution_m`` set in the catalog entry is not replaced."""
        raster_path = tmp_path / "raster.tif"
        _write_one_band_raster(raster_path, "EPSG:4326", 0.001)

        dataset_entry = _load_local_catalog_file(
            tmp_path, {"data_source": "local", "path": str(raster_path), "resolution_m": 100}
        )

        assert dataset_entry["resolution_m"] == 100
