"""Unit tests for envoi.config — catalog validation and user-catalog state.

These cover the error paths that turn invalid YAML / dicts into a clear
CatalogError, plus the accumulate/reset semantics of update_catalog() and
reset_catalog(). Hitting them directly is far faster than exercising them
through the full extract() pipeline.
"""

from __future__ import annotations

import pytest

from envoi import list_datasets, reset_catalog, update_catalog
from envoi.config import (
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

    def test_names_returns_sorted_list_of_strings(self, capsys):
        # Default verbosity is "names" — should return a sorted list of
        # dataset keys and print one key per line.
        result = list_datasets()
        captured = capsys.readouterr()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)
        # Sorted is part of the contract so output is deterministic.
        assert result == sorted(result)
        # Every returned name should also appear in the printed output.
        printed_lines = captured.out.strip().splitlines()
        assert set(result) == set(printed_lines)

    def test_names_includes_a_known_builtin_dataset(self, capsys):
        # Smoke check that the merged view actually exposes built-in
        # datasets (not just an empty list). dem_aster is a stable entry
        # in the bundled catalog used elsewhere in the README/tests.
        result = list_datasets("names")
        capsys.readouterr()  # discard printed output
        assert "dem_aster" in result

    def test_info_returns_records_with_expected_fields(self, capsys):
        # "info" verbosity returns a list of dicts with the human-readable
        # metadata fields. Missing fields are stored as None so callers can
        # filter consistently without KeyError surprises.
        records = list_datasets("info")
        capsys.readouterr()  # discard printed output
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)
        expected_fields = {
            "name",
            "data_source",
            "data_type",
            "description",
            "citation",
            "ee_source_url",
            "source_url",
        }
        # Every record must expose the full info-level field set, even if
        # some values are None — the shape is part of the contract.
        for record in records:
            assert expected_fields <= set(record.keys())

    def test_full_returns_complete_entries(self, capsys):
        # "full" verbosity returns the entire catalog entry plus a "name"
        # field. For at least one known dataset we expect to see the
        # top-level keys present in the YAML (path, data_source, etc.).
        records = list_datasets("full")
        capsys.readouterr()  # discard printed output
        by_name = {r["name"]: r for r in records}
        dem_aster = by_name["dem_aster"]
        assert dem_aster["data_source"] == "earth_engine"
        assert "path" in dem_aster

    def test_includes_user_registered_datasets(self, capsys):
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
        capsys.readouterr()  # discard printed output
        assert "my_test_dataset" in names

    def test_invalid_verbosity_raises_value_error(self):
        # A typo in verbosity should fail loudly, not silently default —
        # the error message must surface the supported levels so the
        # caller can fix the typo without reading the source.
        with pytest.raises(ValueError, match="verbosity must be one of"):
            list_datasets("verbose")
