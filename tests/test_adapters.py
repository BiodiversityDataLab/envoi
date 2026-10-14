"""Tests of the adapter contract that ``extract()`` relies on.

``BaseAdapter`` declares the members that ``extract()`` calls, the
capability declarations, and the class-method hooks, and every registered
adapter keeps the same signatures. The adapter registry rejects an unknown
``data_source`` with a ``ValueError``. The Earth Engine adapter handles
each reducer that it declares. No test builds an adapter, so none
needs Earth Engine or a raster file. ``tests/test_provider_contract.py``
tests ``register()``.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import pytest

from envoi.adapters import _REG, get_adapter
from envoi.adapters.base import BaseAdapter
from envoi.adapters.earth_engine import GeeRasterAdapter
from envoi.adapters.earth_engine._reducers import _EE_REDUCER_MAP
from envoi.reducers import KNOWN_REDUCERS

# The members that extract() uses: the three adapter calls, and the
# context-manager lifecycle of `with AdapterClass(spec) as adapter:`.
_INTERFACE_METHODS = frozenset({"fetch_stats_batch", "export_tiles", "build_dataset_meta"})
_LIFECYCLE_METHODS = frozenset({"close", "__enter__", "__exit__"})
# The capability declarations that each adapter sets as class attributes.
_CAPABILITY_ATTRIBUTES = frozenset(
    {"output_types", "supported_reducers", "computable_derived_bands", "catalog_keys"}
)
# The class-method hooks: the pre-flight check that extract() calls before it
# builds any adapter, and the catalog hooks that envoi.catalog calls.
_CLASS_METHOD_HOOKS = frozenset(
    {"check_dataset", "validate_catalog_entry", "inspect_catalog_entry"}
)


def _public_methods(cls: type) -> set[str]:
    """Return the names of the public methods that ``cls`` itself declares."""
    return {
        name for name, member in vars(cls).items() if callable(member) and not name.startswith("_")
    }


def _public_class_methods(cls: type) -> set[str]:
    """Return the names of the public class methods that ``cls`` itself declares.

    ``_public_methods`` does not find them: a ``classmethod`` object in the
    class namespace is not callable.
    """
    return {
        name
        for name, member in vars(cls).items()
        if isinstance(member, classmethod) and not name.startswith("_")
    }


def _parameters_without_annotations(method: Any) -> list[tuple[str, Any, Any]]:
    """Return ``(name, kind, default)`` of each parameter, in order.

    Type annotations are left out on purpose: the adapters annotate differently.
    """
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(method).parameters.values()
    ]


def _class_method_mismatches(adapter_class: type, base_class: type) -> list[str]:
    """Return a description of each class method of ``base_class`` that ``adapter_class`` changes.

    Each class method of ``base_class`` must stay a class method in
    ``adapter_class``, inherited or overridden, with the same parameter names,
    kinds, and defaults. An empty list means that every class method matches.
    """
    mismatches = []
    for method_name in sorted(_public_class_methods(base_class)):
        # getattr_static returns the classmethod object itself. A plain
        # getattr returns a bound method, also for an instance method.
        if not isinstance(inspect.getattr_static(adapter_class, method_name), classmethod):
            mismatches.append(f"{method_name} is not a classmethod")
            continue
        adapter_parameters = _parameters_without_annotations(getattr(adapter_class, method_name))
        base_parameters = _parameters_without_annotations(getattr(base_class, method_name))
        if adapter_parameters != base_parameters:
            mismatches.append(f"{method_name} has different parameters")
    return mismatches


class TestBaseAdapterInterface:
    def test_base_adapter_declares_only_the_contract_members(self):
        """``BaseAdapter`` declares the calls, the lifecycle, the capabilities, and the hooks only."""
        assert _public_methods(BaseAdapter) == _INTERFACE_METHODS | {"close"}
        assert _public_class_methods(BaseAdapter) == _CLASS_METHOD_HOOKS
        assert _LIFECYCLE_METHODS <= set(vars(BaseAdapter))
        public_members = {name for name in vars(BaseAdapter) if not name.startswith("_")}
        assert (
            public_members
            == _INTERFACE_METHODS | {"close"} | _CAPABILITY_ATTRIBUTES | _CLASS_METHOD_HOOKS
        )

    def test_default_check_dataset_accepts_any_spec(self):
        """The default pre-flight check of ``BaseAdapter`` checks nothing and returns None."""
        spec = {"data_source": "any", "path": "not/a/file"}
        assert BaseAdapter.check_dataset(spec, context="Output 'x': dataset 'y'") is None
        assert spec == {"data_source": "any", "path": "not/a/file"}

    def test_default_catalog_hooks_accept_and_copy_the_entry(self):
        """The default catalog hooks check nothing and return an equal new dict."""
        entry = {"data_source": "any", "path": "not/a/file", "bands": [1]}

        assert BaseAdapter.validate_catalog_entry("y", entry) is None
        inspected_entry = BaseAdapter.inspect_catalog_entry("y", entry)

        assert inspected_entry == entry
        assert inspected_entry is not entry
        assert entry == {"data_source": "any", "path": "not/a/file", "bands": [1]}

    def test_parity_check_covers_inherited_and_overridden_check_dataset(self):
        """The Earth Engine adapter inherits ``check_dataset`` and the local adapter overrides it.

        ``TestAdapterSignatureParity.test_class_methods_match_base_adapter``
        then checks both cases.
        """
        assert "check_dataset" not in vars(_REG["earth_engine"])
        assert isinstance(inspect.getattr_static(_REG["local"], "check_dataset"), classmethod)
        assert "check_dataset" in vars(_REG["local"])

    def test_capability_declarations_default_to_empty_frozensets(self):
        """Each capability declaration of ``BaseAdapter`` is an empty ``frozenset``."""
        for attribute_name in sorted(_CAPABILITY_ATTRIBUTES):
            declaration = vars(BaseAdapter)[attribute_name]
            assert type(declaration) is frozenset, attribute_name
            assert declaration == frozenset(), attribute_name

    def test_interface_methods_raise_not_implemented(self):
        """The three adapter calls of ``BaseAdapter`` raise ``NotImplementedError``."""
        base_adapter = BaseAdapter()
        with pytest.raises(NotImplementedError):
            base_adapter.fetch_stats_batch([0.0], [0.0], 100, ["mean"])
        with pytest.raises(NotImplementedError):
            base_adapter.export_tiles([0.0], [0.0], 100, "unused_output_dir")
        with pytest.raises(NotImplementedError):
            base_adapter.build_dataset_meta({})

    def test_both_built_in_adapters_are_registered(self):
        """The registry has the two built-in adapters, so the parity test below covers both."""
        assert {"earth_engine", "local"} <= set(_REG)


class TestBuiltInCapabilityDeclarations:
    @pytest.mark.parametrize(
        (
            "data_source",
            "expected_output_types",
            "expected_reducers",
            "expected_derived_bands",
            "expected_catalog_keys",
        ),
        [
            (
                "earth_engine",
                {"tabular", "raster"},
                KNOWN_REDUCERS,
                {"slope", "aspect"},
                {"dataset_spec", "max_workers", "crs"},
            ),
            (
                "local",
                {"tabular", "raster"},
                KNOWN_REDUCERS,
                set(),
                {"crs", "nodata", "resolution_m", "type", "band_count"},
            ),
        ],
    )
    def test_built_in_adapter_declarations(
        self,
        data_source,
        expected_output_types,
        expected_reducers,
        expected_derived_bands,
        expected_catalog_keys,
    ):
        """Each adapter has the expected values of its four capability declarations.

        Both built-in adapters support both output types and all known
        reducers. Each one accepts the catalog keys that it reads, and the
        local adapter also the keys that its inspection adds.
        """
        adapter_class = _REG[data_source]
        assert adapter_class.output_types == expected_output_types
        assert adapter_class.supported_reducers == expected_reducers
        assert adapter_class.computable_derived_bands == expected_derived_bands
        assert adapter_class.catalog_keys == expected_catalog_keys


# The reducers that the Earth Engine adapter handles without an entry in
# _EE_REDUCER_MAP: class_fraction, which it computes from the class_count
# histogram, and point, which samples the pixel at the point.
_EE_REDUCERS_OUTSIDE_THE_MAP = frozenset({"class_fraction", "point"})


def _is_ee_percentile_reducer(reducer_name: str) -> bool:
    """Return whether ``_get_ee_reducer`` reads ``reducer_name`` as a percentile.

    The rule of ``_get_ee_reducer``: ``q`` or ``p``, then digits with a value
    from 1 to 100. The test does not call ``_get_ee_reducer``, because its
    ``ee.Reducer`` calls need an initialized Earth Engine.
    """
    if reducer_name[:1] not in ("q", "p") or not reducer_name[1:].isdigit():
        return False
    return 0 < int(reducer_name[1:]) <= 100


class TestEarthEngineReducerHandling:
    def test_every_supported_reducer_is_handled_by_the_adapter(self):
        """The Earth Engine adapter handles each reducer of its ``supported_reducers``.

        Its declaration is ``KNOWN_REDUCERS``, which grows with the local
        reducer registry. A reducer that only the local registry has would
        pass the pre-flight check and fail in ``fetch_stats_batch``.
        """
        unhandled_reducers = sorted(
            reducer_name
            for reducer_name in GeeRasterAdapter.supported_reducers
            if reducer_name not in _EE_REDUCER_MAP
            and reducer_name not in _EE_REDUCERS_OUTSIDE_THE_MAP
            and not _is_ee_percentile_reducer(reducer_name)
        )
        assert unhandled_reducers == []


# Stand-in classes for the test of the class-method check: a base class with
# one class-method hook, and subclasses that keep or change it.
class _HookBase:
    @classmethod
    def check_spec(cls, spec, *, context=None):
        return None


class _InheritsHook(_HookBase):
    pass


class _OverridesHook(_HookBase):
    @classmethod
    def check_spec(cls, spec, *, context=None):
        return spec


class _HookAsInstanceMethod(_HookBase):
    def check_spec(self, spec, *, context=None):
        return None


class _HookWithOtherParameters(_HookBase):
    @classmethod
    def check_spec(cls, spec, context=None):
        return None


class TestClassMethodCheck:
    """The class-method check of the parity tests finds a changed hook.

    Stand-in classes show that the check works, independent of the class
    methods that ``BaseAdapter`` declares.
    """

    def test_inherited_or_matching_hook_passes(self):
        """An inherited hook, and an override with the same parameters, are no mismatch."""
        assert _class_method_mismatches(_InheritsHook, _HookBase) == []
        assert _class_method_mismatches(_OverridesHook, _HookBase) == []

    def test_hook_that_is_no_class_method_is_found(self):
        """An override as an instance method is a mismatch."""
        assert _class_method_mismatches(_HookAsInstanceMethod, _HookBase) == [
            "check_spec is not a classmethod"
        ]

    def test_hook_with_other_parameters_is_found(self):
        """An override with a different parameter kind is a mismatch."""
        assert _class_method_mismatches(_HookWithOtherParameters, _HookBase) == [
            "check_spec has different parameters"
        ]


@pytest.mark.parametrize("data_source", sorted(_REG))
class TestAdapterSignatureParity:
    def test_adapter_subclasses_base_adapter(self, data_source):
        """Every registered adapter is a ``BaseAdapter``."""
        assert issubclass(_REG[data_source], BaseAdapter)

    def test_adapter_overrides_the_calls_of_its_output_types(self, data_source):
        """An adapter overrides ``build_dataset_meta``, and the call of each output type it declares.

        ``fetch_stats_batch`` for ``"tabular"``, ``export_tiles`` for ``"raster"``.
        """
        adapter_class = _REG[data_source]
        required_methods = {"build_dataset_meta"}
        if "tabular" in adapter_class.output_types:
            required_methods.add("fetch_stats_batch")
        if "raster" in adapter_class.output_types:
            required_methods.add("export_tiles")
        for method_name in sorted(required_methods):
            assert getattr(adapter_class, method_name) is not getattr(
                BaseAdapter, method_name
            ), method_name

    def test_capability_declarations_are_not_dataclass_fields(self, data_source):
        """A capability declaration of a dataclass adapter is a class attribute, not a field."""
        adapter_class = _REG[data_source]
        if dataclasses.is_dataclass(adapter_class):
            field_names = {field.name for field in dataclasses.fields(adapter_class)}
            assert field_names.isdisjoint(_CAPABILITY_ATTRIBUTES)

    def test_public_method_signatures_match_base_adapter(self, data_source):
        """Each public ``BaseAdapter`` method has the same parameter names, kinds, defaults."""
        adapter_class = _REG[data_source]
        for method_name in sorted(_public_methods(BaseAdapter)):
            adapter_method = getattr(adapter_class, method_name)
            base_method = getattr(BaseAdapter, method_name)
            adapter_parameters = _parameters_without_annotations(adapter_method)
            base_parameters = _parameters_without_annotations(base_method)
            assert adapter_parameters == base_parameters, method_name

    def test_class_methods_match_base_adapter(self, data_source):
        """Each class method of ``BaseAdapter`` stays a class method with the same parameters."""
        assert _class_method_mismatches(_REG[data_source], BaseAdapter) == []


class TestAdapterRegistry:
    def test_get_adapter_unknown_name_raises_value_error(self):
        """An unknown ``data_source`` raises ``ValueError`` that lists the registered adapters."""
        with pytest.raises(ValueError, match="Unknown data_source 'gee'") as error_info:
            get_adapter("gee")
        assert "earth_engine" in str(error_info.value)
        assert "local" in str(error_info.value)
