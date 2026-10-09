"""Tests of the adapter contract that ``extract()`` relies on.

``BaseAdapter`` declares the members that ``extract()`` calls, and every
registered adapter keeps the same signatures. The adapter registry rejects an
unknown ``data_source`` with a ``ValueError``. No test builds an adapter, so
none needs Earth Engine or a raster file.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from envoi.adapters import _REG, get_adapter
from envoi.adapters.base import BaseAdapter

# The members that extract() uses: the three adapter calls, and the
# context-manager lifecycle of `with AdapterClass(spec) as adapter:`.
_INTERFACE_METHODS = frozenset({"fetch_stats_batch", "export_tiles", "build_dataset_meta"})
_LIFECYCLE_METHODS = frozenset({"close", "__enter__", "__exit__"})


def _public_methods(cls: type) -> set[str]:
    """Return the names of the public methods that ``cls`` itself declares."""
    return {
        name for name, member in vars(cls).items() if callable(member) and not name.startswith("_")
    }


def _parameters_without_annotations(method: Any) -> list[tuple[str, Any, Any]]:
    """Return ``(name, kind, default)`` of each parameter, in order.

    Type annotations are left out on purpose: the adapters annotate differently.
    """
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(method).parameters.values()
    ]


class TestBaseAdapterInterface:
    def test_base_adapter_declares_only_the_members_extract_uses(self):
        """``BaseAdapter`` declares the three adapter calls and the lifecycle, nothing else."""
        assert _public_methods(BaseAdapter) == _INTERFACE_METHODS | {"close"}
        assert _LIFECYCLE_METHODS <= set(vars(BaseAdapter))

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


@pytest.mark.parametrize("data_source", sorted(_REG))
class TestAdapterSignatureParity:
    def test_adapter_subclasses_base_adapter(self, data_source):
        """Every registered adapter is a ``BaseAdapter``."""
        assert issubclass(_REG[data_source], BaseAdapter)

    def test_adapter_overrides_each_interface_method(self, data_source):
        """Every registered adapter defines its own version of each adapter call."""
        adapter_class = _REG[data_source]
        for method_name in sorted(_INTERFACE_METHODS):
            assert getattr(adapter_class, method_name) is not getattr(BaseAdapter, method_name)

    def test_public_method_signatures_match_base_adapter(self, data_source):
        """Each public ``BaseAdapter`` method has the same parameter names, kinds, defaults."""
        adapter_class = _REG[data_source]
        for method_name in sorted(_public_methods(BaseAdapter)):
            adapter_method = getattr(adapter_class, method_name)
            base_method = getattr(BaseAdapter, method_name)
            adapter_parameters = _parameters_without_annotations(adapter_method)
            base_parameters = _parameters_without_annotations(base_method)
            assert adapter_parameters == base_parameters, method_name


class TestAdapterRegistry:
    def test_get_adapter_unknown_name_raises_value_error(self):
        """An unknown ``data_source`` raises ``ValueError`` that lists the registered adapters."""
        with pytest.raises(ValueError, match="Unknown data_source 'gee'") as error_info:
            get_adapter("gee")
        assert "earth_engine" in str(error_info.value)
        assert "local" in str(error_info.value)
