# src/envoi/adapters/__init__.py
"""The adapter registry: one adapter class for each ``data_source`` value.

Each built-in adapter module calls ``register()`` at its end, and the imports
at the end of this file load those modules, so ``import envoi`` registers them.
``register()`` is internal. It is not in ``envoi.__all__``, and new providers
live in the envoi source tree.
"""

from __future__ import annotations

import re

from ..reducers import KNOWN_REDUCERS
from .base import BaseAdapter

_REG: dict[str, type[BaseAdapter]] = {}

# A data_source name: lower-case ASCII letters, digits, and "_", starting with
# a letter. Users type it in catalog entries, and error messages show it.
_ADAPTER_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

# The output types that an adapter can declare in `output_types`.
_OUTPUT_TYPES = frozenset({"tabular", "raster"})

# The capability declarations of BaseAdapter. Each one must be a frozenset of str.
_CAPABILITY_ATTRIBUTES = (
    "output_types",
    "supported_reducers",
    "computable_derived_bands",
    "catalog_keys",
)


def register(name: str, adapter_class: type[BaseAdapter]) -> None:
    """Register ``adapter_class`` as the adapter of the data source ``name``.

    A second call with the class that is registered under ``name`` changes
    nothing. A class with the same module and qualified name as the registered
    class (the class of a reloaded module) replaces it.

    Args:
        name: The ``data_source`` value of the catalog entries that use the adapter.
        adapter_class: A subclass of ``BaseAdapter`` with valid capability declarations.

    Raises:
        ValueError: If ``name`` does not match ``^[a-z][a-z0-9_]*$``,
            ``adapter_class`` is not a subclass of ``BaseAdapter``, a different
            class is registered under ``name``, or a capability declaration is
            not valid (see ``BaseAdapter``). The message names the adapter and
            the rule.
    """
    if not isinstance(name, str) or not _ADAPTER_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"Cannot register adapter {name!r}: the name must match ^[a-z][a-z0-9_]*$ "
            f"(lower-case letters, digits, and '_', starting with a letter)."
        )
    if not isinstance(adapter_class, type) or not issubclass(adapter_class, BaseAdapter):
        raise ValueError(
            f"Cannot register adapter {name!r}: {adapter_class!r} is not a subclass of "
            f"BaseAdapter."
        )
    registered_class = _REG.get(name)
    if registered_class is not None and not _is_same_class_definition(
        registered_class, adapter_class
    ):
        raise ValueError(
            f"Cannot register adapter {name!r} as {adapter_class.__qualname__}: the class "
            f"{registered_class.__module__}.{registered_class.__qualname__} is already "
            f"registered under this name."
        )
    _validate_capability_declarations(name, adapter_class)
    _REG[name] = adapter_class


def _is_same_class_definition(registered_class: type, adapter_class: type) -> bool:
    """Return True when both classes come from the same class statement.

    A reloaded module makes a new class object from the same statement, with
    the same module and qualified name.
    """
    return registered_class is adapter_class or (
        registered_class.__module__ == adapter_class.__module__
        and registered_class.__qualname__ == adapter_class.__qualname__
    )


def _validate_capability_declarations(name: str, adapter_class: type[BaseAdapter]) -> None:
    """Raise ValueError when a capability declaration of ``adapter_class`` is not valid."""
    for attribute_name in _CAPABILITY_ATTRIBUTES:
        declaration = getattr(adapter_class, attribute_name)
        if not isinstance(declaration, frozenset) or not all(
            isinstance(item, str) for item in declaration
        ):
            raise ValueError(
                f"Cannot register adapter {name!r}: {attribute_name} must be a frozenset "
                f"of str, got {declaration!r}."
            )

    output_types = adapter_class.output_types
    if not output_types or not output_types <= _OUTPUT_TYPES:
        raise ValueError(
            f"Cannot register adapter {name!r}: output_types must be a non-empty subset of "
            f"{sorted(_OUTPUT_TYPES)}, got {sorted(output_types)}."
        )

    supported_reducers = adapter_class.supported_reducers
    unknown_reducers = supported_reducers - KNOWN_REDUCERS
    if unknown_reducers:
        raise ValueError(
            f"Cannot register adapter {name!r}: supported_reducers has unknown reducer(s) "
            f"{sorted(unknown_reducers)}. Known reducers: {sorted(KNOWN_REDUCERS)}."
        )
    if "tabular" in output_types and not supported_reducers:
        raise ValueError(
            f"Cannot register adapter {name!r}: supported_reducers must not be empty, "
            f"because output_types has 'tabular'."
        )


def get_adapter(name):
    # Catalog loading and update_catalog() already reject an unknown
    # data_source, so this is a last check. It raises ValueError (invalid
    # config), with the offending name and the registered options.
    try:
        return _REG[name]
    except KeyError:
        raise ValueError(
            f"Unknown data_source {name!r}. Registered adapters: {sorted(_REG)}."
        ) from None


def all_computable_derived_bands() -> frozenset[str]:
    """Return the derived bands that at least one registered adapter can compute.

    This is the union of ``computable_derived_bands`` of the registered
    adapters. Config parsing splits a per-call band list on this set, so a
    derived band of another provider (for example ``slope`` for a local
    raster) is rejected with a clear message instead of being read as a
    source band. The registry is read at call time.
    """
    derived_bands: frozenset[str] = frozenset()
    for adapter_class in _REG.values():
        derived_bands |= adapter_class.computable_derived_bands
    return derived_bands


# --- import built-in adapters so they self-register on import ---
from . import (
    earth_engine,  # noqa: F401
    local_adapter,  # noqa: F401
)
