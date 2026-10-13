# src/envoi/adapters/__init__.py

_REG: dict[str, type] = {}


def register(name, cls):
    _REG[name] = cls


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


# --- import built-in adapters so they self-register on import ---
from . import (
    earth_engine,  # noqa: F401
    local_adapter,  # noqa: F401
)
