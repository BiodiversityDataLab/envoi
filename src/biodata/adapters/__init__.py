# src/biodata/adapters/__init__.py

_REG: dict[str, type] = {}


def register(name, cls):
    _REG[name] = cls


def get_adapter(name):
    return _REG[name]


# --- import built-in adapters so they self-register on import ---
# local_raster imports `register` from this module and calls it.
from . import local_raster  # noqa: E402,F401
