# src/biodata/adapters/__init__.py

_REG: dict[str, type] = {}


def register(name, cls):
    _REG[name] = cls


def get_adapter(name):
    return _REG[name]


# --- import built-in adapters so they self-register on import ---
from . import local_adapter  # noqa: E402,F401
from . import gee_adapter  # noqa: E402,F401
