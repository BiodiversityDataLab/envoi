# src/envoi/adapters/__init__.py

_REG: dict[str, type] = {}


def register(name, cls):
    _REG[name] = cls


def get_adapter(name):
    # Helpful error if the catalog entry references an unknown data_source.
    # Bare KeyError leaves the user staring at a stack trace; this surfaces
    # the offending name and the registered options in one message.
    try:
        return _REG[name]
    except KeyError:
        raise KeyError(
            f"Unknown data_source {name!r}. Registered adapters: {sorted(_REG)}."
        ) from None


# --- import built-in adapters so they self-register on import ---
from . import local_adapter  # noqa: E402,F401
from . import gee_adapter  # noqa: E402,F401
