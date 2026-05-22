# src/envoi/adapters/earth_engine/__init__.py
"""Earth Engine adapter package.

Importing this package runs ``adapter.py``, whose module-level registration
side effect adds ``GeeRasterAdapter`` to the adapter registry under the
``earth_engine`` data source. Public re-exports below mirror the surface the
rest of the codebase used to import from the old single-file
``gee_adapter`` module.
"""

from .adapter import GeeRasterAdapter
from ._image import KNOWN_DERIVED_BANDS

__all__ = ["GeeRasterAdapter", "KNOWN_DERIVED_BANDS"]
