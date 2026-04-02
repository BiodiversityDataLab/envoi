from importlib.metadata import version, PackageNotFoundError

from .enrich import enrich
from .reducers import list_reducers

try:
    __version__ = version("biodata-enricher")
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0+local"

__all__ = ["enrich", "list_reducers", "__version__"]
