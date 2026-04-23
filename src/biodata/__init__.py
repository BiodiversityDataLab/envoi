from importlib.metadata import version, PackageNotFoundError

from .extract import extract
from .reducers import list_reducers

try:
    __version__ = version("biodata-enricher")
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0+local"

__all__ = ["extract", "list_reducers", "__version__"]
