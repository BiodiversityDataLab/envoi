from importlib.metadata import version, PackageNotFoundError

from .extract import extract
from .reducers import list_reducers
from .config import update_catalog, reset_catalog

try:
    __version__ = version("envoi")
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0+local"

__all__ = ["extract", "update_catalog", "reset_catalog", "list_reducers", "__version__"]
