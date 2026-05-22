from importlib.metadata import version, PackageNotFoundError

from .extract import extract
from .reducers import list_reducers
from .catalog import update_catalog, reset_catalog, list_datasets, CatalogError
from .auth import init_gee

try:
    __version__ = version("envoi")
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0+local"

__all__ = [
    "extract",
    "update_catalog",
    "reset_catalog",
    "list_datasets",
    "list_reducers",
    "init_gee",
    "CatalogError",
    "__version__",
]
