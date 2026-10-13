from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from ..progress import ProgressStepCallback

# The mark that a per-call band list puts in the merged config of its dataset
# (``_normalize_dataset_entry`` in ``_config_parsing.py`` adds it to the band
# overrides). With the mark, an adapter names every statistic key after its
# band (``{band}_{reducer}``, ``{band}_point``), also for one band. Without
# it, the per-call list ``["B4"]`` could give the bare keys of a single-band
# catalog default (``mean``), and the column name would not say which band was
# selected.
_BAND_NAMING_KEY = "band_naming"
_PER_BAND_NAMING = "per_band"


class BaseAdapter:
    """The adapter interface: the calls that ``extract()`` makes, and the capability declarations.

    Every adapter subclasses this class. It overrides ``build_dataset_meta``,
    ``fetch_stats_batch`` when it declares the ``"tabular"`` output type, and
    ``export_tiles`` when it declares ``"raster"``, with the same parameter
    names, kinds, and defaults (``tests/test_adapters.py`` checks this).
    It sets the capability declarations below as class attributes, and
    ``envoi.adapters.register`` checks them. It can override the class
    methods ``check_dataset`` (the pre-flight check of one dataset),
    ``validate_catalog_entry``, and ``inspect_catalog_entry`` (the catalog
    hooks), with the same parameters.
    ``docs/architecture.md`` ("Adapter interface") describes each method.
    """

    # ---- capability declarations --------------------------------------------
    # Each adapter sets these as plain class attributes (frozensets of str),
    # without an annotation: the built-in adapters are dataclasses, and in a
    # dataclass an annotation makes the attribute a field. The empty defaults
    # declare nothing.

    # The output types that the adapter supports: a non-empty subset of
    # {"tabular", "raster"}.
    output_types: ClassVar[frozenset[str]] = frozenset()
    # The reducer names that fetch_stats_batch supports: a subset of
    # envoi.reducers.KNOWN_REDUCERS. Empty only when "tabular" is not in
    # output_types.
    supported_reducers: ClassVar[frozenset[str]] = frozenset()
    # The derived bands that the adapter can compute from a source band, for
    # example "slope". Empty when the adapter computes none.
    computable_derived_bands: ClassVar[frozenset[str]] = frozenset()
    # The top-level catalog keys that the adapter accepts in addition to the
    # generic keys that every catalog entry can have (envoi.catalog lists
    # them), including the keys that inspect_catalog_entry adds. envoi.catalog
    # rejects every other key.
    catalog_keys: ClassVar[frozenset[str]] = frozenset()

    # ---- catalog hooks ------------------------------------------------------
    @classmethod
    def validate_catalog_entry(cls, name: str, entry: dict) -> None:
        """Check the rules of this adapter for one catalog entry.

        ``envoi.catalog`` calls this for each entry of this adapter's
        ``data_source`` when a catalog is loaded or registered, after the
        generic checks: ``data_source`` names this adapter, ``path`` is a
        non-empty text, ``data_type`` is absent, null, or a known value, the
        derived bands are in ``computable_derived_bands``, and every key is a
        generic key or in ``catalog_keys``. The default checks nothing.

        An override must not import an optional SDK, must not send a network
        request, and must not change ``entry``.

        Args:
            name: The name of the dataset (its key under ``datasets``).
            entry: A copy of the catalog entry, with ``path`` as a ``str``.

        Raises:
            CatalogError: If the entry breaks a rule of the adapter. The
                message starts with ``datasets.<name>: ``, names the key, and
                lists the allowed values or keys. ``envoi.catalog`` raises it
                unchanged.
            ValueError: If the entry breaks a rule of the adapter.
                ``envoi.catalog`` raises a ``CatalogError`` with the same
                message, after ``datasets.<name>: ``.
        """
        return

    @classmethod
    def inspect_catalog_entry(cls, name: str, entry: dict) -> dict:
        """Return a copy of a catalog entry with values read from its data source.

        ``envoi.catalog`` calls this for each entry of this adapter's
        ``data_source`` in a catalog file (the built-in catalog,
        ``load_catalog()``, and ``update_catalog()`` with a path), after
        ``validate_catalog_entry``, and stores the returned dict. It does not
        call it for the entries of a catalog dict. The default returns an
        unchanged copy.

        An override returns a new dict and never changes ``entry``. It adds
        only keys that are in ``catalog_keys``, and does not replace a value
        that the entry sets. It must not import an optional SDK and must not
        send a network request. When it cannot read the data source, it
        should log a warning and return the entry without the missing values,
        so that the catalog still loads.

        Args:
            name: The name of the dataset (its key under ``datasets``).
            entry: A copy of the catalog entry that passed
                ``validate_catalog_entry``.

        Returns:
            A new dict: ``entry`` with the values that the adapter read.

        Raises:
            CatalogError, ValueError: As for ``validate_catalog_entry``.
        """
        return dict(entry)

    # ---- pre-flight check ---------------------------------------------------
    @classmethod
    def check_dataset(cls, spec: dict, *, context: str) -> None:
        """Check one dataset of a run config before ``extract()`` builds any adapter.

        ``extract()`` calls this for every dataset of every run config, after
        it parsed the run configs and before it builds the first adapter. An
        adapter overrides it to check what it can check without a data
        request, so that an error stops the call before any output is written
        or any quota is used. The default checks nothing.

        An override must not send a network request and must not change
        ``spec``. It can check that an optional SDK can be imported and that
        the required credentials exist.

        Args:
            spec: The merged catalog entry: the catalog entry plus the run
                config's band overrides.
            context: The start of each error message, for example
                ``"Output 'terrain': dataset 'my_dem'"``.

        Raises:
            ValueError: If the configuration of the dataset is not valid. The
                message starts with ``context`` and says how to fix it.
            FileNotFoundError: If a resource that the dataset needs, for
                example a file, does not exist. An override can raise another
                ``OSError`` for another problem with such a resource.
        """
        return

    # ---- resource lifecycle -------------------------------------------------
    # Adapters may hold external resources (e.g. an open rasterio dataset) that
    # need to be released when processing for a given dataset finishes. The
    # default implementations below are no-ops so adapters that don't need
    # cleanup (e.g. the GEE adapter, which is purely server-side) inherit
    # safe behaviour. Adapters with real resources override `close()`.
    #
    # The context-manager hooks let callers in extract.py write
    # `with AdapterClass(spec) as adapter: ...` uniformly, so cleanup happens
    # even if the adapter call raises.
    def close(self) -> None:
        """Release any external resources held by the adapter. Default: no-op."""
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        # Returning False (or None) propagates any in-flight exception.
        return False

    def fetch_stats_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        reducer_names: Sequence[str],
        *,
        dates: Sequence | None = None,
        progress_desc: str | None = None,
        disable_progress: bool = False,
        progress_callback: ProgressStepCallback | None = None,
    ) -> list[tuple[dict, dict]]:
        """Return one ``(stats_dict, meta_dict)`` pair for each point, in input order."""
        raise NotImplementedError

    def export_tiles(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        output_dir: str | Path,
        *,
        ids: Sequence[str] | None = None,
        dates: Sequence | None = None,
        dataset_name: str = "dataset",
        resample_m: float | None = None,
        filename_suffix: str | None = None,
        progress_desc: str | None = None,
        disable_progress: bool = False,
        progress_callback: ProgressStepCallback | None = None,
    ) -> tuple[list[Path | None], list[dict]]:
        """Export a GeoTIFF tile for each point. Return ``(paths, meta_list)``, in input order."""
        raise NotImplementedError

    def build_dataset_meta(
        self,
        spec: dict,
        meta_list: list | None = None,
        exported_paths: list | None = None,
        quality: dict | None = None,
        lats: Sequence[float] | None = None,
        lons: Sequence[float] | None = None,
    ) -> dict:
        """Build the per-dataset metadata dict for the sidecar JSON.

        Each adapter overrides this to use its own cached internal state
        (CRS, bands, timestamps, ...) and optionally include quality stats,
        date-selection info, and tile-export summary under the dataset's entry.
        """
        raise NotImplementedError
