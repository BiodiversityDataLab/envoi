from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from ..progress import ProgressStepCallback


class BaseAdapter:
    """The adapter interface: the calls that ``extract()`` makes, and the capability declarations.

    Every adapter subclasses this class. It overrides ``build_dataset_meta``,
    ``fetch_stats_batch`` when it declares the ``"tabular"`` output type, and
    ``export_tiles`` when it declares ``"raster"``, with the same parameter
    names, kinds, and defaults (``tests/test_adapters.py`` checks this).
    It sets the capability declarations below as class attributes, and
    ``envoi.adapters.register`` checks them.
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
    # generic keys that every catalog entry can have.
    catalog_keys: ClassVar[frozenset[str]] = frozenset()

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
