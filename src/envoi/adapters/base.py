from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..progress import ProgressStepCallback


class BaseAdapter:
    """The adapter interface that ``extract()`` uses, and nothing else.

    Every adapter subclasses this class and overrides ``fetch_stats_batch``,
    ``export_tiles``, and ``build_dataset_meta`` with the same parameter names,
    kinds, and defaults (``tests/test_adapters.py`` checks this).
    ``docs/architecture.md`` ("Adapter interface") describes each method.
    """

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
