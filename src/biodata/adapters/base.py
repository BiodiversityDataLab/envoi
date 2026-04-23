from __future__ import annotations

from typing import List, Sequence


class BaseAdapter:
    def fetch_values(self, lat: float, lon: float, window_m: int, *, return_meta: bool = False):
        raise NotImplementedError

    def fetch_stats_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        reducer_names: Sequence[str],
        *,
        dates: Sequence | None = None,
    ) -> List[tuple[dict, dict]]:
        raise NotImplementedError

    def fetch_batch(
        self,
        lats: Sequence[float],
        lons: Sequence[float],
        window_m: int,
        *,
        dates: Sequence | None = None,
        return_meta: bool = False,
    ) -> list:
        """Fetch values for multiple points.

        Default implementation: sequential loop over fetch_values.
        Adapters that benefit from parallelism (e.g. GEE) override this.
        """
        results = []
        for lat, lon in zip(lats, lons):
            result = self.fetch_values(lat, lon, window_m, return_meta=return_meta)
            results.append(result)
        return results

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
