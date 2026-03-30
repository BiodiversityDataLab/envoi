from __future__ import annotations

from typing import Sequence


class BaseAdapter:
    def fetch_values(self, lat: float, lon: float, window_m: int, *, return_meta: bool = False):
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
