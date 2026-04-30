from __future__ import annotations
from typing import Iterable, Callable, Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------- helpers ----------


def _to_array(vals: Iterable) -> np.ndarray:
    """Convert any iterable to a 1D float array."""
    return np.asarray(list(vals), dtype=float).ravel()


def _finite(vals: Iterable) -> np.ndarray:
    """Return only finite values (drops NaN / inf)."""
    arr = _to_array(vals)
    return arr[np.isfinite(arr)]


def _nan_if_empty(arr: np.ndarray) -> float | None:
    """
    Return NaN if the array is empty, else None.

    Callers use:
        maybe = _nan_if_empty(arr)
        return maybe if maybe is not None else <real computation>
    """
    return float("nan") if arr.size == 0 else None


# ---------- basic reducers ----------


def r_mean(vals: Iterable) -> float:
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    return maybe if maybe is not None else float(np.mean(arr))


def r_median(vals: Iterable) -> float:
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    return maybe if maybe is not None else float(np.median(arr))


def r_min(vals: Iterable) -> float:
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    return maybe if maybe is not None else float(np.min(arr))


def r_max(vals: Iterable) -> float:
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    return maybe if maybe is not None else float(np.max(arr))


def r_sum(vals: Iterable) -> float:
    """Sum of finite values."""
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    return maybe if maybe is not None else float(np.sum(arr))


def r_std(vals: Iterable) -> float:
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    return maybe if maybe is not None else float(np.std(arr))


def r_var(vals: Iterable) -> float:
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    return maybe if maybe is not None else float(np.var(arr))


def r_count(vals: Iterable) -> int:
    """Number of finite pixels in the window."""
    return int(np.isfinite(_to_array(vals)).sum())


def r_mode(vals: Iterable) -> float:
    """Most frequent value in the window.

    For continuous data with no repeats, returns the smallest value
    (equivalent to min). Ties between equally frequent values are broken
    by returning the smallest. This matches the expected behaviour for
    integer-coded rasters (e.g. land cover classes).
    """
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    if maybe is not None:
        return maybe
    values, counts = np.unique(arr, return_counts=True)
    return float(values[np.argmax(counts)])


# ---------- quantiles ----------


def make_quantile(q: float) -> Callable[[Iterable], float]:
    """
    Factory for quantile reducers.

    q is in [0, 1] (e.g. 0.1 for 10th percentile).
    The function name becomes r_qXX for debugging.
    """

    def _q(vals: Iterable) -> float:
        arr = _finite(vals)
        maybe = _nan_if_empty(arr)
        return maybe if maybe is not None else float(np.percentile(arr, q * 100.0))

    _q.__name__ = f"r_q{int(q * 100)}"
    return _q


# ---------- registry ----------

# Add new reducers here, then they are available in configs
# via their dictionary key, e.g. "mean", "std", "q10", "sum", ...
_REGISTRY: Dict[str, Callable] = {
    # core stats
    "mean": r_mean,
    "median": r_median,
    "min": r_min,
    "max": r_max,
    "sum": r_sum,
    "std": r_std,
    "var": r_var,
    "count": r_count,
    "mode": r_mode,
    # quantiles (rich but still lightweight)
    "q05": make_quantile(0.05),
    "q10": make_quantile(0.10),
    "q25": make_quantile(0.25),
    "q50": make_quantile(0.50),  # alias for median-ish
    "q75": make_quantile(0.75),
    "q90": make_quantile(0.90),
    "q95": make_quantile(0.95),
}


# Reducer names that are NOT backed by a numpy function in the registry.
# These are handled at the adapter level (e.g. "point" samples the exact pixel
# value instead of reducing a window of pixels), so looking them up via
# get_reducer() is always a bug in the caller.
SPECIAL_REDUCERS = frozenset({"point"})


def get_reducer(name: str) -> Callable:
    """
    Look up a reducer by name (case-insensitive).

    Example:
        fn = get_reducer("mean")
        value = fn(window_values)
    """
    lower = name.lower()
    # Guard: "point" is handled by the adapter's fetch_stats_batch, not as a
    # numpy reducer. If it ends up here, the extract pipeline routed incorrectly.
    if lower in SPECIAL_REDUCERS:
        raise ValueError(
            f"'{lower}' is an adapter-level reducer; "
            f"call adapter.fetch_stats_batch instead of get_reducer"
        )
    fn = _REGISTRY.get(lower)
    if fn is None:
        raise ValueError(f"Unknown reducer: {name}. Valid: {list(_REGISTRY)}")
    return fn


CONTINUOUS_ONLY_REDUCERS = {
    "mean",
    "median",
    "std",
    "var",
    "sum",
    "min",
    "max",
    "q05",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
}


def validate_reducers(
    reducer_names: list[str], data_type: str | None, dataset_name: str
) -> str | None:
    """Log a warning if any reducer is inappropriate for the data type.

    Returns the warning message string if a warning was raised, else None.
    The caller can collect these for inclusion in the run metadata.
    """
    if data_type != "categorical":
        return None
    bad = [r for r in reducer_names if r.lower() in CONTINUOUS_ONLY_REDUCERS]
    if bad:
        msg = (
            f"Dataset '{dataset_name}' is categorical but reducers {bad} assume "
            f"continuous data. Consider using 'point', 'mode' or 'count' instead."
        )
        logger.warning(msg)
        return msg
    return None


__all__ = ["get_reducer", "_REGISTRY", "validate_reducers"]


def list_reducers() -> list[str]:
    """Return sorted names of registered reducers."""
    return sorted(_REGISTRY.keys())
