from __future__ import annotations
from typing import Iterable, Callable, Dict
import numpy as np


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
    # quantiles (rich but still lightweight)
    "q05": make_quantile(0.05),
    "q10": make_quantile(0.10),
    "q25": make_quantile(0.25),
    "q50": make_quantile(0.50),  # alias for median-ish
    "q75": make_quantile(0.75),
    "q90": make_quantile(0.90),
    "q95": make_quantile(0.95),
}


def get_reducer(name: str) -> Callable:
    """
    Look up a reducer by name (case-insensitive).

    Example:
        fn = get_reducer("mean")
        value = fn(window_values)
    """
    fn = _REGISTRY.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown reducer: {name}. Valid: {list(_REGISTRY)}")
    return fn


__all__ = ["get_reducer", "_REGISTRY"]


def list_reducers() -> list[str]:
    """Return sorted names of registered reducers."""
    return sorted(_REGISTRY.keys())
