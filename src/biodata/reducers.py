from __future__ import annotations
from typing import Iterable, Callable, Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------- helpers ----------


def _to_array(values: Iterable) -> np.ndarray:
    """Convert any iterable to a 1D float array."""
    return np.asarray(values, dtype=float).ravel()


def _finite(values: Iterable) -> np.ndarray:
    """Return only finite values (drops NaN / inf)."""
    finite_values = _to_array(values)
    return finite_values[np.isfinite(finite_values)]


def _nan_if_empty(finite_values: np.ndarray) -> float | None:
    """
    Return NaN if the array is empty, else None.

    Callers use:
        maybe = _nan_if_empty(finite_values)
        return maybe if maybe is not None else <real computation>
    """
    return float("nan") if finite_values.size == 0 else None


# ---------- basic reducers ----------


def r_mean(values: Iterable) -> float:
    """Mean of finite values in the window."""
    finite_values = _finite(values)
    maybe = _nan_if_empty(finite_values)
    return maybe if maybe is not None else float(np.mean(finite_values))


def r_median(values: Iterable) -> float:
    """Median of finite values in the window."""
    finite_values = _finite(values)
    maybe = _nan_if_empty(finite_values)
    return maybe if maybe is not None else float(np.median(finite_values))


def r_min(values: Iterable) -> float:
    """Minimum of finite values in the window."""
    finite_values = _finite(values)
    maybe = _nan_if_empty(finite_values)
    return maybe if maybe is not None else float(np.min(finite_values))


def r_max(values: Iterable) -> float:
    """Maximum of finite values in the window."""
    finite_values = _finite(values)
    maybe = _nan_if_empty(finite_values)
    return maybe if maybe is not None else float(np.max(finite_values))


def r_sum(values: Iterable) -> float:
    """Sum of finite values."""
    finite_values = _finite(values)
    maybe = _nan_if_empty(finite_values)
    return maybe if maybe is not None else float(np.sum(finite_values))


def r_std(values: Iterable) -> float:
    """Sample standard deviation of finite values in the window (ddof=1)."""
    finite_values = _finite(values)
    maybe = _nan_if_empty(finite_values)
    return maybe if maybe is not None else float(np.std(finite_values, ddof=1))


def r_var(values: Iterable) -> float:
    """Sample variance of finite values in the window (ddof=1)."""
    finite_values = _finite(values)
    maybe = _nan_if_empty(finite_values)
    return maybe if maybe is not None else float(np.var(finite_values, ddof=1))


def r_count(values: Iterable) -> int:
    """Number of finite pixels in the window."""
    return int(np.isfinite(_to_array(values)).sum())


def r_mode(values: Iterable) -> float:
    """Most frequent value in the window.

    For continuous data with no repeats, returns the smallest value
    (equivalent to min). Ties between equally frequent values are broken
    by returning the smallest. This matches the expected behaviour for
    integer-coded rasters (e.g. land cover classes).
    """
    finite_values = _finite(values)
    maybe = _nan_if_empty(finite_values)
    if maybe is not None:
        return maybe
    unique_values, counts = np.unique(finite_values, return_counts=True)
    return float(unique_values[np.argmax(counts)])


# ---------- quantiles ----------


def make_quantile(quantile: float) -> Callable[[Iterable], float]:
    """
    Factory for quantile reducers.

    quantile must be in [0, 1] (e.g. 0.1 for 10th percentile).
    The function name becomes r_qXX for debugging.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    def _q(values: Iterable) -> float:
        finite_values = _finite(values)
        maybe = _nan_if_empty(finite_values)
        return maybe if maybe is not None else float(np.percentile(finite_values, quantile * 100.0))

    _q.__name__ = f"r_q{int(quantile * 100)}"
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


# Reducers that assume continuous data and are inappropriate for categorical data.
# Used by validate_reducers() to warn users about incompatible reducer/data_type combinations.
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


# ---------- public API ----------


def get_reducer(name: str) -> Callable:
    """
    Look up a reducer by name (case-insensitive).

    Example:
        reducer_fn = get_reducer("mean")
        value = reducer_fn(window_values)
    """
    lower = name.lower()
    # Guard: "point" is handled by the adapter's fetch_stats_batch, not as a
    # numpy reducer. If it ends up here, the extract pipeline routed incorrectly.
    if lower in SPECIAL_REDUCERS:
        raise ValueError(
            f"'{lower}' is an adapter-level reducer; "
            f"call adapter.fetch_stats_batch instead of get_reducer"
        )
    reducer_fn = _REGISTRY.get(lower)
    if reducer_fn is None:
        raise ValueError(f"Unknown reducer: {name}. Valid: {list(_REGISTRY)}")
    return reducer_fn


def validate_reducers(
    reducer_names: list[str], data_type: str | None, dataset_name: str
) -> str | None:
    """Log a warning if any reducer is inappropriate for the data type.

    Returns the warning message string if a warning was raised, else None.
    The caller can collect these for inclusion in the run metadata.
    """
    if data_type != "categorical":
        return None
    invalid_reducer_names = [
        reducer_name
        for reducer_name in reducer_names
        if reducer_name.lower() in CONTINUOUS_ONLY_REDUCERS
    ]
    if invalid_reducer_names:
        msg = (
            f"Dataset '{dataset_name}' is categorical but reducers {invalid_reducer_names} assume "
            f"continuous data. Consider using 'point', 'mode' or 'count' instead."
        )
        logger.warning(msg)
        return msg
    return None


def list_reducers() -> list[str]:
    """Return sorted names of registered reducers."""
    return sorted(_REGISTRY.keys())


__all__ = ["get_reducer", "validate_reducers", "list_reducers"]
