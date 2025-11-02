# src/biodata/reducers.py
from __future__ import annotations
from typing import Iterable, Callable, Dict
import numpy as np


def _to_array(vals: Iterable) -> np.ndarray:
    return np.asarray(list(vals), dtype=float).ravel()


def _finite(vals: Iterable) -> np.ndarray:
    arr = _to_array(vals)
    return arr[np.isfinite(arr)]


def _nan_if_empty(arr: np.ndarray) -> float | None:
    return float("nan") if arr.size == 0 else None


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


def r_std(vals: Iterable) -> float:
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    return maybe if maybe is not None else float(np.std(arr))


def r_var(vals: Iterable) -> float:
    arr = _finite(vals)
    maybe = _nan_if_empty(arr)
    return maybe if maybe is not None else float(np.var(arr))


def r_count(vals: Iterable) -> int:
    return int(np.isfinite(_to_array(vals)).sum())


def make_quantile(q: float) -> Callable[[Iterable], float]:
    def _q(vals: Iterable) -> float:
        arr = _finite(vals)
        maybe = _nan_if_empty(arr)
        return maybe if maybe is not None else float(np.percentile(arr, q * 100.0))

    _q.__name__ = f"r_q{int(q*100)}"
    return _q


_REGISTRY: Dict[str, Callable] = {
    "mean": r_mean,
    "median": r_median,
    "min": r_min,
    "max": r_max,
    "std": r_std,
    "var": r_var,
    "count": r_count,
    "q10": make_quantile(0.10),
    "q90": make_quantile(0.90),
}


def get_reducer(name: str) -> Callable:
    fn = _REGISTRY.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown reducer: {name}. Valid: {list(_REGISTRY)}")
    return fn


__all__ = ["get_reducer", "_REGISTRY"]
