from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class ProgressEvent:
    """Structured progress update emitted by :func:`envoi.extract`."""

    batch_id: str
    dataset: str
    window_size_m: int
    mode: Literal["tabular", "raster"]
    completed: int
    total: int
    unit: str


ProgressCallback = Callable[[ProgressEvent], None]
ProgressStepCallback = Callable[[int, int], None]


def emit_progress_step(
    callback: ProgressStepCallback | None,
    completed: int,
    total: int,
) -> None:
    """Emit a low-level adapter progress update when a callback is present."""

    if callback is not None:
        callback(int(completed), int(total))
