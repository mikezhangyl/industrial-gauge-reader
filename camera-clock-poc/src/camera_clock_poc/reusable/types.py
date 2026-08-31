"""Small interfaces shared by camera experiment runners and readers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    captured_at: datetime
    image: np.ndarray


@dataclass(frozen=True)
class Observation:
    captured_at: datetime
    detected: bool
    bbox: tuple[int, int, int, int] | None
    pointer_found: bool
    center: tuple[float, float] | None
    pointer_tip: tuple[float, float] | None
    angle_degrees: float | None
    value: float | None
    confidence: float | None
    failure_reason: str | None
    processing_ms: float
    method: str
    tilt_degrees: float | None = None
    perspective_rectified: bool = False
    scale_reference_labels: int = 0
    scale_reference_rotation_degrees: float | None = None


class FrameReader(Protocol):
    def read(self, frame: np.ndarray, captured_at: datetime) -> Observation: ...
