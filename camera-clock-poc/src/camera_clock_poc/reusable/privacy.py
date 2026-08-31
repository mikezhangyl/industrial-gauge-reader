"""Privacy-safe presentation frames that retain only the detected instrument."""

from __future__ import annotations

import cv2
import numpy as np


def _strong_blur(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    reduced = cv2.resize(
        frame,
        (max(8, width // 28), max(8, height // 28)),
        interpolation=cv2.INTER_AREA,
    )
    pixelated = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(pixelated, (0, 0), sigmaX=5.0, sigmaY=5.0)


def protect_background(
    frame: np.ndarray,
    instrument_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Blur the background only after an instrument box has been established."""

    if instrument_bbox is None:
        return frame.copy()

    blurred = _strong_blur(frame)

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = instrument_bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    margin_x = round(box_width * 0.08)
    margin_y = round(box_height * 0.08)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)
    if x2 <= x1 or y2 <= y1:
        return frame.copy()

    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 255, -1, cv2.LINE_AA)
    feather = max(3.0, min(box_width, box_height) * 0.035)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather)
    inner_axes = (max(1, round(axes[0] * 0.66)), max(1, round(axes[1] * 0.66)))
    cv2.ellipse(mask, center, inner_axes, 0.0, 0.0, 360.0, 255, -1, cv2.LINE_AA)
    weight = mask.astype(np.float32)[:, :, None] / 255.0
    protected = frame.astype(np.float32) * weight + blurred.astype(np.float32) * (
        1.0 - weight
    )
    return np.rint(protected).clip(0, 255).astype(np.uint8)
