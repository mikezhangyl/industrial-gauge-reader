"""Classical full-image fallback and relative red/green segment interpretation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

RECTIFIED_SIZE = 640
DIAL_RADIUS = RECTIFIED_SIZE * 0.47
POLAR_ANGLE_SAMPLES = 1440


@dataclass(frozen=True)
class CircleDetection:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class AngularRun:
    start: float
    end: float
    confidence: float

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def width(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class ColorScaleResult:
    reading: float
    confidence: float
    zero_angle: float
    positive_direction: int
    units_per_segment: float
    red_segments: int
    green_segments: int


def detect_circular_gauge(image: np.ndarray) -> CircleDetection:
    """Locate a frontal circular dial when the pretrained detector returns nothing."""
    height, width = image.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    resized_height, resized_width = resized.shape[:2]
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=resized_height * 0.25,
        param1=100,
        param2=40,
        minRadius=int(resized_height * 0.10),
        maxRadius=int(resized_height * 0.48),
    )
    if circles is None:
        raise ValueError("Gauge detector and circular fallback both returned no dial")

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    y_grid, x_grid = np.ogrid[:resized_height, :resized_width]
    candidates: list[tuple[float, float, float, float]] = []
    for center_x, center_y, radius in circles[0]:
        if (
            center_x - radius < 0
            or center_y - radius < 0
            or center_x + radius >= resized_width
            or center_y + radius >= resized_height
        ):
            continue
        interior = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 < (
            radius * 0.75
        ) ** 2
        saturation = hsv[:, :, 1][interior]
        value = hsv[:, :, 2][interior]
        bright_neutral = float(np.mean((saturation < 60) & (value > 120)))
        dark_fraction = float(np.mean(value < 60))
        if bright_neutral < 0.55 or not 0.02 <= dark_fraction <= 0.35:
            continue
        candidates.append((bright_neutral, center_x, center_y, radius))
    if not candidates:
        raise ValueError("Circular candidates do not look like a gauge face")

    confidence, center_x, center_y, radius = max(candidates)
    center_x /= scale
    center_y /= scale
    radius /= scale
    crop_radius = radius * 1.5
    bbox = (
        max(0, round(center_x - crop_radius)),
        max(0, round(center_y - crop_radius)),
        min(width, round(center_x + crop_radius)),
        min(height, round(center_y + crop_radius)),
    )
    return CircleDetection(bbox, confidence)


def _fill_short_gaps(active: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = active.copy()
    size = len(result)
    inactive = np.flatnonzero(~result)
    if len(inactive) == 0:
        return result
    for start in inactive:
        if result[(start - 1) % size]:
            length = 0
            while length <= maximum_gap and not result[(start + length) % size]:
                length += 1
            if length <= maximum_gap and result[(start + length) % size]:
                for offset in range(length):
                    result[(start + offset) % size] = True
    return result


def _circular_runs(active: np.ndarray, scores: np.ndarray) -> list[AngularRun]:
    size = len(active)
    if not np.any(active):
        return []
    starts = np.flatnonzero(active & ~np.roll(active, 1))
    runs: list[AngularRun] = []
    for start in starts:
        end = int(start)
        while active[end % size] and end - start < size:
            end += 1
        length = end - int(start)
        if length < int(POLAR_ANGLE_SAMPLES * 5 / 360):
            continue
        indices = np.arange(int(start), end) % size
        runs.append(
            AngularRun(
                start=float(start) * 360.0 / size,
                end=float(end) * 360.0 / size,
                confidence=float(np.mean(scores[indices])),
            )
        )
    return runs


def _color_runs(rectified: np.ndarray) -> tuple[list[AngularRun], list[AngularRun]]:
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    green = ((hue >= 35) & (hue <= 95) & (saturation >= 70) & (value >= 45)).astype(
        np.uint8
    )
    red = (((hue <= 12) | (hue >= 168)) & (saturation >= 80) & (value >= 45)).astype(
        np.uint8
    )

    def runs(
        mask: np.ndarray,
        radial_start: float,
        radial_end: float,
        active_threshold: float,
    ) -> list[AngularRun]:
        polar = cv2.warpPolar(
            mask,
            (300, POLAR_ANGLE_SAMPLES),
            (RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2),
            DIAL_RADIUS,
            cv2.WARP_POLAR_LINEAR,
        )
        annulus = polar[:, int(300 * radial_start) : int(300 * radial_end)]
        scores = np.mean(annulus > 0, axis=1)
        active = _fill_short_gaps(scores > active_threshold, maximum_gap=4)
        return _circular_runs(active, scores)

    candidates: list[
        tuple[int, float, int, list[AngularRun], list[AngularRun]]
    ] = []
    for index, (radial_start, radial_end, active_threshold) in enumerate(
        ((0.72, 0.98, 0.30), (0.50, 0.82, 0.22))
    ):
        red_runs = runs(red, radial_start, radial_end, active_threshold)
        green_runs = runs(green, radial_start, radial_end, active_threshold)
        if not red_runs or not green_runs:
            continue
        try:
            selected_red, green_chain, _zero, _direction = _positive_chain(
                red_runs, green_runs
            )
        except ValueError:
            continue
        mean_confidence = float(
            np.mean([selected_red.confidence, *[run.confidence for run in green_chain]])
        )
        candidates.append(
            (len(green_chain), mean_confidence, -index, red_runs, green_runs)
        )
    if not candidates:
        return [], []
    _count, _confidence, _preference, red_runs, green_runs = max(
        candidates, key=lambda item: item[:3]
    )
    return red_runs, green_runs


def _positive_chain(
    red_runs: list[AngularRun], green_runs: list[AngularRun]
) -> tuple[AngularRun, list[AngularRun], float, int]:
    choices: list[tuple[float, AngularRun, AngularRun, int, float]] = []
    for red in red_runs:
        for green in green_runs:
            increasing_gap = (green.start - red.end) % 360.0
            decreasing_gap = (red.start - green.end) % 360.0
            choices.append(
                (
                    increasing_gap,
                    red,
                    green,
                    1,
                    (red.end + increasing_gap / 2.0) % 360.0,
                )
            )
            choices.append(
                (
                    decreasing_gap,
                    red,
                    green,
                    -1,
                    (red.start - decreasing_gap / 2.0) % 360.0,
                )
            )
    gap, red, first_green, direction, zero = min(choices, key=lambda item: item[0])
    if gap > 15.0:
        raise ValueError("Red and green scale segments are not adjacent")

    def directed_distance(angle: float) -> float:
        return (direction * (angle - zero)) % 360.0

    ordered = sorted(green_runs, key=lambda run: directed_distance(run.center))
    chain: list[AngularRun] = []
    previous_center: float | None = None
    for run in ordered:
        center = directed_distance(run.center)
        if center > 180.0:
            continue
        if previous_center is not None and center - previous_center > 60.0:
            break
        chain.append(run)
        previous_center = center
    if not chain or first_green not in chain:
        raise ValueError("Could not build a green segment chain from the zero boundary")
    return red, chain, zero, direction


def read_color_segments(
    rectified: np.ndarray,
    pointer_direction: np.ndarray,
    units_per_segment: float = 1.0,
) -> ColorScaleResult:
    """Read a relative scale where red is negative and green is positive."""
    if units_per_segment <= 0:
        raise ValueError("units_per_segment must be positive")
    red_runs, green_runs = _color_runs(rectified)
    if not red_runs or not green_runs:
        raise ValueError("No adjacent red/green scale segments were found")
    red, green_chain, zero, direction = _positive_chain(red_runs, green_runs)

    def directed_distance(angle: float) -> float:
        return (direction * (angle - zero)) % 360.0

    centers = [-((direction * (zero - red.center)) % 360.0)]
    centers.extend(directed_distance(run.center) for run in green_chain)
    center_steps = np.diff(np.asarray(centers, dtype=np.float64))
    if len(center_steps) == 0 or np.any(center_steps <= 5.0):
        raise ValueError("Colored segment centers do not define a stable scale")
    segment_angle = float(np.median(center_steps))
    consistency = float(max(0.0, 1.0 - np.std(center_steps) / max(segment_angle, 1.0)))

    pointer_angle = (
        math.degrees(
            math.atan2(float(pointer_direction[1]), float(pointer_direction[0]))
        )
        % 360.0
    )
    pointer_distance = directed_distance(pointer_angle)
    if pointer_distance > 180.0:
        pointer_distance -= 360.0
    minimum = centers[0] - segment_angle / 2.0
    maximum = centers[-1] + segment_angle / 2.0
    if not minimum - 5.0 <= pointer_distance <= maximum + 5.0:
        raise ValueError("Pointer is outside the detected colored scale")

    reading = pointer_distance / segment_angle * units_per_segment
    color_confidence = float(
        np.mean([red.confidence, *[run.confidence for run in green_chain]])
    )
    return ColorScaleResult(
        reading=reading,
        confidence=min(color_confidence, consistency),
        zero_angle=zero,
        positive_direction=direction,
        units_per_segment=units_per_segment,
        red_segments=1,
        green_segments=len(green_chain),
    )
