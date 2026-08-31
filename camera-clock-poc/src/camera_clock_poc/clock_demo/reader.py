"""Training-free analog-clock locator and moving second-hand reader."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

import cv2
import numpy as np

from camera_clock_poc.reusable.types import Observation

FACE_SIZE = 512


@dataclass(frozen=True)
class DialCandidate:
    center: tuple[float, float]
    radius: float
    bbox: tuple[int, int, int, int]
    source: str
    confidence: float = 0.35
    ellipse_center: tuple[float, float] | None = None
    ellipse_axes: tuple[float, float] | None = None
    ellipse_angle_degrees: float | None = None
    hub_detected: bool = False


@dataclass(frozen=True)
class ClockDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    candidate_class: str = "clock"


class ClockObjectDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[ClockDetection]: ...


@dataclass(frozen=True)
class ClockReferenceCorrection:
    transform: np.ndarray
    rotation_degrees: float
    label_count: int


class ClockOrientationEstimator(Protocol):
    def estimate(
        self,
        image: np.ndarray,
        center: tuple[float, float],
        radius: float,
    ) -> ClockReferenceCorrection | None: ...


@dataclass(frozen=True)
class HandCandidate:
    tip: tuple[float, float]
    angle_degrees: float
    length_ratio: float
    line_score: float
    motion_score: float = 0.0


@dataclass(frozen=True)
class TrackedHand:
    candidate: HandCandidate
    velocity_degrees_per_second: float
    confidence: float
    is_predicted: bool = False


@dataclass
class _AngleTrack:
    history: list[tuple[datetime, float, HandCandidate]]
    missed: int = 0


def _signed_angle_delta(current: float, previous: float) -> float:
    return (current - previous + 180.0) % 360.0 - 180.0


class SecondHandTracker:
    """Select the radial line moving clockwise at approximately six degrees/second."""

    def __init__(
        self,
        min_evidence_seconds: float = 1.0,
        match_tolerance_degrees: float = 10.0,
        hold_seconds: float = 1.25,
        lock_prediction_tolerance_degrees: float = 2.0,
    ) -> None:
        self.min_evidence_seconds = min_evidence_seconds
        self.match_tolerance_degrees = match_tolerance_degrees
        self.hold_seconds = hold_seconds
        self.lock_prediction_tolerance_degrees = lock_prediction_tolerance_degrees
        self._tracks: list[_AngleTrack] = []
        self._locked: TrackedHand | None = None
        self._locked_at: datetime | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._locked = None
        self._locked_at = None

    def _hold_locked(
        self, candidates: list[HandCandidate], captured_at: datetime
    ) -> TrackedHand | None:
        if self._locked is None or self._locked_at is None:
            return None
        elapsed = (captured_at - self._locked_at).total_seconds()
        if elapsed < 0.0 or elapsed > self.hold_seconds:
            self._tracks.clear()
            self._locked = None
            self._locked_at = None
            return None
        predicted_angle = (
            self._locked.candidate.angle_degrees
            + self._locked.velocity_degrees_per_second * elapsed
        ) % 360.0
        nearby = [
            candidate
            for candidate in candidates
            if abs(_signed_angle_delta(candidate.angle_degrees, predicted_angle))
            <= max(4.0, self.lock_prediction_tolerance_degrees * 2.0)
        ]
        if nearby:
            candidate = min(
                nearby,
                key=lambda item: abs(
                    _signed_angle_delta(item.angle_degrees, predicted_angle)
                ),
            )
            held = TrackedHand(
                candidate,
                self._locked.velocity_degrees_per_second,
                self._locked.confidence * 0.88,
                True,
            )
            return held
        previous = self._locked.candidate
        predicted = HandCandidate(
            tip=previous.tip,
            angle_degrees=predicted_angle,
            length_ratio=previous.length_ratio,
            line_score=previous.line_score,
        )
        return TrackedHand(
            predicted,
            self._locked.velocity_degrees_per_second,
            self._locked.confidence * max(0.25, 1.0 - elapsed / self.hold_seconds),
            True,
        )

    def update(
        self, candidates: list[HandCandidate], captured_at: datetime
    ) -> TrackedHand | None:
        unmatched = set(range(len(candidates)))
        for track in self._tracks:
            previous_angle = track.history[-1][2].angle_degrees
            choices = sorted(
                (
                    (
                        abs(
                            _signed_angle_delta(
                                candidates[index].angle_degrees, previous_angle
                            )
                        ),
                        index,
                    )
                    for index in unmatched
                ),
                key=lambda item: item[0],
            )
            if not choices or choices[0][0] > self.match_tolerance_degrees:
                track.missed += 1
                continue
            _, index = choices[0]
            candidate = candidates[index]
            previous_unwrapped = track.history[-1][1]
            unwrapped = previous_unwrapped + _signed_angle_delta(
                candidate.angle_degrees, previous_angle
            )
            track.history.append((captured_at, unwrapped, candidate))
            track.history = [
                item
                for item in track.history
                if (captured_at - item[0]).total_seconds() <= 4.0
            ]
            track.missed = 0
            unmatched.remove(index)

        for index in unmatched:
            candidate = candidates[index]
            self._tracks.append(
                _AngleTrack([(captured_at, candidate.angle_degrees, candidate)])
            )
        self._tracks = [track for track in self._tracks if track.missed <= 2]

        eligible: list[tuple[float, int, TrackedHand]] = []
        for track in self._tracks:
            if track.missed > 0 or len(track.history) < 3:
                continue
            elapsed = (track.history[-1][0] - track.history[0][0]).total_seconds()
            if elapsed < self.min_evidence_seconds:
                continue
            displacement = track.history[-1][1] - track.history[0][1]
            velocity = displacement / elapsed
            velocity_error = abs(velocity - 6.0)
            if velocity <= 1.5 or velocity_error > 3.5:
                continue
            candidate = track.history[-1][2]
            confidence = max(0.0, 1.0 - velocity_error / 3.5)
            eligible.append(
                (
                    velocity_error,
                    -len(track.history),
                    TrackedHand(candidate, velocity, confidence),
                )
            )
        if self._locked is not None and self._locked_at is not None:
            elapsed = (captured_at - self._locked_at).total_seconds()
            predicted_angle = (
                self._locked.candidate.angle_degrees
                + self._locked.velocity_degrees_per_second * max(0.0, elapsed)
            ) % 360.0
            same_identity = [
                item
                for item in eligible
                if abs(
                    _signed_angle_delta(
                        item[2].candidate.angle_degrees, predicted_angle
                    )
                )
                <= self.lock_prediction_tolerance_degrees
            ]
            if not same_identity:
                return self._hold_locked(candidates, captured_at)
            selected = min(same_identity, key=lambda item: item[:2])[2]
        else:
            if not eligible:
                return None
            selected = min(eligible, key=lambda item: item[:2])[2]
        self._locked = selected
        self._locked_at = captured_at
        return selected


@dataclass(frozen=True)
class NormalizedDial:
    image: np.ndarray
    gray: np.ndarray
    center: tuple[float, float]
    radius: float
    bbox: tuple[int, int, int, int]
    normalized_to_global: np.ndarray | None = None
    tilt_degrees: float = 0.0
    rectified: bool = False


def angle_from_points(center: tuple[float, float], tip: tuple[float, float]) -> float:
    delta_x = tip[0] - center[0]
    delta_y = tip[1] - center[1]
    return (math.degrees(math.atan2(delta_x, -delta_y)) + 360.0) % 360.0


def angle_to_seconds(angle_degrees: float) -> float:
    return (angle_degrees % 360.0) / 6.0


def circular_seconds_error(actual: float, expected: float) -> float:
    delta = abs((actual - expected) % 60.0)
    return min(delta, 60.0 - delta)


def locate_dials(image: np.ndarray, max_candidates: int = 12) -> list[DialCandidate]:
    height, width = image.shape[:2]
    max_dimension = max(height, width)
    scale = min(1.0, 960.0 / max_dimension)
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.8)
    short_side = min(gray.shape)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=max(30, round(short_side * 0.16)),
        param1=110,
        param2=30,
        minRadius=max(18, round(short_side * 0.04)),
        maxRadius=max(30, round(short_side * 0.47)),
    )
    if circles is None:
        return []
    candidates: list[DialCandidate] = []
    for raw_x, raw_y, raw_radius in circles[0][:max_candidates]:
        center_x = float(raw_x / scale)
        center_y = float(raw_y / scale)
        radius = float(raw_radius / scale)
        margin = radius * 1.10
        x1 = round(center_x - margin)
        y1 = round(center_y - margin)
        x2 = round(center_x + margin)
        y2 = round(center_y + margin)
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            continue
        candidates.append(
            DialCandidate(
                (center_x, center_y), radius, (x1, y1, x2, y2), "hough-circle"
            )
        )
    return candidates


def _fit_dial_ellipse(
    crop: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Find the outer dial ellipse produced by an oblique camera view."""

    crop_height, crop_width = crop.shape[:2]
    short_side = min(crop_height, crop_width)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    expected_center = np.asarray((crop_width * 0.5, crop_height * 0.5))
    candidates: list[
        tuple[
            float,
            tuple[tuple[float, float], tuple[float, float], float],
        ]
    ] = []
    for contour in contours:
        if len(contour) < 30:
            continue
        center, axes, angle = cv2.fitEllipse(contour)
        axis_width, axis_height = (float(axes[0]), float(axes[1]))
        minor_axis, major_axis = sorted((axis_width, axis_height))
        center_error = float(
            np.linalg.norm(np.asarray(center) - expected_center) / short_side
        )
        area_ratio = (
            math.pi * axis_width * axis_height / 4.0 / (crop_width * crop_height)
        )
        aspect_ratio = minor_axis / max(major_axis, 1e-6)
        ellipse_perimeter = math.pi * (
            3.0 * (axis_width + axis_height)
            - math.sqrt(
                (3.0 * axis_width + axis_height) * (axis_width + 3.0 * axis_height)
            )
        )
        contour_coverage = cv2.arcLength(contour, False) / max(ellipse_perimeter, 1e-6)
        if center_error > 0.22:
            continue
        if minor_axis < short_side * 0.55:
            continue
        if major_axis < short_side * 0.72:
            continue
        if major_axis > max(crop_height, crop_width) * 1.10:
            continue
        if not 0.30 <= area_ratio <= 1.05 or aspect_ratio < 0.35:
            continue
        # The dial face normally occupies about two thirds of the detector box.
        # Favor that complete inner rim over larger housing or blur boundaries.
        score = (
            -abs(area_ratio - 0.64) * 6.0
            - center_error * 3.0
            + min(contour_coverage, 1.0) * 1.5
        )
        center_tuple = (float(center[0]), float(center[1]))
        axes_tuple = (float(axes[0]), float(axes[1]))
        candidates.append((score, (center_tuple, axes_tuple, float(angle))))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _find_pointer_hub(
    crop: np.ndarray,
    ellipse_center: tuple[float, float],
    ellipse_axes: tuple[float, float],
) -> tuple[float, float] | None:
    """Find the small pointer axle near the middle of the projected dial."""

    short_side = min(crop.shape[:2])
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(8, round(short_side * 0.04)),
        param1=100,
        param2=12,
        minRadius=max(3, round(short_side * 0.012)),
        maxRadius=max(7, round(short_side * 0.055)),
    )
    if circles is None:
        return None
    center = np.asarray(ellipse_center, dtype=np.float64)
    # The axle must stay close to the projected dial center. A wider search
    # easily mistakes the ends of thick hour/minute hands for a circular hub.
    max_distance = max(ellipse_axes) * 0.11
    nearby = [
        circle
        for circle in circles[0]
        if float(np.linalg.norm(circle[:2] - center)) <= max_distance
    ]
    if not nearby:
        return None
    selected = min(
        nearby,
        key=lambda circle: (
            float(np.linalg.norm(circle[:2] - center)),
            abs(float(circle[2]) - short_side * 0.028),
        ),
    )
    return (float(selected[0]), float(selected[1]))


def _ellipse_bounds(
    center: tuple[float, float],
    axes: tuple[float, float],
    angle_degrees: float,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Return a small padded axis-aligned box around a rotated ellipse."""

    angle = math.radians(angle_degrees)
    half_width = 0.5 * math.sqrt(
        (axes[0] * math.cos(angle)) ** 2 + (axes[1] * math.sin(angle)) ** 2
    )
    half_height = 0.5 * math.sqrt(
        (axes[0] * math.sin(angle)) ** 2 + (axes[1] * math.cos(angle)) ** 2
    )
    margin = 1.08
    image_height, image_width = image_shape[:2]
    return (
        max(0, round(center[0] - half_width * margin)),
        max(0, round(center[1] - half_height * margin)),
        min(image_width, round(center[0] + half_width * margin)),
        min(image_height, round(center[1] + half_height * margin)),
    )


def dial_from_clock_detection(
    image: np.ndarray, detection: ClockDetection
) -> DialCandidate:
    """Refine a pretrained clock box to a circular face without image-specific ROI."""

    image_height, image_width = image.shape[:2]
    raw_x1, raw_y1, raw_x2, raw_y2 = detection.bbox
    x1 = max(0, min(image_width - 1, raw_x1))
    y1 = max(0, min(image_height - 1, raw_y1))
    x2 = max(x1 + 1, min(image_width, raw_x2))
    y2 = max(y1 + 1, min(image_height, raw_y2))
    crop = image[y1:y2, x1:x2]
    crop_height, crop_width = crop.shape[:2]
    short_side = min(crop_height, crop_width)
    expected_center = np.asarray((crop_width * 0.5, crop_height * 0.52))
    expected_radius = short_side * 0.43
    ellipse = _fit_dial_ellipse(crop)
    if ellipse is not None:
        local_center, axes, ellipse_angle = ellipse
        local_hub = _find_pointer_hub(crop, local_center, axes)
        hub_detected = local_hub is not None
        if (
            local_hub is not None
            and math.dist(local_hub, local_center) < max(axes) * 0.035
        ):
            # When both centers coincide, there is no measurable projective
            # shift to correct. Keeping the stable ellipse center avoids hub
            # circle jitter on nearly frontal dials.
            local_hub = local_center
            hub_detected = False
        if local_hub is None:
            local_hub = local_center
        center = (x1 + float(local_hub[0]), y1 + float(local_hub[1]))
        ellipse_center = (
            x1 + float(local_center[0]),
            y1 + float(local_center[1]),
        )
        global_axes = (float(axes[0]), float(axes[1]))
        return DialCandidate(
            center=center,
            radius=max(global_axes) * 0.5,
            bbox=_ellipse_bounds(
                ellipse_center, global_axes, ellipse_angle, image.shape
            ),
            source=f"yolo-{detection.candidate_class}+ellipse",
            confidence=detection.confidence,
            ellipse_center=ellipse_center,
            ellipse_axes=global_axes,
            ellipse_angle_degrees=ellipse_angle,
            hub_detected=hub_detected,
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.6)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, round(short_side * 0.25)),
        param1=110,
        param2=26,
        minRadius=max(14, round(short_side * 0.28)),
        maxRadius=max(20, round(short_side * 0.52)),
    )
    if circles is not None:

        def score(circle: np.ndarray) -> float:
            center_error = float(np.linalg.norm(circle[:2] - expected_center))
            radius_error = abs(float(circle[2]) - expected_radius)
            return center_error + radius_error * 0.45

        local_x, local_y, radius = min(circles[0], key=score)
        center_x = x1 + float(local_x)
        center_y = y1 + float(local_y)
        radius_value = float(radius)
        source = f"yolo-{detection.candidate_class}+hough-circle"
    else:
        center_x = x1 + float(expected_center[0])
        center_y = y1 + float(expected_center[1])
        radius_value = float(expected_radius)
        source = f"yolo-{detection.candidate_class}+box-geometry"

    margin = radius_value * 1.08
    face_x1 = max(0, round(center_x - margin))
    face_y1 = max(0, round(center_y - margin))
    face_x2 = min(image_width, round(center_x + margin))
    face_y2 = min(image_height, round(center_y + margin))
    refined_crop = image[face_y1:face_y2, face_x1:face_x2]
    refined_ellipse = _fit_dial_ellipse(refined_crop)
    if refined_ellipse is not None:
        local_center, axes, ellipse_angle = refined_ellipse
        local_hub = _find_pointer_hub(refined_crop, local_center, axes)
        hub_detected = local_hub is not None
        if (
            local_hub is not None
            and math.dist(local_hub, local_center) < max(axes) * 0.035
        ):
            local_hub = local_center
            hub_detected = False
        if local_hub is None:
            local_hub = local_center
        refined_center = (
            face_x1 + float(local_hub[0]),
            face_y1 + float(local_hub[1]),
        )
        ellipse_center = (
            face_x1 + float(local_center[0]),
            face_y1 + float(local_center[1]),
        )
        global_axes = (float(axes[0]), float(axes[1]))
        return DialCandidate(
            center=refined_center,
            radius=max(global_axes) * 0.5,
            bbox=_ellipse_bounds(
                ellipse_center, global_axes, ellipse_angle, image.shape
            ),
            source=f"{source}+refined-ellipse",
            confidence=detection.confidence,
            ellipse_center=ellipse_center,
            ellipse_axes=global_axes,
            ellipse_angle_degrees=ellipse_angle,
            hub_detected=hub_detected,
        )
    return DialCandidate(
        center=(center_x, center_y),
        radius=radius_value,
        bbox=(face_x1, face_y1, face_x2, face_y2),
        source=source,
        confidence=detection.confidence,
    )


def normalize_dial(image: np.ndarray, dial: DialCandidate) -> NormalizedDial:
    if (
        dial.ellipse_center is not None
        and dial.ellipse_axes is not None
        and dial.ellipse_angle_degrees is not None
    ):
        axis_width, axis_height = dial.ellipse_axes
        angle = math.radians(dial.ellipse_angle_degrees)
        rotation = np.asarray(
            (
                (math.cos(angle), -math.sin(angle)),
                (math.sin(angle), math.cos(angle)),
            ),
            dtype=np.float64,
        )
        semi_axes = np.asarray((axis_width * 0.5, axis_height * 0.5))
        ellipse_center = np.asarray(dial.ellipse_center, dtype=np.float64)
        ellipse_metric = rotation @ np.diag(1.0 / semi_axes**2) @ rotation.T
        conic = np.zeros((3, 3), dtype=np.float64)
        conic[:2, :2] = ellipse_metric
        conic[:2, 2] = -ellipse_metric @ ellipse_center
        conic[2, :2] = conic[:2, 2]
        conic[2, 2] = ellipse_center @ ellipse_metric @ ellipse_center - 1.0
        hub = np.asarray(dial.center, dtype=np.float64)
        projective = np.eye(3, dtype=np.float64)
        if dial.hub_detected:
            vanishing_line = conic @ np.asarray((hub[0], hub[1], 1.0))
            if abs(float(vanishing_line[2])) > 1e-9:
                vanishing_line /= vanishing_line[2]
                projective[2, :] = vanishing_line

        phases = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
        boundary = np.asarray(
            [
                ellipse_center
                + rotation
                @ np.asarray(
                    (
                        semi_axes[0] * math.cos(phase),
                        semi_axes[1] * math.sin(phase),
                    )
                )
                for phase in phases
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        affine_boundary = cv2.perspectiveTransform(
            boundary, projective.astype(np.float32)
        )
        affine_hub = cv2.perspectiveTransform(
            hub.astype(np.float32).reshape(1, 1, 2),
            projective.astype(np.float32),
        ).reshape(2)
        _, affine_axes, affine_angle_degrees = cv2.fitEllipse(affine_boundary)
        affine_angle = math.radians(float(affine_angle_degrees))
        affine_rotation = np.asarray(
            (
                (math.cos(affine_angle), -math.sin(affine_angle)),
                (math.sin(affine_angle), math.cos(affine_angle)),
            ),
            dtype=np.float64,
        )
        target_radius = FACE_SIZE * 0.5 / 1.08
        scaling = np.diag(
            (
                target_radius / max(float(affine_axes[0]) * 0.5, 1e-6),
                target_radius / max(float(affine_axes[1]) * 0.5, 1e-6),
            )
        )
        linear = affine_rotation @ scaling @ affine_rotation.T
        destination_center = np.asarray((FACE_SIZE * 0.5, FACE_SIZE * 0.5))
        translation = destination_center - linear @ affine_hub
        affine_normalization = np.eye(3, dtype=np.float64)
        affine_normalization[:2, :2] = linear
        affine_normalization[:2, 2] = translation
        global_to_normalized = (affine_normalization @ projective).astype(np.float32)
        normalized = cv2.warpPerspective(
            image,
            global_to_normalized,
            (FACE_SIZE, FACE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        normalized_to_global = np.linalg.inv(global_to_normalized)
        aspect_ratio = min(axis_width, axis_height) / max(axis_width, axis_height)
        tilt_degrees = math.degrees(math.acos(min(1.0, aspect_ratio)))
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        return NormalizedDial(
            normalized,
            gray,
            (FACE_SIZE * 0.5, FACE_SIZE * 0.5),
            target_radius,
            dial.bbox,
            normalized_to_global,
            tilt_degrees,
            True,
        )

    x1, y1, x2, y2 = dial.bbox
    crop = image[y1:y2, x1:x2]
    normalized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    scale_x = FACE_SIZE / (x2 - x1)
    scale_y = FACE_SIZE / (y2 - y1)
    center = (
        (dial.center[0] - x1) * scale_x,
        (dial.center[1] - y1) * scale_y,
    )
    radius = dial.radius * min(scale_x, scale_y)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    normalized_to_global = np.asarray(
        (
            (1.0 / scale_x, 0.0, float(x1)),
            (0.0, 1.0 / scale_y, float(y1)),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    return NormalizedDial(
        normalized,
        gray,
        center,
        radius,
        dial.bbox,
        normalized_to_global,
    )


def estimate_tick_rotation(face: NormalizedDial) -> float:
    """Use the twelve longer clock ticks to estimate roll modulo 30 degrees."""

    gray = cv2.GaussianBlur(face.gray, (3, 3), 0.55)
    angles = np.arange(0.0, 360.0, 0.25, dtype=np.float32)
    radii = np.linspace(
        face.radius * 0.84,
        face.radius * 0.95,
        30,
        dtype=np.float32,
    )
    radians = np.deg2rad(angles)[:, None]
    map_x = (face.center[0] + np.sin(radians) * radii[None, :]).astype(np.float32)
    map_y = (face.center[1] - np.cos(radians) * radii[None, :]).astype(np.float32)
    darkness = cv2.remap(
        255 - gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ).mean(axis=1)
    phases = np.arange(0.0, 30.0, 0.25)
    scores: list[float] = []
    for phase in phases:
        indexes = [
            round((phase + index * 30.0) / 0.25) % len(darkness) for index in range(12)
        ]
        scores.append(float(np.mean(darkness[indexes])))
    phase = float(phases[int(np.argmax(scores))])
    return phase if phase <= 15.0 else phase - 30.0


def rotate_normalized_dial(
    face: NormalizedDial, rotation_degrees: float
) -> NormalizedDial:
    """Rotate the rectified face so the 12 o'clock reference points upward."""

    if abs(rotation_degrees) < 0.05:
        return face
    normalized_rotation = cv2.getRotationMatrix2D(
        face.center, rotation_degrees, 1.0
    ).astype(np.float32)
    transform = np.eye(3, dtype=np.float32)
    transform[:2, :] = normalized_rotation
    return transform_normalized_dial(face, transform)


def transform_normalized_dial(
    face: NormalizedDial, transform: np.ndarray
) -> NormalizedDial:
    """Apply a numeric/tick reference transform to a geometrically flat dial."""

    transform = np.asarray(transform, dtype=np.float32)
    image = (
        cv2.warpAffine(
            face.image,
            transform[:2, :],
            (FACE_SIZE, FACE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        if np.allclose(transform[2], (0.0, 0.0, 1.0))
        else cv2.warpPerspective(
            face.image,
            transform,
            (FACE_SIZE, FACE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
    )
    assert face.normalized_to_global is not None
    normalized_to_global = face.normalized_to_global @ np.linalg.inv(transform)
    return NormalizedDial(
        image=image,
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        center=face.center,
        radius=face.radius,
        bbox=face.bbox,
        normalized_to_global=normalized_to_global,
        tilt_degrees=face.tilt_degrees,
        rectified=face.rectified,
    )


def _point_line_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return float("inf")
    offset = point - start
    cross_product = direction[0] * offset[1] - direction[1] * offset[0]
    return abs(float(cross_product)) / length


def detect_hand_candidates(face: NormalizedDial) -> list[HandCandidate]:
    blurred = cv2.GaussianBlur(face.gray, (5, 5), 0.9)
    edges = cv2.Canny(blurred, 45, 135)
    mask = np.zeros_like(edges)
    cv2.circle(
        mask,
        tuple(round(value) for value in face.center),
        round(face.radius * 0.92),
        255,
        -1,
    )
    edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=max(16, round(face.radius * 0.10)),
        minLineLength=max(18, round(face.radius * 0.24)),
        maxLineGap=max(8, round(face.radius * 0.10)),
    )
    if lines is None:
        return []
    center = np.asarray(face.center, dtype=np.float64)
    raw: list[HandCandidate] = []
    for line in lines[:, 0]:
        start = np.asarray(line[:2], dtype=np.float64)
        end = np.asarray(line[2:], dtype=np.float64)
        line_distance = _point_line_distance(center, start, end)
        start_distance = float(np.linalg.norm(start - center))
        end_distance = float(np.linalg.norm(end - center))
        near = min(start_distance, end_distance, line_distance)
        far = max(start_distance, end_distance)
        length = float(np.linalg.norm(end - start))
        if near > face.radius * 0.24:
            continue
        if far < face.radius * 0.48 or far > face.radius * 1.02:
            continue
        if length < face.radius * 0.30:
            continue
        tip_array = start if start_distance > end_distance else end
        tip = (float(tip_array[0]), float(tip_array[1]))
        length_ratio = min(1.0, far / face.radius)
        score = length_ratio + length / face.radius - line_distance / face.radius
        raw.append(
            HandCandidate(
                tip,
                angle_from_points(face.center, tip),
                length_ratio,
                score,
            )
        )

    clustered: list[HandCandidate] = []
    for candidate in sorted(raw, key=lambda item: item.line_score, reverse=True):
        if any(
            min(
                abs(candidate.angle_degrees - kept.angle_degrees),
                360.0 - abs(candidate.angle_degrees - kept.angle_degrees),
            )
            < 4.0
            for kept in clustered
        ):
            continue
        clustered.append(candidate)
    return clustered[:12]


def _align_previous_gray(
    previous: NormalizedDial, current: NormalizedDial
) -> np.ndarray:
    warp = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32)
    mask = np.zeros_like(current.gray)
    center = tuple(round(value) for value in current.center)
    cv2.circle(mask, center, round(current.radius * 0.96), 255, -1)
    cv2.circle(mask, center, round(current.radius * 0.18), 0, -1)
    try:
        cv2.findTransformECC(
            current.gray,
            previous.gray,
            warp,
            cv2.MOTION_EUCLIDEAN,
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                35,
                1e-4,
            ),
            inputMask=mask,
            gaussFiltSize=5,
        )
    except cv2.error:
        return previous.gray
    return cv2.warpAffine(
        previous.gray,
        warp,
        (current.gray.shape[1], current.gray.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )


def detect_motion_hand_candidates(
    previous: NormalizedDial,
    current: NormalizedDial,
    angle_step_degrees: float = 0.5,
) -> list[HandCandidate]:
    """Find the current angle of a moving dark hand after dial registration."""

    aligned_previous = _align_previous_gray(previous, current)
    previous_blurred = cv2.GaussianBlur(aligned_previous, (3, 3), 0.65)
    current_blurred = cv2.GaussianBlur(current.gray, (3, 3), 0.65)
    newly_dark = cv2.subtract(previous_blurred, current_blurred)

    angles = np.arange(0.0, 360.0, angle_step_degrees, dtype=np.float32)
    radii = np.linspace(
        current.radius * 0.28,
        current.radius * 0.92,
        max(48, round(current.radius * 0.70)),
        dtype=np.float32,
    )
    radians = np.deg2rad(angles)[:, None]
    map_x = (current.center[0] + np.sin(radians) * radii[None, :]).astype(np.float32)
    map_y = (current.center[1] - np.cos(radians) * radii[None, :]).astype(np.float32)
    samples = cv2.remap(
        newly_dark,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    energy = samples.astype(np.float64).mean(axis=1)
    median = float(np.median(energy))
    mad = float(np.median(np.abs(energy - median)))
    threshold = median + max(3.0, mad * 5.0)
    peak_indexes = [
        index
        for index, value in enumerate(energy)
        if value >= threshold
        and value >= energy[index - 1]
        and value >= energy[(index + 1) % len(energy)]
    ]

    candidates: list[HandCandidate] = []
    for index in sorted(peak_indexes, key=lambda item: energy[item], reverse=True):
        angle = float(angles[index])
        if any(
            abs(_signed_angle_delta(angle, candidate.angle_degrees)) < 4.0
            for candidate in candidates
        ):
            continue
        angle_radians = math.radians(angle)
        length_ratio = 0.90
        tip = (
            current.center[0] + math.sin(angle_radians) * current.radius * length_ratio,
            current.center[1] - math.cos(angle_radians) * current.radius * length_ratio,
        )
        normalized_energy = min(1.0, float(energy[index]) / 48.0)
        candidates.append(
            HandCandidate(
                tip=tip,
                angle_degrees=angle,
                length_ratio=length_ratio,
                line_score=normalized_energy,
                motion_score=normalized_energy,
            )
        )
        if len(candidates) >= 6:
            break
    return candidates


@dataclass
class _MotionChannelState:
    tracker: SecondHandTracker
    real_hits: int = 0
    confidence_sum: float = 0.0


class AdaptiveMotionHandTracker:
    """Calibrate on motion evidence, then lock the clearest image channel."""

    CHANNELS = ("gray", "local_contrast")

    def __init__(
        self,
        calibration_hits: int = 5,
        calibration_seconds: float = 2.0,
        recalibrate_after_misses: int = 4,
    ) -> None:
        self.calibration_hits = calibration_hits
        self.calibration_seconds = calibration_seconds
        self.recalibrate_after_misses = recalibrate_after_misses
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._states: dict[str, _MotionChannelState] = {}
        self._calibration_started_at: datetime | None = None
        self._selected_channel: str | None = None
        self._selected_misses = 0
        self._last_output: TrackedHand | None = None
        self._last_output_at: datetime | None = None
        self.reset()

    @property
    def selected_channel(self) -> str | None:
        return self._selected_channel

    @property
    def selected_channel_label(self) -> str:
        if self._selected_channel == "local_contrast":
            return "局部对比度"
        if self._selected_channel == "gray":
            return "普通灰度"
        return "校准中"

    def reset(self) -> None:
        self._last_output = None
        self._last_output_at = None
        self._restart_calibration()

    def _restart_calibration(self) -> None:
        self._states = {
            channel: _MotionChannelState(SecondHandTracker())
            for channel in self.CHANNELS
        }
        self._calibration_started_at = None
        self._selected_channel = None
        self._selected_misses = 0

    def _is_temporally_consistent(
        self, tracked: TrackedHand, captured_at: datetime
    ) -> bool:
        if self._last_output is None or self._last_output_at is None:
            return True
        elapsed = (captured_at - self._last_output_at).total_seconds()
        if elapsed < 0.0:
            return False
        if elapsed > 2.5:
            return True
        predicted = (
            self._last_output.candidate.angle_degrees
            + self._last_output.velocity_degrees_per_second * elapsed
        ) % 360.0
        return (
            abs(_signed_angle_delta(tracked.candidate.angle_degrees, predicted)) <= 8.0
        )

    def _remember(
        self, tracked: TrackedHand | None, captured_at: datetime
    ) -> TrackedHand | None:
        if tracked is None or not self._is_temporally_consistent(tracked, captured_at):
            return None
        self._last_output = tracked
        self._last_output_at = captured_at
        return tracked

    def _channel_dial(self, dial: NormalizedDial, channel: str) -> NormalizedDial:
        if channel == "gray":
            return dial
        return replace(dial, gray=self._clahe.apply(dial.gray))

    def _update_channel(
        self,
        channel: str,
        previous: NormalizedDial,
        current: NormalizedDial,
        captured_at: datetime,
    ) -> TrackedHand | None:
        state = self._states[channel]
        candidates = detect_motion_hand_candidates(
            self._channel_dial(previous, channel),
            self._channel_dial(current, channel),
        )
        tracked = state.tracker.update(candidates, captured_at)
        if tracked is not None and not tracked.is_predicted:
            state.real_hits += 1
            state.confidence_sum += tracked.confidence
        return tracked

    def _channel_rank(
        self, channel: str, tracked: TrackedHand | None
    ) -> tuple[int, int, float, int]:
        state = self._states[channel]
        return (
            0 if tracked is None or tracked.is_predicted else 1,
            state.real_hits,
            state.confidence_sum,
            1 if channel == "gray" else 0,
        )

    def _calibrate(
        self,
        previous: NormalizedDial,
        current: NormalizedDial,
        captured_at: datetime,
    ) -> TrackedHand | None:
        if self._calibration_started_at is None:
            self._calibration_started_at = captured_at
        results = {
            channel: self._update_channel(channel, previous, current, captured_at)
            for channel in self.CHANNELS
        }
        best_channel = max(
            self.CHANNELS,
            key=lambda channel: self._channel_rank(channel, results[channel]),
        )
        elapsed = (captured_at - self._calibration_started_at).total_seconds()
        if (
            elapsed >= self.calibration_seconds
            and self._states[best_channel].real_hits >= self.calibration_hits
        ):
            self._selected_channel = best_channel
        ranked_channels = sorted(
            self.CHANNELS,
            key=lambda channel: self._channel_rank(channel, results[channel]),
            reverse=True,
        )
        for channel in ranked_channels:
            accepted = self._remember(results[channel], captured_at)
            if accepted is not None:
                return accepted
        return None

    def update(
        self,
        previous: NormalizedDial,
        current: NormalizedDial,
        captured_at: datetime,
    ) -> TrackedHand | None:
        if self._selected_channel is None:
            return self._calibrate(previous, current, captured_at)

        results = {
            channel: self._update_channel(channel, previous, current, captured_at)
            for channel in self.CHANNELS
        }
        selected = results[self._selected_channel]
        selected = self._remember(selected, captured_at)
        if selected is not None:
            self._selected_misses = 0
            return selected

        backup_channels = [
            channel
            for channel in self.CHANNELS
            if channel != self._selected_channel and results[channel] is not None
        ]
        if backup_channels:
            for backup in sorted(
                backup_channels,
                key=lambda channel: self._channel_rank(channel, results[channel]),
                reverse=True,
            ):
                accepted = self._remember(results[backup], captured_at)
                if accepted is None:
                    continue
                self._selected_channel = backup
                self._selected_misses = 0
                return accepted

        self._selected_misses += 1
        if self._selected_misses < self.recalibrate_after_misses:
            return None

        self._restart_calibration()
        return None


class ClockSecondHandReader:
    def __init__(
        self,
        object_detector: ClockObjectDetector | None = None,
        orientation_estimator: ClockOrientationEstimator | None = None,
    ) -> None:
        self._object_detector = object_detector
        self._orientation_estimator = orientation_estimator
        self._motion_tracker = AdaptiveMotionHandTracker()
        self._previous_dial: DialCandidate | None = None
        self._previous_face: NormalizedDial | None = None
        self._detector_misses = 0
        self._orientation_frames = 0
        self._orientation_correction: ClockReferenceCorrection | None = None
        self._orientation_tick_phase = 0.0
        self._candidate_validation_hold = 0

    @staticmethod
    def _empty(
        captured_at: datetime,
        started_ns: int,
        reason: str,
        detected: bool = False,
        bbox: tuple[int, int, int, int] | None = None,
        center: tuple[float, float] | None = None,
        tilt_degrees: float | None = None,
        perspective_rectified: bool = False,
        scale_reference_labels: int = 0,
        scale_reference_rotation_degrees: float | None = None,
        method: str = "预训练闹钟检测+圆盘校正+秒针速度约束",
    ) -> Observation:
        return Observation(
            captured_at=captured_at,
            detected=detected,
            bbox=bbox,
            pointer_found=False,
            center=center,
            pointer_tip=None,
            angle_degrees=None,
            value=None,
            confidence=None,
            failure_reason=reason,
            processing_ms=(time.perf_counter_ns() - started_ns) / 1e6,
            method=method,
            tilt_degrees=tilt_degrees,
            perspective_rectified=perspective_rectified,
            scale_reference_labels=scale_reference_labels,
            scale_reference_rotation_degrees=scale_reference_rotation_degrees,
        )

    @staticmethod
    def _dial_is_near(previous: DialCandidate, current: DialCandidate) -> bool:
        center_distance = math.dist(previous.center, current.center)
        radius_ratio = current.radius / max(previous.radius, 1e-6)
        return (
            center_distance <= previous.radius * 0.48 and 0.58 <= radius_ratio <= 1.65
        )

    def _tracked_hough_dial(self, frame: np.ndarray) -> DialCandidate | None:
        if self._previous_dial is None or self._detector_misses >= 3:
            return None
        previous_dial = self._previous_dial
        candidates = locate_dials(frame)
        nearby = [
            candidate
            for candidate in candidates
            if self._dial_is_near(previous_dial, candidate)
        ]
        if not nearby:
            return None
        return min(
            nearby,
            key=lambda candidate: math.dist(candidate.center, previous_dial.center),
        )

    def _locate(self, frame: np.ndarray) -> list[DialCandidate]:
        if self._object_detector is None:
            return locate_dials(frame)
        detections = self._object_detector.detect(frame)
        clock_dials = [
            dial_from_clock_detection(frame, detection)
            for detection in detections
            if detection.candidate_class == "clock"
        ]
        if clock_dials:
            self._detector_misses = 0
            self._candidate_validation_hold = 8
            return clock_dials

        relabelled_dials = [
            dial_from_clock_detection(frame, detection)
            for detection in detections
            if detection.candidate_class == "frisbee"
        ]
        for dial in relabelled_dials:
            nearby_confirmed = (
                self._previous_dial is not None
                and self._dial_is_near(self._previous_dial, dial)
                and self._candidate_validation_hold > 0
            )
            if nearby_confirmed:
                self._candidate_validation_hold -= 1
                self._detector_misses = 0
                return [dial]
            if self._orientation_estimator is None:
                continue
            candidate_face = normalize_dial(frame, dial)
            correction = self._orientation_estimator.estimate(
                candidate_face.image,
                candidate_face.center,
                candidate_face.radius,
            )
            if correction is None or correction.label_count < 3:
                continue
            self._orientation_correction = correction
            self._orientation_tick_phase = estimate_tick_rotation(candidate_face)
            self._orientation_frames = 0
            self._candidate_validation_hold = 8
            self._detector_misses = 0
            return [dial]

        self._candidate_validation_hold = max(0, self._candidate_validation_hold - 1)
        self._detector_misses += 1
        tracked = self._tracked_hough_dial(frame)
        return [tracked] if tracked is not None else []

    def _stabilize_dial(self, dial: DialCandidate, frame: np.ndarray) -> DialCandidate:
        previous = self._previous_dial
        if previous is None or not self._dial_is_near(previous, dial):
            return dial
        shift_ratio = math.dist(previous.center, dial.center) / max(
            previous.radius, 1e-6
        )
        alpha = 0.22 if shift_ratio < 0.18 else 0.58
        current_center = dial.center
        if (
            previous.hub_detected
            and not dial.hub_detected
            and previous.ellipse_center is not None
            and dial.ellipse_center is not None
        ):
            current_center = (
                dial.ellipse_center[0]
                + previous.center[0]
                - previous.ellipse_center[0],
                dial.ellipse_center[1]
                + previous.center[1]
                - previous.ellipse_center[1],
            )
        center = (
            previous.center[0] * (1.0 - alpha) + current_center[0] * alpha,
            previous.center[1] * (1.0 - alpha) + current_center[1] * alpha,
        )
        ellipse_center: tuple[float, float] | None = None
        ellipse_axes: tuple[float, float] | None = None
        ellipse_angle: float | None = None
        if (
            previous.ellipse_center is not None
            and previous.ellipse_axes is not None
            and previous.ellipse_angle_degrees is not None
            and dial.ellipse_center is not None
            and dial.ellipse_axes is not None
            and dial.ellipse_angle_degrees is not None
        ):
            ellipse_center = (
                previous.ellipse_center[0] * (1.0 - alpha)
                + dial.ellipse_center[0] * alpha,
                previous.ellipse_center[1] * (1.0 - alpha)
                + dial.ellipse_center[1] * alpha,
            )
            ellipse_axes = (
                previous.ellipse_axes[0] * (1.0 - alpha) + dial.ellipse_axes[0] * alpha,
                previous.ellipse_axes[1] * (1.0 - alpha) + dial.ellipse_axes[1] * alpha,
            )
            angle_delta = (
                dial.ellipse_angle_degrees - previous.ellipse_angle_degrees + 90.0
            ) % 180.0 - 90.0
            ellipse_angle = (
                previous.ellipse_angle_degrees + angle_delta * alpha
            ) % 180.0
            radius = max(ellipse_axes) * 0.5
            bbox = _ellipse_bounds(
                ellipse_center, ellipse_axes, ellipse_angle, frame.shape
            )
        else:
            radius = previous.radius * (1.0 - alpha) + dial.radius * alpha
            margin = radius * 1.08
            height, width = frame.shape[:2]
            bbox = (
                max(0, round(center[0] - margin)),
                max(0, round(center[1] - margin)),
                min(width, round(center[0] + margin)),
                min(height, round(center[1] + margin)),
            )
        return DialCandidate(
            center=center,
            radius=radius,
            bbox=bbox,
            source=f"{dial.source}+temporal-stabilization",
            confidence=dial.confidence,
            ellipse_center=ellipse_center,
            ellipse_axes=ellipse_axes,
            ellipse_angle_degrees=ellipse_angle,
            hub_detected=previous.hub_detected or dial.hub_detected,
        )

    def read(self, frame: np.ndarray, captured_at: datetime) -> Observation:
        started_ns = time.perf_counter_ns()
        dials = self._locate(frame)
        if not dials:
            if self._detector_misses >= 3 or self._object_detector is None:
                self._previous_dial = None
                self._previous_face = None
                self._motion_tracker.reset()
                self._orientation_correction = None
                self._orientation_frames = 0
                self._candidate_validation_hold = 0
            return self._empty(captured_at, started_ns, "预训练模型未找到闹钟")

        raw_dial = max(
            dials,
            key=lambda item: (
                item.confidence * 4.0
                + (
                    1.0
                    if self._previous_dial is not None
                    and self._dial_is_near(self._previous_dial, item)
                    else 0.0
                )
            ),
        )
        same_dial = self._previous_dial is not None and self._dial_is_near(
            self._previous_dial, raw_dial
        )
        if self._previous_dial is not None and not same_dial:
            self._motion_tracker.reset()
            self._previous_face = None
            self._orientation_correction = None
            self._orientation_frames = 0
            self._candidate_validation_hold = 0
        dial = self._stabilize_dial(raw_dial, frame)
        face = normalize_dial(frame, dial)
        tick_phase = estimate_tick_rotation(face)
        self._orientation_frames += 1
        tick_change = (tick_phase - self._orientation_tick_phase + 15.0) % 30.0 - 15.0
        should_refresh_orientation = self._orientation_estimator is not None and (
            (self._orientation_correction is None and self._orientation_frames % 4 == 1)
            or (self._orientation_correction is not None and abs(tick_change) > 4.0)
        )
        if should_refresh_orientation and self._orientation_estimator is not None:
            estimated_correction = self._orientation_estimator.estimate(
                face.image, face.center, face.radius
            )
            if estimated_correction is not None:
                self._orientation_correction = estimated_correction
                self._orientation_tick_phase = tick_phase
        if self._orientation_correction is None:
            face_rotation = tick_phase
            face = rotate_normalized_dial(face, face_rotation)
        else:
            tick_adjustment = cv2.getRotationMatrix2D(
                face.center, tick_change, 1.0
            ).astype(np.float32)
            adjustment = np.eye(3, dtype=np.float32)
            adjustment[:2, :] = tick_adjustment
            face = transform_normalized_dial(
                face,
                adjustment @ self._orientation_correction.transform,
            )
        tracked = (
            self._motion_tracker.update(self._previous_face, face, captured_at)
            if self._previous_face is not None and same_dial
            else None
        )
        self._previous_dial = dial
        self._previous_face = face
        if tracked is None:
            detector_method = (
                "COCO飞盘候选+钟面数字验证"
                if "yolo-frisbee" in dial.source
                else "预训练闹钟检测"
            )
            return self._empty(
                captured_at,
                started_ns,
                "已找到闹钟，正在等待配准帧差中的秒针运动证据",
                detected=True,
                bbox=dial.bbox,
                center=dial.center,
                tilt_degrees=face.tilt_degrees,
                perspective_rectified=face.rectified,
                scale_reference_labels=(
                    self._orientation_correction.label_count
                    if self._orientation_correction is not None
                    else 0
                ),
                scale_reference_rotation_degrees=(
                    self._orientation_correction.rotation_degrees
                    if self._orientation_correction is not None
                    else None
                ),
                method=(
                    f"{detector_method}+自适应图像通道"
                    f"({self._motion_tracker.selected_channel_label})+"
                    "正在等待秒针运动证据"
                ),
            )

        selected = tracked.candidate
        confidence = min(
            0.98,
            0.35 + dial.confidence * 0.25 + tracked.confidence * 0.38,
        )

        angle_radians = math.radians(selected.angle_degrees)
        tip_length = face.radius * selected.length_ratio
        normalized_tip = (
            face.center[0] + math.sin(angle_radians) * tip_length,
            face.center[1] - math.cos(angle_radians) * tip_length,
        )
        assert face.normalized_to_global is not None

        def to_global(point: tuple[float, float]) -> tuple[float, float]:
            homogeneous = face.normalized_to_global @ np.asarray(
                (point[0], point[1], 1.0)
            )
            scale = max(abs(float(homogeneous[2])), 1e-9)
            if homogeneous[2] < 0:
                scale = -scale
            return (
                float(homogeneous[0] / scale),
                float(homogeneous[1] / scale),
            )

        global_center = to_global(face.center)
        global_tip = to_global(normalized_tip)
        detector_method = (
            "COCO飞盘候选+钟面数字验证"
            if "yolo-frisbee" in dial.source
            else "预训练闹钟检测"
        )
        if self._orientation_correction is not None:
            rectification = "完整透视校正+刻度数字定向+"
        elif face.rectified:
            rectification = "完整透视校正+刻度定向+"
        else:
            rectification = ""
        return Observation(
            captured_at=captured_at,
            detected=True,
            bbox=dial.bbox,
            pointer_found=True,
            center=global_center,
            pointer_tip=global_tip,
            angle_degrees=selected.angle_degrees,
            value=angle_to_seconds(selected.angle_degrees),
            confidence=confidence,
            failure_reason=None,
            processing_ms=(time.perf_counter_ns() - started_ns) / 1e6,
            method=(
                f"{detector_method}+{rectification}自适应图像通道"
                f"({self._motion_tracker.selected_channel_label})+"
                "稳定配准帧差+秒针短时预测保持"
                if tracked.is_predicted
                else f"{detector_method}+{rectification}自适应图像通道"
                f"({self._motion_tracker.selected_channel_label})+"
                "稳定配准帧差+约6度每秒的秒针轨迹"
            ),
            tilt_degrees=face.tilt_degrees,
            perspective_rectified=face.rectified,
            scale_reference_labels=(
                self._orientation_correction.label_count
                if self._orientation_correction is not None
                else 0
            ),
            scale_reference_rotation_degrees=(
                self._orientation_correction.rotation_degrees
                if self._orientation_correction is not None
                else None
            ),
        )
