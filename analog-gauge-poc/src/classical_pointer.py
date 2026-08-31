"""Generic colored-hub and continuous-line fallback for tiny pointer gauges."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from src.gauge_reader import angle_from_points


@dataclass(frozen=True)
class ClassicalPointerDetection:
    center: np.ndarray
    tip: np.ndarray
    angle_degrees: float
    confidence: float
    mask: np.ndarray


def _colored_hub(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    active = ((saturation > 85) & (value > 55)).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(active)
    image_center = np.asarray((width / 2, height / 2), dtype=np.float64)
    image_area = height * width
    candidates: list[tuple[float, int]] = []
    for index in range(1, count):
        _, _, component_width, component_height, area = stats[index]
        if not max(12, image_area * 0.0015) <= area <= image_area * 0.12:
            continue
        aspect = component_width / max(1, component_height)
        fill = area / max(1, component_width * component_height)
        if not 0.45 <= aspect <= 2.2 or fill < 0.38:
            continue
        center = centers[index]
        normalized_distance = float(
            np.linalg.norm((center - image_center) / np.asarray((width, height)))
        )
        if normalized_distance > 0.32:
            continue
        radius = math.sqrt(area / math.pi)
        score = normalized_distance - min(radius / max(height, width), 0.15)
        candidates.append((score, index))
    if not candidates:
        raise ValueError("No distinct central colored hub")
    _, selected = min(candidates)
    center = centers[selected].astype(np.float64)
    radius = math.sqrt(float(stats[selected, cv2.CC_STAT_AREA]) / math.pi)
    return center, radius


def _circular_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def detect_classical_pointer(image: np.ndarray) -> ClassicalPointerDetection:
    """Find one thin, continuous radial pointer without a learned segmenter."""
    center, hub_radius = _colored_hub(image)
    height, width = image.shape[:2]
    dial_radius = min(height, width) / 2.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV).detect(gray)[0]
    if lines is None:
        raise ValueError("No continuous straight pointer candidates")

    segments: list[tuple[float, float, float, float, float, np.ndarray]] = []
    for raw_line in lines[:, 0]:
        start = raw_line[:2].astype(np.float64)
        end = raw_line[2:].astype(np.float64)
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length == 0:
            continue
        midpoint = (start + end) / 2.0
        radial = midpoint - center
        radial_length = float(np.linalg.norm(radial))
        if radial_length == 0:
            continue
        alignment = abs(float(np.dot(segment / length, radial / radial_length)))
        start_distance = float(np.linalg.norm(start - center))
        end_distance = float(np.linalg.norm(end - center))
        nearest = min(start_distance, end_distance)
        farthest = max(start_distance, end_distance)
        if (
            alignment < 0.82
            or nearest > dial_radius * 0.60
            or farthest < dial_radius * 0.38
            or length < dial_radius * 0.12
        ):
            continue
        tip = start if start_distance > end_distance else end
        angle = angle_from_points(center, tip)
        segments.append((angle, length, nearest, farthest, alignment, tip))
    if not segments:
        raise ValueError("No center-constrained continuous pointer")

    groups: list[list[tuple[float, float, float, float, float, np.ndarray]]] = []
    for segment in sorted(segments, key=lambda item: item[1], reverse=True):
        for group in groups:
            if _circular_difference(segment[0], group[0][0]) <= 8.0:
                group.append(segment)
                break
        else:
            groups.append([segment])

    scored: list[tuple[float, float, np.ndarray, float]] = []
    for group in groups:
        total_length = float(sum(item[1] for item in group))
        nearest = float(min(item[2] for item in group))
        farthest = float(max(item[3] for item in group))
        alignment = float(np.mean([item[4] for item in group]))
        evidence = (
            total_length
            + farthest
            - nearest
            + alignment * dial_radius
            + (len(group) - 1) * dial_radius * 0.40
        )
        weighted_directions = np.asarray(
            [
                (item[5] - center) / np.linalg.norm(item[5] - center) * item[1]
                for item in group
            ]
        )
        direction = np.sum(weighted_directions, axis=0)
        direction /= np.linalg.norm(direction)
        angle = angle_from_points(np.zeros(2), direction)
        scored.append((evidence, angle, direction, farthest))
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else best[0] - dial_radius
    separation = max(0.0, (best[0] - runner_up) / max(1.0, dial_radius))
    confidence = float(np.clip(0.45 + separation * 0.30, 0.45, 0.90))
    tip = center + best[2] * best[3]

    mask = np.zeros((height, width), dtype=np.uint8)
    center_int = tuple(np.rint(center).astype(int))
    tip_int = tuple(np.rint(tip).astype(int))
    cv2.line(mask, center_int, tip_int, 1, max(2, round(dial_radius * 0.025)))
    cv2.circle(mask, center_int, max(3, round(hub_radius)), 1, -1)
    return ClassicalPointerDetection(
        center=center,
        tip=tip,
        angle_degrees=best[1],
        confidence=confidence,
        mask=mask,
    )
