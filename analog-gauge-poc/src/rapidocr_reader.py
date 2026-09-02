"""ETHZ gauge models plus generic ellipse rectification and PP-OCRv6 OCR."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from rapidocr import RapidOCR
from ultralytics import YOLO

from src.classical_pointer import detect_classical_pointer
from src.color_scale_reader import (
    ColorScaleResult,
    detect_circular_gauge,
    read_color_segments,
)
from src.ethz_vision_reader import (
    Ellipse,
    NumericLabel,
    ellipse_phase,
    ellipse_residual,
)
from src.gauge_reader import GaugeResult, StageTimings, angle_from_points, synchronize
from src.instrument_reading import InstrumentReadingInterpreter

DETECTION_SIZE = 640
SEGMENTATION_SIZE = 448
RECTIFIED_SIZE = 640
DIAL_RADIUS = RECTIFIED_SIZE * 0.47
SECTOR_STEP_DEGREES = 15
NUMBER_PATTERN = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
RAPIDOCR_PARAMS: dict[str, object] = {
    "Global.use_cls": False,
    "Global.log_level": "error",
    "Rec.rec_batch_num": 32,
    "EngineConfig.onnxruntime.use_coreml": False,
    "EngineConfig.onnxruntime.intra_op_num_threads": 4,
    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
}


@dataclass(frozen=True)
class RobustScaleFit:
    slope: float
    intercept: float
    direction: int
    origin: float
    phase_min: float
    phase_max: float
    rmse: float


@dataclass(frozen=True)
class DialCandidate:
    bbox: tuple[int, int, int, int]
    confidence: float


def is_plausible_dial_bbox(
    bbox: tuple[int, int, int, int], image_shape: tuple[int, int]
) -> bool:
    """Reject boxes whose local geometry cannot reasonably describe a dial."""
    image_height, image_width = image_shape
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width < 24 or height < 24:
        return False
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return False
    aspect = width / height
    return 1.0 / 3.0 <= aspect <= 3.0


def is_unclipped_tile_candidate(
    bbox: tuple[int, int, int, int],
    tile: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> bool:
    """Reject detections cut by an artificial tile edge, not a real image edge."""
    image_height, image_width = image_shape
    x1, y1, x2, y2 = bbox
    tile_x1, tile_y1, tile_x2, tile_y2 = tile
    margin = max(3, round(min(tile_x2 - tile_x1, tile_y2 - tile_y1) * 0.01))
    if tile_x1 > 0 and x1 <= tile_x1 + margin:
        return False
    if tile_y1 > 0 and y1 <= tile_y1 + margin:
        return False
    if tile_x2 < image_width and x2 >= tile_x2 - margin:
        return False
    return not (tile_y2 < image_height and y2 >= tile_y2 - margin)


def deduplicate_dial_candidates(
    candidates: list[DialCandidate], iou_threshold: float = 0.35
) -> list[DialCandidate]:
    """Keep the strongest non-overlapping detection for each physical dial."""
    selected: list[DialCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if all(
            _bbox_iou(candidate.bbox, existing.bbox) < iou_threshold
            for existing in selected
        ):
            selected.append(candidate)
    return selected


def _bbox_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _window_starts(length: int, window: int) -> list[int]:
    if window >= length:
        return [0]
    stride = max(1, round(window * 0.60))
    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def adaptive_tile_levels(
    image_shape: tuple[int, int],
) -> list[list[tuple[int, int, int, int]]]:
    """Build overlapping multiscale windows for any image aspect ratio."""
    image_height, image_width = image_shape
    shortest = min(image_height, image_width)
    levels: list[list[tuple[int, int, int, int]]] = []
    for level, scale in enumerate((1.0, 0.60)):
        base = max(160, round(shortest * scale))
        if image_width >= image_height:
            primary = (min(image_width, 2 * base), min(image_height, base))
        else:
            primary = (min(image_width, base), min(image_height, 2 * base))
        shapes = {primary}
        if level == 1:
            shapes.update(
                {
                    (min(image_width, base), min(image_height, base)),
                    (min(image_width, 2 * base), min(image_height, base)),
                    (min(image_width, base), min(image_height, 2 * base)),
                }
            )
        tiles: set[tuple[int, int, int, int]] = set()
        for tile_width, tile_height in shapes:
            for y1 in _window_starts(image_height, tile_height):
                for x1 in _window_starts(image_width, tile_width):
                    tile = (x1, y1, x1 + tile_width, y1 + tile_height)
                    if tile != (0, 0, image_width, image_height):
                        tiles.add(tile)
        if tiles:
            levels.append(
                sorted(tiles, key=lambda item: (item[1], item[0], item[3], item[2]))
            )
    return levels


def pointer_center_and_tip(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the pointer hub from its thickest section and choose the longer end."""
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) < 20:
        raise ValueError("Pointer mask has too few pixels")
    points = np.column_stack((x_coords, y_coords)).astype(np.float64)
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    axis = eigenvectors[:, 1]
    perpendicular = eigenvectors[:, 0]
    longitudinal = centered @ axis
    lateral = centered @ perpendicular
    lower = float(np.percentile(longitudinal, 1))
    upper = float(np.percentile(longitudinal, 99))
    if upper - lower < 5.0:
        raise ValueError("Pointer mask is too short")

    bin_count = int(np.clip((upper - lower) / 5.0, 24, 80))
    edges = np.linspace(lower, upper, bin_count + 1)
    widths = np.zeros(bin_count, dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int32)
    for index in range(bin_count):
        active = (longitudinal >= edges[index]) & (longitudinal < edges[index + 1])
        values = lateral[active]
        counts[index] = len(values)
        if len(values) >= 2:
            widths[index] = float(np.percentile(values, 95) - np.percentile(values, 5))
    widths = np.convolve(widths, np.ones(3) / 3.0, mode="same")
    margin = max(1, round(bin_count * 0.08))
    valid = np.arange(margin, bin_count - margin)
    valid = valid[counts[valid] > 0]
    if len(valid) == 0 or float(np.max(widths[valid])) <= 0:
        raise ValueError("Pointer hub could not be estimated")
    hub_index = int(valid[int(np.argmax(widths[valid]))])
    hub_active = (longitudinal >= edges[max(0, hub_index - 1)]) & (
        longitudinal < edges[min(bin_count, hub_index + 2)]
    )
    hub = np.mean(points[hub_active], axis=0)
    hub_projection = float((hub - mean) @ axis)
    tip_projection = (
        lower if hub_projection - lower >= upper - hub_projection else upper
    )
    tip = mean + axis * tip_projection
    return hub, tip


def detect_dial_ellipse(crop: np.ndarray, pointer_mask: np.ndarray) -> Ellipse:
    """Find the dial rim from edges, using only the pointer mask as a center prior."""
    center_prior, _ = pointer_center_and_tip(pointer_mask)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    height, width = crop.shape[:2]
    candidates: list[tuple[float, Ellipse]] = []
    for contour in contours:
        if len(contour) < 30:
            continue
        raw = cv2.fitEllipse(contour)
        ellipse: Ellipse = (
            (float(raw[0][0]), float(raw[0][1])),
            (float(raw[1][0]), float(raw[1][1])),
            float(raw[2]),
        )
        (center_x, center_y), (axis_a, axis_b), _ = ellipse
        minor, major = sorted((axis_a, axis_b))
        if major < width * 0.25 or minor < height * 0.25:
            continue
        if major > width * 1.30 or minor > height * 1.30 or minor / major < 0.20:
            continue
        points = contour[:, 0, :].astype(np.float64)
        contour_error = float(np.median(ellipse_residual(ellipse, points)))
        center_error = float(
            np.linalg.norm(np.asarray((center_x, center_y)) - center_prior)
            / max(height, width)
        )
        candidates.append((contour_error + center_error, ellipse))
    if not candidates:
        raise ValueError("Could not find a stable dial ellipse")
    score, ellipse = min(candidates, key=lambda candidate: candidate[0])
    if score > 0.15:
        raise ValueError(f"Dial ellipse confidence is too low: score={score:.3f}")
    return ellipse


def ellipse_rectification(ellipse: Ellipse) -> np.ndarray:
    """Return an affine transform that maps the observed ellipse to a circle."""
    (center_x, center_y), (axis_a, axis_b), angle = ellipse
    theta = math.radians(angle)
    rotation = np.asarray(
        ((math.cos(theta), -math.sin(theta)), (math.sin(theta), math.cos(theta))),
        dtype=np.float64,
    )
    scale = (
        rotation
        @ np.diag((2 * DIAL_RADIUS / axis_a, 2 * DIAL_RADIUS / axis_b))
        @ rotation.T
    )
    target_center = np.asarray((RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2))
    translation = target_center - scale @ np.asarray((center_x, center_y))
    return np.column_stack((scale, translation)).astype(np.float32)


def rectify_dial(crop: np.ndarray, ellipse: Ellipse) -> tuple[np.ndarray, np.ndarray]:
    transform = ellipse_rectification(ellipse)
    rectified = cv2.warpAffine(
        crop,
        transform,
        (RECTIFIED_SIZE, RECTIFIED_SIZE),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rectified, transform


def unwrap_scale_ring(rectified: np.ndarray) -> np.ndarray:
    """Unwrap the numeric annulus to a horizontal diagnostic strip."""
    radial_samples = 300
    angular_samples = 1440
    center = (RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2)
    polar = cv2.warpPolar(
        rectified,
        (radial_samples, angular_samples),
        center,
        DIAL_RADIUS,
        cv2.WARP_POLAR_LINEAR,
    )
    ring = polar[:, int(radial_samples * 0.45) : int(radial_samples * 0.98)]
    return cv2.rotate(ring, cv2.ROTATE_90_CLOCKWISE)


def sector_crops(rectified: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """Make overlapping, locally straight crops around the unwrapped scale ring."""
    center = RECTIFIED_SIZE / 2
    crops: list[np.ndarray] = []
    phases: list[float] = []
    for phase in range(0, 360, SECTOR_STEP_DEGREES):
        radians = math.radians(phase)
        crop_center = (
            center + DIAL_RADIUS * 0.55 * math.cos(radians),
            center + DIAL_RADIUS * 0.55 * math.sin(radians),
        )
        sector = cv2.getRectSubPix(rectified, (100, 70), crop_center)
        sector = cv2.resize(sector, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        sector = cv2.addWeighted(
            sector, 1.8, cv2.GaussianBlur(sector, (0, 0), 1.0), -0.8, 0
        )
        crops.append(sector)
        phases.append(float(phase))
    return crops, phases


def recognize_numeric_sectors(
    ocr: RapidOCR, rectified: np.ndarray
) -> list[NumericLabel]:
    crops, phases = sector_crops(rectified)
    recognition = ocr.recognize_txt(crops)
    if recognition.txts is None or recognition.scores is None:
        return []
    center = RECTIFIED_SIZE / 2
    labels: list[NumericLabel] = []
    for phase, text_raw, confidence_raw in zip(
        phases, recognition.txts, recognition.scores, strict=True
    ):
        text = text_raw.strip().replace(" ", "")
        confidence = float(confidence_raw)
        if confidence < 0.5 or not NUMBER_PATTERN.fullmatch(text) or len(text) > 6:
            continue
        radians = math.radians(phase)
        labels.append(
            NumericLabel(
                text=text,
                value=float(text.replace(",", ".")),
                confidence=confidence,
                center=(
                    center + DIAL_RADIUS * 0.55 * math.cos(radians),
                    center + DIAL_RADIUS * 0.55 * math.sin(radians),
                ),
            )
        )
    return labels


def visible_text_from_ocr_result(result: object) -> str:
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if texts is None or scores is None:
        return ""
    return " ".join(
        text.strip()
        for text, score in zip(texts, scores, strict=True)
        if float(score) >= 0.5 and text.strip()
    )


def recognize_full_dial(
    ocr: RapidOCR,
    rectified: np.ndarray,
    recognition: object | None = None,
) -> list[NumericLabel]:
    """Detect numeric text across the dial when the default annulus has too few hits."""
    result = recognition if recognition is not None else ocr(rectified)
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or texts is None or scores is None:
        return []
    dial_center = np.asarray((RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2))
    labels: list[NumericLabel] = []
    for box, text_raw, confidence_raw in zip(boxes, texts, scores, strict=True):
        text = text_raw.strip().replace(" ", "")
        confidence = float(confidence_raw)
        if confidence < 0.5 or not NUMBER_PATTERN.fullmatch(text) or len(text) > 6:
            continue
        center = np.mean(box, axis=0)
        normalized_radius = float(np.linalg.norm(center - dial_center) / DIAL_RADIUS)
        if not 0.42 <= normalized_radius <= 0.88:
            continue
        labels.append(
            NumericLabel(
                text=text,
                value=float(text.replace(",", ".")),
                confidence=confidence,
                center=(float(center[0]), float(center[1])),
            )
        )
    return labels


def robust_scale_fit(
    labels: list[NumericLabel], ellipse: Ellipse
) -> tuple[RobustScaleFit, list[NumericLabel], list[NumericLabel]]:
    """RANSAC-style fit that rejects overlapping crops and spurious OCR numbers."""
    labels = consolidate_duplicate_labels(labels, ellipse)
    if len(labels) < 5:
        raise ValueError(f"OCR found only {len(labels)} numeric scale candidates")
    raw_phases = np.asarray(
        [ellipse_phase(ellipse, np.asarray(label.center)) for label in labels]
    )
    values = np.asarray([label.value for label in labels], dtype=np.float64)
    best: tuple[int, float, float, RobustScaleFit, list[int]] | None = None
    origins = [0.0, *raw_phases.tolist()]
    for direction in (1, -1):
        directed = (direction * raw_phases) % 360.0
        for origin in origins:
            phases = (directed - origin) % 360.0
            for first in range(len(labels)):
                for second in range(first + 1, len(labels)):
                    phase_delta = phases[second] - phases[first]
                    if abs(phase_delta) < 30.0:
                        continue
                    slope = (values[second] - values[first]) / phase_delta
                    if slope <= 0:
                        continue
                    intercept = values[first] - slope * phases[first]
                    predicted = slope * phases + intercept
                    tolerance = max(0.35, abs(values[second] - values[first]) * 0.06)
                    residuals = np.abs(values - predicted)
                    provisional = np.flatnonzero(residuals <= tolerance).tolist()
                    by_value: dict[float, int] = {}
                    for index in provisional:
                        previous = by_value.get(values[index])
                        if previous is None or (
                            residuals[index],
                            -labels[index].confidence,
                        ) < (residuals[previous], -labels[previous].confidence):
                            by_value[values[index]] = index
                    selected = list(by_value.values())
                    if len(selected) < 5:
                        continue
                    selected_phases = phases[selected]
                    selected_values = values[selected]
                    refit_slope, refit_intercept = np.polyfit(
                        selected_phases, selected_values, 1
                    )
                    if refit_slope <= 0:
                        continue
                    refit_residuals = np.abs(
                        values - (refit_slope * phases + refit_intercept)
                    )
                    value_span = float(np.ptp(selected_values))
                    final_tolerance = max(0.35, value_span * 0.06)
                    final_by_value: dict[float, int] = {}
                    for index in np.flatnonzero(refit_residuals <= final_tolerance):
                        previous = final_by_value.get(values[index])
                        if previous is None or (
                            refit_residuals[index],
                            -labels[index].confidence,
                        ) < (refit_residuals[previous], -labels[previous].confidence):
                            final_by_value[values[index]] = int(index)
                    final = list(final_by_value.values())
                    if len(final) < 5:
                        continue
                    final_phases = phases[final]
                    final_values = values[final]
                    final_slope, final_intercept = np.polyfit(
                        final_phases, final_values, 1
                    )
                    final_errors = final_values - (
                        final_slope * final_phases + final_intercept
                    )
                    rmse = float(np.sqrt(np.mean(final_errors**2)))
                    mean_confidence = float(
                        np.mean([labels[index].confidence for index in final])
                    )
                    fit = RobustScaleFit(
                        slope=float(final_slope),
                        intercept=float(final_intercept),
                        direction=direction,
                        origin=float(origin),
                        phase_min=float(np.min(final_phases)),
                        phase_max=float(np.max(final_phases)),
                        rmse=rmse,
                    )
                    candidate = (len(final), mean_confidence, -rmse, fit, final)
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
    if best is None:
        raise ValueError("Numeric OCR candidates do not form a monotonic scale")
    fit = best[3]
    selected_indices = set(best[4])
    inliers = [labels[index] for index in best[4]]
    rejected = [
        label for index, label in enumerate(labels) if index not in selected_indices
    ]
    return fit, inliers, rejected


def consolidate_duplicate_labels(
    labels: list[NumericLabel], ellipse: Ellipse
) -> list[NumericLabel]:
    """Merge adjacent overlapping sector hits before fitting their scale positions."""
    by_value: dict[float, list[tuple[float, NumericLabel]]] = {}
    for label in labels:
        phase = ellipse_phase(ellipse, np.asarray(label.center))
        by_value.setdefault(label.value, []).append((phase, label))

    consolidated: list[NumericLabel] = []
    max_gap = SECTOR_STEP_DEGREES * 1.1
    for value, observations in by_value.items():
        observations.sort(key=lambda item: item[0])
        clusters: list[list[tuple[float, NumericLabel]]] = []
        for observation in observations:
            if not clusters or observation[0] - clusters[-1][-1][0] > max_gap:
                clusters.append([observation])
            else:
                clusters[-1].append(observation)
        if (
            len(clusters) > 1
            and clusters[0][0][0] + 360.0 - clusters[-1][-1][0] <= max_gap
        ):
            wrapped = clusters[-1] + [
                (phase + 360.0, label) for phase, label in clusters[0]
            ]
            clusters = [wrapped, *clusters[1:-1]]
        cluster = max(
            clusters,
            key=lambda items: (
                len(items),
                float(np.mean([item[1].confidence for item in items])),
            ),
        )
        phases = np.asarray([item[0] for item in cluster], dtype=np.float64)
        phase = float(np.mean(phases)) % 360.0
        confidence = float(np.mean([item[1].confidence for item in cluster]))
        text = max(cluster, key=lambda item: item[1].confidence)[1].text
        radians = math.radians(phase)
        center_x, center_y = ellipse[0]
        radius = DIAL_RADIUS * 0.55
        consolidated.append(
            NumericLabel(
                text=text,
                value=value,
                confidence=confidence,
                center=(
                    center_x + radius * math.cos(radians),
                    center_y + radius * math.sin(radians),
                ),
            )
        )
    return consolidated


def pointer_from_rectified_mask(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) < 20:
        raise ValueError("Pointer mask has too few rectified pixels")
    center = np.asarray((RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2))
    points = np.column_stack((x_coords, y_coords)).astype(np.float64)
    centered = points - center
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = centered @ axis
    positive_values = projections[projections > 0]
    negative_values = -projections[projections < 0]
    positive = (
        float(np.percentile(positive_values, 99)) if len(positive_values) else 0.0
    )
    negative = (
        float(np.percentile(negative_values, 99)) if len(negative_values) else 0.0
    )
    if positive <= 0 and negative <= 0:
        raise ValueError("Pointer direction could not be estimated")
    direction = axis if positive >= negative else -axis
    aspect_ratio = float(np.min(eigenvalues) / np.max(eigenvalues))
    if aspect_ratio > 0.02:
        radii = np.linalg.norm(centered, axis=1)
        tip_region = centered[radii >= np.percentile(radii, 95)]
        tip_direction = np.mean(tip_region, axis=0)
        if np.linalg.norm(tip_direction) > 0:
            direction = tip_direction / np.linalg.norm(tip_direction)
    return direction, center + direction * DIAL_RADIUS


def reading_from_fit(direction: np.ndarray, fit: RobustScaleFit) -> float:
    raw_phase = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
    phase = (fit.direction * raw_phase - fit.origin) % 360.0
    equivalents = [phase + 360.0 * turn for turn in (-1, 0, 1)]
    target = (fit.phase_min + fit.phase_max) / 2.0
    selected_phase = min(equivalents, key=lambda candidate: abs(candidate - target))
    return fit.slope * selected_phase + fit.intercept


class EthzPaddleGaugeReader:
    """Resident ETHZ models and PP-OCRv6 ONNX sessions for steady-state tests."""

    def __init__(
        self,
        detector_path: Path,
        segmentation_path: Path,
        device: str,
        units_per_major_segment: float = 1.0,
        reading_interpreter: InstrumentReadingInterpreter | None = None,
    ):
        if units_per_major_segment <= 0:
            raise ValueError("units_per_major_segment must be positive")
        self.device = device
        self.units_per_major_segment = units_per_major_segment
        self.reading_interpreter = reading_interpreter
        self.detector = YOLO(str(detector_path))
        self.segmenter = YOLO(str(segmentation_path))
        self.ocr = RapidOCR(params=RAPIDOCR_PARAMS)
        self.last_rectified: np.ndarray | None = None
        self.last_ring: np.ndarray | None = None
        self._visible_text_cache: dict[tuple[str, int, int], str] = {}

    def recognize_isolated_text_lines(self, crops: list[np.ndarray]) -> object:
        """Reuse the bounded resident OCR session for specialized text crops."""
        return self.ocr.recognize_txt(crops)

    def close(self) -> None:
        """Release ONNX Runtime sessions before interpreter shutdown."""
        for component_name in ("text_det", "text_cls", "text_rec"):
            component = getattr(self.ocr, component_name, None)
            session_holder = getattr(component, "session", None)
            if session_holder is not None and hasattr(session_holder, "session"):
                session_holder.session = None

    def _interpret_result(self, result: GaugeResult, visible_text: str) -> GaugeResult:
        if self.reading_interpreter is None:
            return result
        return self.reading_interpreter.interpret(result, visible_text)

    def _full_image_visible_text(self, image_path: Path, image: np.ndarray) -> str:
        stat = image_path.stat()
        key = (str(image_path.resolve()), stat.st_size, stat.st_mtime_ns)
        cached = self._visible_text_cache.get(key)
        if cached is not None:
            return cached
        visible_text = visible_text_from_ocr_result(self.ocr(image))
        self._visible_text_cache[key] = visible_text
        return visible_text

    def detect_dial_candidates(self, image_path: Path) -> tuple[DialCandidate, ...]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot decode image: {image_path}")
        detection = self.detector.predict(
            image, imgsz=DETECTION_SIZE, conf=0.20, device=self.device, verbose=False
        )[0]
        candidates = self._result_candidates(detection, (0, 0), image.shape[:2])
        return tuple(deduplicate_dial_candidates(candidates))

    @staticmethod
    def _result_candidates(
        result: object,
        offset: tuple[int, int],
        image_shape: tuple[int, int],
        tile: tuple[int, int, int, int] | None = None,
    ) -> list[DialCandidate]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        image_height, image_width = image_shape
        offset_x, offset_y = offset
        candidates: list[DialCandidate] = []
        for raw_box, raw_confidence in zip(
            boxes.xyxy.detach().cpu().numpy(),
            boxes.conf.detach().cpu().numpy(),
            strict=True,
        ):
            x1, y1, x2, y2 = raw_box.round().astype(int).tolist()
            bbox = (
                max(0, min(image_width, x1 + offset_x)),
                max(0, min(image_height, y1 + offset_y)),
                max(0, min(image_width, x2 + offset_x)),
                max(0, min(image_height, y2 + offset_y)),
            )
            if is_plausible_dial_bbox(bbox, image_shape) and (
                tile is None or is_unclipped_tile_candidate(bbox, tile, image_shape)
            ):
                candidates.append(DialCandidate(bbox, float(raw_confidence)))
        return candidates

    def _segment_candidates(
        self, image: np.ndarray, candidates: list[DialCandidate]
    ) -> tuple[DialCandidate, np.ndarray, float, str] | None:
        unique: dict[tuple[int, int, int, int], DialCandidate] = {}
        for candidate in sorted(
            candidates, key=lambda item: item.confidence, reverse=True
        ):
            unique.setdefault(candidate.bbox, candidate)
        selected = list(unique.values())[:16]
        if not selected:
            return None
        crops = [
            image[y1:y2, x1:x2]
            for x1, y1, x2, y2 in (candidate.bbox for candidate in selected)
        ]
        inputs = [
            cv2.resize(crop, (SEGMENTATION_SIZE, SEGMENTATION_SIZE)) for crop in crops
        ]
        segmentations = self.segmenter.predict(
            inputs,
            imgsz=SEGMENTATION_SIZE,
            conf=0.12,
            device=self.device,
            verbose=False,
        )
        best: tuple[float, float, DialCandidate, np.ndarray] | None = None
        for candidate, crop, segmentation in zip(
            selected, crops, segmentations, strict=True
        ):
            if segmentation.masks is None or len(segmentation.masks.data) == 0:
                continue
            pointer_index = int(torch.argmax(segmentation.boxes.conf).item())
            pointer_confidence = float(
                segmentation.boxes.conf[pointer_index].detach().cpu()
            )
            mask = (
                segmentation.masks.data[pointer_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            mask = cv2.resize(
                mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            try:
                pointer_center, _ = pointer_center_and_tip(mask)
            except ValueError:
                continue
            normalized_center = pointer_center / np.asarray(
                (crop.shape[1], crop.shape[0]), dtype=np.float64
            )
            if np.any(normalized_center < 0.10) or np.any(normalized_center > 0.90):
                continue
            score = (pointer_confidence, candidate.confidence, candidate, mask)
            if best is None or score[:2] > best[:2]:
                best = score
        if best is None:
            return None
        return best[2], best[3], best[0], "model-segmentation"

    @staticmethod
    def _classical_candidates(
        image: np.ndarray, candidates: list[DialCandidate]
    ) -> tuple[DialCandidate, np.ndarray, float, str] | None:
        best: tuple[float, float, DialCandidate, np.ndarray] | None = None
        for candidate in sorted(
            candidates, key=lambda item: item.confidence, reverse=True
        )[:8]:
            x1, y1, x2, y2 = candidate.bbox
            crop = image[y1:y2, x1:x2]
            try:
                pointer = detect_classical_pointer(crop)
            except ValueError:
                continue
            score = (pointer.confidence, candidate.confidence, candidate, pointer.mask)
            if best is None or score[:2] > best[:2]:
                best = score
        if best is None:
            return None
        return best[2], best[3], best[0], "colored-hub+line-segment"

    def _tile_candidates(
        self,
        image: np.ndarray,
        tiles: list[tuple[int, int, int, int]],
    ) -> list[DialCandidate]:
        crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
        detections = self.detector.predict(
            crops,
            imgsz=DETECTION_SIZE,
            conf=0.20,
            device=self.device,
            verbose=False,
        )
        candidates: list[DialCandidate] = []
        for tile, detection in zip(tiles, detections, strict=True):
            x1, y1, _, _ = tile
            candidates.extend(
                self._result_candidates(detection, (x1, y1), image.shape[:2], tile=tile)
            )
        return candidates

    def save_artifacts(self, result_path: Path) -> tuple[Path, Path] | None:
        if self.last_rectified is None or self.last_ring is None:
            return None
        suffix = result_path.stem.removeprefix("result")
        rectified_path = result_path.with_name(f"rectified{suffix}.jpg")
        ring_path = result_path.with_name(f"scale-ring{suffix}.jpg")
        rectified_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rectified_path), self.last_rectified)
        cv2.imwrite(str(ring_path), self.last_ring)
        return rectified_path, ring_path

    def read(self, image_path: Path, *, visible_text_context: str = "") -> GaugeResult:
        preprocess_start = time.perf_counter_ns()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot decode image: {image_path}")
        preprocess_end = time.perf_counter_ns()

        synchronize(self.device)
        inference_start = time.perf_counter_ns()
        image_visible_text = self._full_image_visible_text(image_path, image)
        visible_text_context = f"{visible_text_context} {image_visible_text}".strip()
        detection = self.detector.predict(
            image, imgsz=DETECTION_SIZE, conf=0.25, device=self.device, verbose=False
        )[0]
        has_raw_detection = detection.boxes is not None and len(detection.boxes) > 0
        candidates = self._result_candidates(detection, (0, 0), image.shape[:2])
        segmented = self._segment_candidates(image, candidates)
        if segmented is None:
            segmented = self._classical_candidates(image, candidates)
        if segmented is None and not has_raw_detection:
            try:
                circle_detection = detect_circular_gauge(image)
                circle_candidate = DialCandidate(
                    circle_detection.bbox, circle_detection.confidence
                )
                if is_plausible_dial_bbox(circle_candidate.bbox, image.shape[:2]):
                    candidates.append(circle_candidate)
                    segmented = self._segment_candidates(image, [circle_candidate])
                    if segmented is None:
                        segmented = self._classical_candidates(
                            image, [circle_candidate]
                        )
            except ValueError:
                pass
        if segmented is None:
            for tiles in adaptive_tile_levels(image.shape[:2]):
                tiled_candidates = self._tile_candidates(image, tiles)
                candidates.extend(tiled_candidates)
                segmented = self._segment_candidates(image, tiled_candidates)
                if segmented is None:
                    segmented = self._classical_candidates(image, tiled_candidates)
                if segmented is not None:
                    break
        if segmented is None:
            synchronize(self.device)
            inference_end = time.perf_counter_ns()
            candidate = max(candidates, key=lambda item: item.confidence, default=None)
            result = GaugeResult(
                candidate is not None,
                candidate.bbox if candidate else None,
                candidate.confidence if candidate else None,
                False,
                None,
                None,
                None,
                None,
                None,
                None,
                candidate.confidence if candidate else None,
                None,
                StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    0.0,
                ),
                failure_reason=(
                    "Pointer segmenter returned no valid mask"
                    if candidate
                    else "No geometrically valid gauge candidate"
                ),
            )
            return self._interpret_result(result, visible_text_context)
        candidate, mask, pointer_confidence, pointer_method = segmented
        bbox = candidate.bbox
        detection_confidence = candidate.confidence
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        try:
            ellipse = detect_dial_ellipse(crop, mask)
            rectified, transform = rectify_dial(crop, ellipse)
            rectified_mask = cv2.warpAffine(
                mask,
                transform,
                (RECTIFIED_SIZE, RECTIFIED_SIZE),
                flags=cv2.INTER_NEAREST,
            )
            rectified_center, _ = pointer_center_and_tip(rectified_mask)
            direction, rectified_tip = pointer_from_rectified_mask(rectified_mask)
            rectified_tip = rectified_center + direction * DIAL_RADIUS
            full_dial_recognition = self.ocr(rectified)
            visible_text = " ".join(
                (
                    visible_text_context,
                    visible_text_from_ocr_result(full_dial_recognition),
                )
            ).strip()
            labels = recognize_numeric_sectors(self.ocr, rectified)
            color_result: ColorScaleResult | None = None
            if (
                len(
                    consolidate_duplicate_labels(
                        labels,
                        (
                            (RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2),
                            (2 * DIAL_RADIUS, 2 * DIAL_RADIUS),
                            0.0,
                        ),
                    )
                )
                < 5
            ):
                try:
                    color_result = read_color_segments(
                        rectified, direction, self.units_per_major_segment
                    )
                except ValueError:
                    labels = recognize_full_dial(
                        self.ocr, rectified, full_dial_recognition
                    )
            self.last_rectified = rectified
            self.last_ring = unwrap_scale_ring(rectified)
        except ValueError as error:
            synchronize(self.device)
            inference_end = time.perf_counter_ns()
            fallback_center, fallback_tip = pointer_center_and_tip(mask)
            global_center = fallback_center + np.asarray((x1, y1))
            global_tip = fallback_tip + np.asarray((x1, y1))
            result = GaugeResult(
                True,
                bbox,
                detection_confidence,
                True,
                (float(global_center[0]), float(global_center[1])),
                (float(global_tip[0]), float(global_tip[1])),
                angle_from_points(fallback_center, fallback_tip),
                None,
                None,
                None,
                min(detection_confidence, pointer_confidence),
                f"{pointer_method}+unrectified-pointer-fallback",
                StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    0.0,
                ),
                failure_reason=str(error),
            )
            return self._interpret_result(result, visible_text_context)
        synchronize(self.device)
        inference_end = time.perf_counter_ns()

        postprocess_start = time.perf_counter_ns()
        circle: Ellipse = (
            (RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2),
            (2 * DIAL_RADIUS, 2 * DIAL_RADIUS),
            0.0,
        )
        inverse = cv2.invertAffineTransform(transform)
        crop_center = inverse @ np.asarray(
            (rectified_center[0], rectified_center[1], 1.0)
        )
        crop_tip = inverse @ np.asarray((rectified_tip[0], rectified_tip[1], 1.0))
        global_center = crop_center + np.asarray((x1, y1))
        global_tip = crop_tip + np.asarray((x1, y1))
        angle = angle_from_points(rectified_center, rectified_tip)
        try:
            fit, inliers, rejected = robust_scale_fit(labels, circle)
            reading = float(reading_from_fit(direction, fit))
            ocr_confidence = float(np.mean([label.confidence for label in inliers]))
            value_span = max(1.0, float(np.ptp([label.value for label in inliers])))
            scale_confidence = max(0.0, 1.0 - fit.rmse / value_span)
            confidence = min(
                detection_confidence,
                pointer_confidence,
                ocr_confidence,
                scale_confidence,
            )
            postprocess_end = time.perf_counter_ns()
            result = GaugeResult(
                detected=True,
                bbox=bbox,
                detection_confidence=detection_confidence,
                pointer_found=True,
                center=(float(global_center[0]), float(global_center[1])),
                pointer_tip=(float(global_tip[0]), float(global_tip[1])),
                angle_degrees=angle,
                sweep_fraction=None,
                reading=reading,
                unit=None,
                confidence=confidence,
                center_method=(f"{pointer_method}+edge-ellipse+affine-rectification"),
                timings=StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    (postprocess_end - postprocess_start) / 1e6,
                ),
                ocr_labels=tuple(
                    label.text for label in sorted(inliers, key=lambda item: item.value)
                ),
                rejected_numeric_labels=tuple(label.text for label in rejected),
                scale_rmse=fit.rmse,
            )
            return self._interpret_result(result, visible_text)
        except ValueError as numeric_error:
            if color_result is None:
                try:
                    color_result = read_color_segments(
                        rectified, direction, self.units_per_major_segment
                    )
                except ValueError:
                    color_result = None
            if color_result is not None:
                postprocess_end = time.perf_counter_ns()
                result = GaugeResult(
                    detected=True,
                    bbox=bbox,
                    detection_confidence=detection_confidence,
                    pointer_found=True,
                    center=(float(global_center[0]), float(global_center[1])),
                    pointer_tip=(float(global_tip[0]), float(global_tip[1])),
                    angle_degrees=angle,
                    sweep_fraction=None,
                    reading=color_result.reading,
                    unit="relative_segment",
                    confidence=min(
                        detection_confidence,
                        pointer_confidence,
                        color_result.confidence,
                    ),
                    center_method=f"{pointer_method}+edge-ellipse+color-segment-scale",
                    timings=StageTimings(
                        (preprocess_end - preprocess_start) / 1e6,
                        (inference_end - inference_start) / 1e6,
                        (postprocess_end - postprocess_start) / 1e6,
                    ),
                )
                return self._interpret_result(result, visible_text)
            postprocess_end = time.perf_counter_ns()
            result = GaugeResult(
                detected=True,
                bbox=bbox,
                detection_confidence=detection_confidence,
                pointer_found=True,
                center=(float(global_center[0]), float(global_center[1])),
                pointer_tip=(float(global_tip[0]), float(global_tip[1])),
                angle_degrees=angle,
                sweep_fraction=None,
                reading=None,
                unit=None,
                confidence=min(detection_confidence, pointer_confidence),
                center_method=(f"{pointer_method}+edge-ellipse+affine-rectification"),
                timings=StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    (postprocess_end - postprocess_start) / 1e6,
                ),
                ocr_labels=tuple(label.text for label in labels),
                failure_reason=str(numeric_error),
            )
            return self._interpret_result(result, visible_text)
