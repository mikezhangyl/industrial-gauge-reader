"""ETHZ gauge detection/segmentation plus local Vision scale interpretation."""

from __future__ import annotations

import itertools
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.gauge_reader import (
    GaugeResult,
    StageTimings,
    angle_from_points,
    synchronize,
)
from src.vision_ocr import OCRObservation, VisionOCR

DETECTION_SIZE = 640
SEGMENTATION_SIZE = 448
NUMBER_PATTERN = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")


@dataclass(frozen=True)
class NumericLabel:
    text: str
    value: float
    confidence: float
    center: tuple[float, float]


@dataclass(frozen=True)
class ScaleFit:
    slope: float
    intercept: float
    direction: int
    phase_min: float
    phase_max: float
    rmse: float


Ellipse = tuple[tuple[float, float], tuple[float, float], float]


def numeric_labels(
    observations: list[OCRObservation],
    image_shape: tuple[int, ...],
    bbox: tuple[int, int, int, int],
) -> list[NumericLabel]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    labels: list[NumericLabel] = []
    for observation in observations:
        text = observation.text.strip().replace(" ", "")
        if observation.confidence < 0.5 or not NUMBER_PATTERN.fullmatch(text):
            continue
        center_x = (observation.x + observation.width / 2.0) * width
        center_y = (1.0 - observation.y - observation.height / 2.0) * height
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            labels.append(
                NumericLabel(
                    text=text,
                    value=float(text.replace(",", ".")),
                    confidence=observation.confidence,
                    center=(center_x, center_y),
                )
            )
    return labels


def ellipse_residual(ellipse: Ellipse, points: np.ndarray) -> np.ndarray:
    (center_x, center_y), (axis_a, axis_b), angle = ellipse
    theta = math.radians(angle)
    delta_x = points[:, 0] - center_x
    delta_y = points[:, 1] - center_y
    local_x = delta_x * math.cos(theta) + delta_y * math.sin(theta)
    local_y = -delta_x * math.sin(theta) + delta_y * math.cos(theta)
    normalized = (local_x / (axis_a / 2.0)) ** 2 + (local_y / (axis_b / 2.0)) ** 2
    return np.abs(normalized - 1.0)


def fit_label_ellipse(
    labels: list[NumericLabel], bbox: tuple[int, int, int, int]
) -> tuple[Ellipse, list[NumericLabel], list[NumericLabel]]:
    if len(labels) < 5:
        raise ValueError("Need at least five numeric OCR labels to fit a dial ellipse")
    labels = sorted(labels, key=lambda label: label.confidence, reverse=True)[:12]
    points = np.asarray([label.center for label in labels], dtype=np.float32)
    x1, y1, x2, y2 = bbox
    box_width, box_height = x2 - x1, y2 - y1
    best: tuple[int, float, Ellipse, np.ndarray] | None = None
    for indices in itertools.combinations(range(len(labels)), 5):
        sample = points[list(indices)].reshape(-1, 1, 2)
        ellipse_raw = cv2.fitEllipse(sample)
        ellipse: Ellipse = (
            (float(ellipse_raw[0][0]), float(ellipse_raw[0][1])),
            (float(ellipse_raw[1][0]), float(ellipse_raw[1][1])),
            float(ellipse_raw[2]),
        )
        (center_x, center_y), (axis_a, axis_b), _ = ellipse
        minor, major = sorted((axis_a, axis_b))
        if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
            continue
        if minor < max(40.0, box_height * 0.12) or major < max(100.0, box_width * 0.20):
            continue
        if (
            major > box_width * 1.15
            or minor > box_height * 1.15
            or minor / major < 0.20
        ):
            continue
        residuals = ellipse_residual(ellipse, points)
        inlier_mask = residuals < 0.22
        inlier_count = int(np.count_nonzero(inlier_mask))
        mean_error = (
            float(np.mean(residuals[inlier_mask])) if inlier_count else math.inf
        )
        candidate = (inlier_count, mean_error, ellipse, inlier_mask)
        if best is None or (candidate[0], -candidate[1]) > (best[0], -best[1]):
            best = candidate
    if best is None or best[0] < 5:
        raise ValueError("Numeric OCR labels do not form a stable dial ellipse")

    inlier_points = points[best[3]].reshape(-1, 1, 2)
    refit_raw = cv2.fitEllipse(inlier_points)
    refit: Ellipse = (
        (float(refit_raw[0][0]), float(refit_raw[0][1])),
        (float(refit_raw[1][0]), float(refit_raw[1][1])),
        float(refit_raw[2]),
    )
    residuals = ellipse_residual(refit, points)
    final_mask = residuals < 0.22
    inliers = [label for label, keep in zip(labels, final_mask, strict=True) if keep]
    rejected = [
        label for label, keep in zip(labels, final_mask, strict=True) if not keep
    ]
    return refit, inliers, rejected


def ellipse_phase(ellipse: Ellipse, point: np.ndarray) -> float:
    (center_x, center_y), (axis_a, axis_b), angle = ellipse
    theta = math.radians(angle)
    delta_x = float(point[0] - center_x)
    delta_y = float(point[1] - center_y)
    local_x = delta_x * math.cos(theta) + delta_y * math.sin(theta)
    local_y = -delta_x * math.sin(theta) + delta_y * math.cos(theta)
    return (
        math.degrees(math.atan2(local_y / (axis_b / 2.0), local_x / (axis_a / 2.0)))
        + 360.0
    ) % 360.0


def fit_linear_scale(ellipse: Ellipse, labels: list[NumericLabel]) -> ScaleFit:
    unique: dict[float, NumericLabel] = {}
    for label in labels:
        previous = unique.get(label.value)
        if previous is None or label.confidence > previous.confidence:
            unique[label.value] = label
    ordered = sorted(unique.values(), key=lambda label: label.value)
    if len(ordered) < 2:
        raise ValueError("Need at least two distinct scale labels")

    values = np.asarray([label.value for label in ordered], dtype=np.float64)
    raw_phases = np.asarray(
        [ellipse_phase(ellipse, np.asarray(label.center)) for label in ordered],
        dtype=np.float64,
    )
    fits: list[ScaleFit] = []
    for direction in (1, -1):
        directed = (direction * raw_phases) % 360.0
        unwrapped = [float(directed[0])]
        for phase in directed[1:]:
            candidate = float(phase)
            while candidate <= unwrapped[-1]:
                candidate += 360.0
            unwrapped.append(candidate)
        phases = np.asarray(unwrapped, dtype=np.float64)
        if phases[-1] - phases[0] > 720.0:
            continue
        slope, intercept = np.polyfit(phases, values, 1)
        predicted = slope * phases + intercept
        rmse = float(np.sqrt(np.mean((values - predicted) ** 2)))
        if slope > 0:
            fits.append(
                ScaleFit(
                    slope=float(slope),
                    intercept=float(intercept),
                    direction=direction,
                    phase_min=float(phases[0]),
                    phase_max=float(phases[-1]),
                    rmse=rmse,
                )
            )
    if not fits:
        raise ValueError("Could not fit a monotonic scale")
    fit = min(fits, key=lambda candidate: candidate.rmse)
    value_span = float(np.ptp(values))
    if fit.rmse > max(0.35, value_span * 0.05):
        raise ValueError(f"Scale fit residual is too high: {fit.rmse:.3f}")
    return fit


def pointer_from_mask(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    ellipse: Ellipse,
) -> tuple[np.ndarray, np.ndarray]:
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) < 20:
        raise ValueError("Pointer mask has too few pixels")
    x1, y1, x2, y2 = bbox
    points = np.column_stack(
        (
            x1 + x_coords * (x2 - x1) / SEGMENTATION_SIZE,
            y1 + y_coords * (y2 - y1) / SEGMENTATION_SIZE,
        )
    ).astype(np.float64)
    center = np.asarray(ellipse[0], dtype=np.float64)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = centered @ axis
    positive_extent = (
        float(np.percentile(projections[projections > 0], 99))
        if np.any(projections > 0)
        else 0.0
    )
    negative_extent = (
        float(np.percentile(-projections[projections < 0], 99))
        if np.any(projections < 0)
        else 0.0
    )
    direction = axis if positive_extent >= negative_extent else -axis

    (center_x, center_y), (axis_a, axis_b), angle = ellipse
    theta = math.radians(angle)
    local_x = direction[0] * math.cos(theta) + direction[1] * math.sin(theta)
    local_y = -direction[0] * math.sin(theta) + direction[1] * math.cos(theta)
    radius = 1.0 / math.sqrt(
        (local_x / (axis_a / 2.0)) ** 2 + (local_y / (axis_b / 2.0)) ** 2
    )
    tip = np.asarray((center_x, center_y), dtype=np.float64) + direction * radius
    return direction, tip


def reading_from_pointer(
    ellipse: Ellipse, direction: np.ndarray, fit: ScaleFit
) -> float:
    center = np.asarray(ellipse[0], dtype=np.float64)
    raw_phase = (fit.direction * ellipse_phase(ellipse, center + direction)) % 360.0
    equivalents = [raw_phase + 360.0 * turn for turn in range(-2, 5)]
    target = (fit.phase_min + fit.phase_max) / 2.0
    phase = min(equivalents, key=lambda candidate: abs(candidate - target))
    return fit.slope * phase + fit.intercept


class EthzVisionGaugeReader:
    """Keep ETHZ models resident; Vision OCR remains part of every timed read."""

    def __init__(
        self,
        detector_path: Path,
        segmentation_path: Path,
        device: str,
        vision_ocr: VisionOCR,
    ):
        self.device = device
        self.detector = YOLO(str(detector_path))
        self.segmenter = YOLO(str(segmentation_path))
        self.vision_ocr = vision_ocr

    def read(self, image_path: Path) -> GaugeResult:
        preprocess_start = time.perf_counter_ns()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot decode image: {image_path}")
        preprocess_end = time.perf_counter_ns()

        synchronize(self.device)
        inference_start = time.perf_counter_ns()
        detection = self.detector.predict(
            image,
            imgsz=DETECTION_SIZE,
            conf=0.25,
            device=self.device,
            verbose=False,
        )[0]
        if detection.boxes is None or len(detection.boxes) == 0:
            synchronize(self.device)
            inference_end = time.perf_counter_ns()
            return GaugeResult(
                detected=False,
                bbox=None,
                detection_confidence=None,
                pointer_found=False,
                center=None,
                pointer_tip=None,
                angle_degrees=None,
                sweep_fraction=None,
                reading=None,
                unit=None,
                confidence=None,
                center_method=None,
                timings=StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    0.0,
                ),
            )
        best_index = int(torch.argmax(detection.boxes.conf).item())
        detection_confidence = float(detection.boxes.conf[best_index].detach().cpu())
        height, width = image.shape[:2]
        raw_box = (
            detection.boxes.xyxy[best_index].detach().cpu().numpy().round().astype(int)
        )
        x1, y1, x2, y2 = raw_box.tolist()
        bbox = (
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        )
        x1, y1, x2, y2 = bbox
        crop = cv2.resize(
            image[y1:y2, x1:x2],
            (SEGMENTATION_SIZE, SEGMENTATION_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        segmentation = self.segmenter.predict(
            crop,
            imgsz=SEGMENTATION_SIZE,
            conf=0.20,
            device=self.device,
            verbose=False,
        )[0]
        observations = self.vision_ocr.recognize(image_path)
        synchronize(self.device)
        inference_end = time.perf_counter_ns()

        postprocess_start = time.perf_counter_ns()
        labels = numeric_labels(observations, image.shape, bbox)
        try:
            ellipse, inliers, rejected = fit_label_ellipse(labels, bbox)
            scale = fit_linear_scale(ellipse, inliers)
            if segmentation.masks is None or len(segmentation.masks.data) == 0:
                raise ValueError("Pointer segmentation returned no mask")
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
            if mask.shape != (SEGMENTATION_SIZE, SEGMENTATION_SIZE):
                mask = cv2.resize(
                    mask,
                    (SEGMENTATION_SIZE, SEGMENTATION_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
            direction, tip = pointer_from_mask(mask, bbox, ellipse)
            reading = float(reading_from_pointer(ellipse, direction, scale))
            center = np.asarray(ellipse[0], dtype=np.float64)
            angle = angle_from_points(center, center + direction)
            ocr_confidence = float(np.mean([label.confidence for label in inliers]))
            confidence = min(
                detection_confidence,
                pointer_confidence,
                ocr_confidence,
                max(0.0, 1.0 - scale.rmse),
            )
            postprocess_end = time.perf_counter_ns()
            return GaugeResult(
                detected=True,
                bbox=bbox,
                detection_confidence=detection_confidence,
                pointer_found=True,
                center=(float(center[0]), float(center[1])),
                pointer_tip=(float(tip[0]), float(tip[1])),
                angle_degrees=angle,
                sweep_fraction=None,
                reading=reading,
                unit=None,
                confidence=confidence,
                center_method="ocr_label_ellipse+pca_pointer_mask",
                timings=StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    (postprocess_end - postprocess_start) / 1e6,
                ),
                ocr_labels=tuple(
                    label.text for label in sorted(inliers, key=lambda item: item.value)
                ),
                rejected_numeric_labels=tuple(label.text for label in rejected),
                scale_rmse=scale.rmse,
            )
        except ValueError:
            postprocess_end = time.perf_counter_ns()
            return GaugeResult(
                detected=True,
                bbox=bbox,
                detection_confidence=detection_confidence,
                pointer_found=False,
                center=None,
                pointer_tip=None,
                angle_degrees=None,
                sweep_fraction=None,
                reading=None,
                unit=None,
                confidence=detection_confidence,
                center_method=None,
                timings=StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    (postprocess_end - postprocess_start) / 1e6,
                ),
                ocr_labels=tuple(label.text for label in labels),
            )
