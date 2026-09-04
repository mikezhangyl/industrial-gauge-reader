"""Full-image gauge pose detection and pointer-angle extraction."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.inference_device import synchronize

MODEL_IMAGE_SIZE = 640


@dataclass(frozen=True)
class StageTimings:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass(frozen=True)
class GaugeResult:
    detected: bool
    bbox: tuple[int, int, int, int] | None
    detection_confidence: float | None
    pointer_found: bool
    center: tuple[float, float] | None
    pointer_tip: tuple[float, float] | None
    angle_degrees: float | None
    sweep_fraction: float | None
    reading: float | str | None
    unit: str | None
    confidence: float | None
    center_method: str | None
    timings: StageTimings
    ocr_labels: tuple[str, ...] = ()
    rejected_numeric_labels: tuple[str, ...] = ()
    scale_rmse: float | None = None
    failure_reason: str | None = None
    raw_reading: float | str | None = None
    instrument_type_id: str | None = None
    readout_channel_id: str | None = None
    interpretation_method: str | None = None
    reading_candidates: tuple[float, ...] = ()

    @property
    def level1(self) -> bool:
        return self.detected

    @property
    def level2(self) -> bool:
        return self.detected and self.pointer_found

    @property
    def level3(self) -> bool:
        return self.reading is not None


def angle_from_points(
    center: tuple[float, float] | np.ndarray,
    tip: tuple[float, float] | np.ndarray,
) -> float:
    """Return 0 degrees at 12 o'clock, increasing clockwise."""
    delta_x = float(tip[0] - center[0])
    delta_y = float(tip[1] - center[1])
    return (math.degrees(math.atan2(delta_x, -delta_y)) + 360.0) % 360.0


def format_reading_value(value: float | str) -> str:
    """Format continuous values and categorical dial labels without coercion."""
    return value if isinstance(value, str) else f"{value:.2f}"


def sweep_position(
    start: np.ndarray, center: np.ndarray, end: np.ndarray, tip: np.ndarray
) -> tuple[float, float, float | None]:
    """Return absolute tip angle, learned sweep angle and normalized position."""
    start_angle = angle_from_points(center, start)
    tip_angle = angle_from_points(center, tip)
    end_angle = angle_from_points(center, end)
    total_sweep = (end_angle - start_angle) % 360.0
    tip_sweep = (tip_angle - start_angle) % 360.0
    if total_sweep < 1.0 or tip_sweep > total_sweep + 5.0:
        return tip_angle, total_sweep, None
    return tip_angle, total_sweep, float(np.clip(tip_sweep / total_sweep, 0.0, 1.0))


def refine_tip_with_hough(
    rectified: np.ndarray,
    center: np.ndarray,
    pose_tip: np.ndarray,
    angle_tolerance: float = 10.0,
) -> tuple[np.ndarray, bool]:
    """Apply the upstream project's generic line-based pose-tip sanity check."""
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 5)
    edges = cv2.Canny(blurred, 20, 60)
    lines = cv2.HoughLinesP(
        edges,
        rho=3,
        theta=np.pi / 180.0,
        threshold=10,
        minLineLength=int(MODEL_IMAGE_SIZE * 0.09),
        maxLineGap=10,
    )
    if lines is None or len(lines) == 0:
        return pose_tip, False

    candidates: list[tuple[float, float, np.ndarray]] = []
    for raw_line in lines[:, 0]:
        start = raw_line[:2].astype(np.float32)
        end = raw_line[2:].astype(np.float32)
        start_distance = float(np.linalg.norm(start - center))
        end_distance = float(np.linalg.norm(end - center))
        nearest = min(start_distance, end_distance)
        length = float(np.linalg.norm(end - start))
        if nearest <= MODEL_IMAGE_SIZE * 0.16:
            far_endpoint = start if start_distance > end_distance else end
            candidates.append((nearest, -length, far_endpoint))
    if not candidates:
        return pose_tip, False

    _, _, candidate_tip = min(candidates, key=lambda item: (item[0], item[1]))
    pose_angle = angle_from_points(center, pose_tip)
    candidate_angle = angle_from_points(center, candidate_tip)
    difference = abs((candidate_angle - pose_angle + 180.0) % 360.0 - 180.0)
    if difference > angle_tolerance:
        return candidate_tip, True
    return pose_tip, False


def _empty_result(
    timings: StageTimings,
    *,
    detected: bool = False,
    bbox: tuple[int, int, int, int] | None = None,
    detection_confidence: float | None = None,
) -> GaugeResult:
    return GaugeResult(
        detected=detected,
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
        timings=timings,
    )


class AnalogGaugeReader:
    """Keep both pretrained YOLO Pose models resident across benchmark runs."""

    def __init__(self, detector_path: Path, pointer_path: Path, device: str):
        self.device = device
        self.detector = YOLO(str(detector_path))
        self.pointer_pose = YOLO(str(pointer_path))

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
            imgsz=MODEL_IMAGE_SIZE,
            conf=0.25,
            device=self.device,
            verbose=False,
        )[0]
        if (
            detection.boxes is None
            or len(detection.boxes) == 0
            or detection.keypoints is None
            or len(detection.keypoints.xy) == 0
        ):
            synchronize(self.device)
            inference_end = time.perf_counter_ns()
            return _empty_result(
                StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    0.0,
                )
            )

        best_index = int(torch.argmax(detection.boxes.conf).item())
        detection_confidence = float(detection.boxes.conf[best_index].detach().cpu())
        box_values = detection.boxes.xyxy[best_index].detach().cpu().numpy()
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box_values.round().astype(int).tolist()
        bbox = (
            max(0, min(width, x1)),
            max(0, min(height, y1)),
            max(0, min(width, x2)),
            max(0, min(height, y2)),
        )
        corners = (
            detection.keypoints.xy[best_index].detach().cpu().numpy().astype(np.float32)
        )
        if corners.shape != (4, 2) or np.any(~np.isfinite(corners)):
            synchronize(self.device)
            inference_end = time.perf_counter_ns()
            return _empty_result(
                StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    0.0,
                ),
                detected=True,
                bbox=bbox,
                detection_confidence=detection_confidence,
            )

        destination = np.asarray(
            [
                (0.0, 0.0),
                (MODEL_IMAGE_SIZE - 1.0, 0.0),
                (MODEL_IMAGE_SIZE - 1.0, MODEL_IMAGE_SIZE - 1.0),
                (0.0, MODEL_IMAGE_SIZE - 1.0),
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(corners, destination)
        rectified = cv2.warpPerspective(
            image, homography, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
        )
        pointer = self.pointer_pose.predict(
            rectified,
            imgsz=MODEL_IMAGE_SIZE,
            conf=0.20,
            device=self.device,
            verbose=False,
        )[0]
        synchronize(self.device)
        inference_end = time.perf_counter_ns()

        postprocess_start = time.perf_counter_ns()
        if (
            pointer.boxes is None
            or len(pointer.boxes) == 0
            or pointer.keypoints is None
            or len(pointer.keypoints.xy) == 0
        ):
            postprocess_end = time.perf_counter_ns()
            return _empty_result(
                StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    (postprocess_end - postprocess_start) / 1e6,
                ),
                detected=True,
                bbox=bbox,
                detection_confidence=detection_confidence,
            )

        pointer_index = int(torch.argmax(pointer.boxes.conf).item())
        pointer_confidence = float(pointer.boxes.conf[pointer_index].detach().cpu())
        keypoints = (
            pointer.keypoints.xy[pointer_index]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if keypoints.shape != (4, 2) or np.any(~np.isfinite(keypoints)):
            postprocess_end = time.perf_counter_ns()
            return _empty_result(
                StageTimings(
                    (preprocess_end - preprocess_start) / 1e6,
                    (inference_end - inference_start) / 1e6,
                    (postprocess_end - postprocess_start) / 1e6,
                ),
                detected=True,
                bbox=bbox,
                detection_confidence=detection_confidence,
            )

        start, center_rectified, end, tip_rectified = keypoints
        tip_rectified, tip_corrected = refine_tip_with_hough(
            rectified, center_rectified, tip_rectified
        )
        angle, _total_sweep, fraction = sweep_position(
            start, center_rectified, end, tip_rectified
        )
        inverse = cv2.getPerspectiveTransform(destination, corners)
        mapped = cv2.perspectiveTransform(
            np.asarray([[center_rectified, tip_rectified]], dtype=np.float32), inverse
        )[0]
        center = (float(mapped[0][0]), float(mapped[0][1]))
        tip = (float(mapped[1][0]), float(mapped[1][1]))
        pointer_length = float(np.linalg.norm(tip_rectified - center_rectified))
        pointer_found = pointer_length >= MODEL_IMAGE_SIZE * 0.08
        confidence = min(detection_confidence, pointer_confidence)
        if fraction is None:
            confidence *= 0.8
        postprocess_end = time.perf_counter_ns()
        return GaugeResult(
            detected=True,
            bbox=bbox,
            detection_confidence=detection_confidence,
            pointer_found=pointer_found,
            center=center if pointer_found else None,
            pointer_tip=tip if pointer_found else None,
            angle_degrees=angle if pointer_found else None,
            sweep_fraction=fraction if pointer_found else None,
            reading=None,
            unit=None,
            confidence=confidence,
            center_method=(
                "pretrained_pose_center+hough_tip_correction"
                if tip_corrected
                else "pretrained_pose_keypoints"
            ),
            timings=StageTimings(
                (preprocess_end - preprocess_start) / 1e6,
                (inference_end - inference_start) / 1e6,
                (postprocess_end - postprocess_start) / 1e6,
            ),
        )


def annotate(image_path: Path, result: GaugeResult, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {image_path}")
    if result.bbox is not None:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 3)
    if result.center is not None:
        center = tuple(round(value) for value in result.center)
        cv2.circle(image, center, 7, (255, 80, 0), -1)
    else:
        center = None
    if center is not None and result.pointer_tip is not None:
        tip = tuple(round(value) for value in result.pointer_tip)
        cv2.arrowedLine(image, center, tip, (0, 0, 255), 4, tipLength=0.08)

    angle_text = (
        "N/A" if result.angle_degrees is None else f"{result.angle_degrees:.2f} deg"
    )
    fraction_text = (
        "N/A"
        if result.sweep_fraction is None
        else f"{result.sweep_fraction * 100.0:.1f}%"
    )
    if result.reading is not None:
        reading_text = (
            f"{format_reading_value(result.reading)} {result.unit or ''}".rstrip()
        )
    elif result.reading_candidates:
        candidates = "/".join(f"{value:.2f}" for value in result.reading_candidates)
        reading_text = f"{candidates} {result.unit or ''} (scale ambiguous)".rstrip()
    else:
        reading_text = "N/A (scale not interpreted)"
    lines: list[str] = []
    if result.instrument_type_id:
        lines.append(f"Instrument: {result.instrument_type_id}")
    lines.extend(
        [
            f"Pointer angle: {angle_text}",
            f"Sweep position: {fraction_text}",
            f"Reading: {reading_text}",
        ]
    )
    if result.raw_reading is not None and result.raw_reading != result.reading:
        lines.append(
            f"Raw visual reading: {format_reading_value(result.raw_reading)}"
        )
    for index, line in enumerate(lines):
        y = 38 + index * 34
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(
            image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to save visualization: {output_path}")
