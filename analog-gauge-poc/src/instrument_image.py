"""Metadata-aware, multi-channel analysis for one full instrument image."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from src.gauge_reader import GaugeResult, StageTimings, angle_from_points
from src.instrument_metadata import (
    InstrumentMetadataCatalog,
    InstrumentTypeMetadata,
    ReadoutChannel,
)
from src.rapidocr_reader import (
    DialCandidate,
    EthzPaddleGaugeReader,
    pointer_center_and_tip,
    visible_text_from_ocr_result,
)

COUNTER_CHARACTER_MAP = str.maketrans(
    {
        "O": "0",
        "D": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "E": "5",
        "G": "6",
        "B": "8",
    }
)
COUNTER_TOKEN_PATTERN = re.compile(r"^[0-9ODQILZSEGB]{3,8}$", re.IGNORECASE)
DISCRETE_DIAL_LABEL_PATTERN = re.compile(r"^\d{1,2}[A-Z]?$", re.IGNORECASE)
POINTER_DISPLAY_TYPES = frozenset(
    {
        "analog_pointer",
        "analog_pointer_with_counterweight",
        "discrete_pointer_dial",
        "single_pointer_circular_counter",
    }
)
GENERATED_VISUALIZATION_FAILURE = (
    "Input image contains a generated gauge visualization overlay; "
    "use the original unannotated photo"
)
_GENERATED_OVERLAY_TOKEN_GROUPS = (
    ("pointer", "angle"),
    ("sweep", "position"),
    ("reading",),
)


@dataclass(frozen=True)
class CounterOCRCandidate:
    raw_text: str
    normalized_display: str
    value: int
    confidence: float
    center: tuple[float, float]
    background_luma: float | None


@dataclass(frozen=True)
class ChannelAnalysis:
    instance_id: str
    channel_id: str
    value: float | int | str | None
    unit: str
    status: str
    method: str | None
    confidence: float | None
    raw_display: str | None = None
    raw_ocr_text: str | None = None
    candidates: tuple[float, ...] = ()
    note_zh: str | None = None


@dataclass(frozen=True)
class InstrumentImageAnalysis:
    image_sha256: str
    instrument_type_id: str | None
    visible_text: str
    instances: tuple[str, ...]
    pointer_results: tuple[GaugeResult, ...]
    channels: tuple[ChannelAnalysis, ...]
    failure_reason: str | None = None


class MetadataAwareImageAnalyzer:
    """Return every metadata-declared channel without mixing in human answers."""

    def __init__(
        self,
        reader: EthzPaddleGaugeReader,
        catalog: InstrumentMetadataCatalog,
    ):
        self.reader = reader
        self.catalog = catalog

    def analyze(self, image_path: Path) -> InstrumentImageAnalysis:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot decode image: {image_path}")
        image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        full_ocr = self.reader.ocr(image)
        visible_text = visible_text_from_ocr_result(full_ocr)
        if has_generated_visualization_overlay(visible_text):
            return InstrumentImageAnalysis(
                image_sha256=image_sha256,
                instrument_type_id=None,
                visible_text=visible_text,
                instances=(),
                pointer_results=(),
                channels=(),
                failure_reason=GENERATED_VISUALIZATION_FAILURE,
            )
        dial_candidates = self.reader.detect_dial_candidates(image_path)
        matches = self.catalog.find(visible_text)
        if len(matches) != 1:
            detected_results = tuple(
                _empty_pointer_result(candidate) for candidate in dial_candidates[:5]
            )
            return InstrumentImageAnalysis(
                image_sha256=image_sha256,
                instrument_type_id=None,
                visible_text=visible_text,
                instances=(),
                pointer_results=detected_results,
                channels=(),
                failure_reason=(
                    "No unique instrument type matched full-image OCR"
                    if not matches
                    else "Multiple instrument types matched full-image OCR"
                ),
            )

        metadata = matches[0]
        counters = extract_counter_candidates(full_ocr, image)
        mechanical_channels = tuple(
            channel
            for channel in metadata.readout_channels
            if channel.display_type == "mechanical_counter"
        )
        if metadata.type_id == "surge_arrester_monitor":
            selected_dials = _select_similar_dial_row(dial_candidates)
            if selected_dials:
                counters = _extract_counters_for_dials(
                    self.reader,
                    image,
                    selected_dials,
                )
            instance_count = max(len(selected_dials), 1)
        else:
            instance_count = len(counters) if mechanical_channels and counters else 1
            selected_dials = _select_instance_dials(dial_candidates, instance_count)
        pointer_results = self._read_dials(
            image,
            selected_dials,
            visible_text,
            image_path.name,
        )
        if instance_count == 1:
            full_result = self.reader.read(
                image_path,
                visible_text_context=visible_text,
            )
            pointer_results = (
                max(
                    (full_result, *pointer_results),
                    key=_result_rank,
                ),
            )
        pointer_results = _recover_type_specific_pointer_results(
            image,
            metadata,
            selected_dials,
            pointer_results,
            visible_text,
            self.reader,
        )
        if len(pointer_results) > instance_count:
            pointer_results = pointer_results[:instance_count]
        instance_count = max(instance_count, len(pointer_results), 1)
        instances = tuple(f"instance_{index + 1}" for index in range(instance_count))
        channels = _channel_analyses(
            metadata,
            instances,
            pointer_results,
            counters,
        )
        return InstrumentImageAnalysis(
            image_sha256=image_sha256,
            instrument_type_id=metadata.type_id,
            visible_text=visible_text,
            instances=instances,
            pointer_results=pointer_results,
            channels=channels,
        )

    def _read_dials(
        self,
        image: np.ndarray,
        candidates: tuple[DialCandidate, ...],
        visible_text: str,
        source_name: str,
    ) -> tuple[GaugeResult, ...]:
        image_height, image_width = image.shape[:2]
        results: list[GaugeResult] = []
        with TemporaryDirectory(prefix="instrument-gauge-") as temporary_directory:
            temporary_path = Path(temporary_directory)
            for index, candidate in enumerate(candidates):
                crop_bbox = _padded_bbox(
                    candidate.bbox,
                    image_width,
                    image_height,
                )
                x1, y1, x2, y2 = crop_bbox
                crop_path = temporary_path / f"{Path(source_name).stem}-{index}.jpg"
                if not cv2.imwrite(str(crop_path), image[y1:y2, x1:x2]):
                    raise RuntimeError(
                        f"Failed to create temporary dial crop: {crop_path}"
                    )
                local_result = self.reader.read(
                    crop_path,
                    visible_text_context=visible_text,
                )
                results.append(_offset_result(local_result, x1, y1))
        return tuple(
            sorted(
                results,
                key=lambda result: result.bbox[0] if result.bbox is not None else 0,
            )
        )


def has_generated_visualization_overlay(visible_text: str) -> bool:
    """Detect this project's rendered debug labels, including noisy OCR variants."""
    tokens = re.findall(r"[a-z]+", visible_text.casefold())
    matched_groups = 0
    for group in _GENERATED_OVERLAY_TOKEN_GROUPS:
        if all(_has_similar_token(tokens, expected) for expected in group):
            matched_groups += 1
    return matched_groups >= 2


def _has_similar_token(tokens: list[str], expected: str) -> bool:
    return any(
        SequenceMatcher(None, token, expected).ratio() >= 0.72 for token in tokens
    )


def select_meter_hub(
    circles: np.ndarray,
    *,
    face_center_x: float,
    crop_shape: tuple[int, int],
    face_bottom_y: float | None = None,
) -> np.ndarray:
    """Select the adjustment screw at the bottom-center of a panel meter."""
    height, width = crop_shape
    candidates: list[tuple[float, np.ndarray]] = []
    for raw_circle in np.asarray(circles, dtype=np.float64).reshape(-1, 3):
        x, y, radius = raw_circle
        if not 0.20 * width <= x <= 0.70 * width:
            continue
        if not 0.62 * height <= y <= 0.84 * height:
            continue
        if not 0.008 * width <= radius <= 0.07 * width:
            continue
        if abs(x - face_center_x) > 0.15 * width:
            continue
        if face_bottom_y is not None and y > face_bottom_y - 0.03 * height:
            continue
        score = abs(x - face_center_x) / width + abs(y - 0.72 * height) / height
        candidates.append((score, np.asarray((x, y), dtype=np.float64)))
    if not candidates:
        raise ValueError("Rectangular meter hub was not found")
    return min(candidates, key=lambda item: item[0])[1]


def select_meter_pointer_line(
    lines: np.ndarray,
    *,
    hub: np.ndarray,
    crop_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select the longest radial line that reaches the panel-meter hub."""
    height, width = crop_shape
    hub = np.asarray(hub, dtype=np.float64)
    candidates: list[tuple[float, float, np.ndarray, float]] = []
    for raw_line in np.asarray(lines).reshape(-1, 4):
        start = raw_line[:2].astype(np.float64)
        end = raw_line[2:].astype(np.float64)
        start_distance = float(np.linalg.norm(start - hub))
        end_distance = float(np.linalg.norm(end - hub))
        tip = start if start_distance >= end_distance else end
        near = end if start_distance >= end_distance else start
        length = float(np.linalg.norm(end - start))
        if length < 0.13 * width or float(np.linalg.norm(near - hub)) > 0.30 * width:
            continue
        if tip[1] > 0.65 * height:
            continue
        angle = angle_from_points(hub, tip)
        if not (angle >= 280.0 or angle <= 80.0):
            continue
        line_vector = end - start
        perpendicular_distance = abs(
            line_vector[0] * (start[1] - hub[1]) - line_vector[1] * (start[0] - hub[0])
        ) / max(float(np.linalg.norm(line_vector)), 1e-6)
        if perpendicular_distance > 0.12 * width:
            continue
        candidates.append((-length, perpendicular_distance, tip, angle))
    if not candidates:
        raise ValueError("Rectangular meter pointer line was not found")
    _, _, tip, angle = min(candidates, key=lambda item: (item[0], item[1]))
    return hub, tip, angle


def select_consensus_discrete_label(
    recognized: list[tuple[str, float]],
) -> tuple[str, float]:
    """Choose a stable pointer-aligned label across nearby OCR rotations."""
    grouped: dict[str, list[tuple[str, float]]] = {}
    for raw_text, raw_confidence in recognized:
        text = "".join(raw_text.split())
        confidence = float(raw_confidence)
        if confidence < 0.55 or not DISCRETE_DIAL_LABEL_PATTERN.fullmatch(text):
            continue
        grouped.setdefault(text.casefold(), []).append((text, confidence))
    stable = [items for items in grouped.values() if len(items) >= 2]
    if not stable:
        raise ValueError("Pointer-aligned discrete label OCR was not stable")
    selected = max(
        stable,
        key=lambda items: (
            len(items),
            sum(confidence for _, confidence in items),
            max(confidence for _, confidence in items),
        ),
    )
    label = max(selected, key=lambda item: item[1])[0]
    confidence = sum(item[1] for item in selected) / len(selected)
    return label, confidence


def detect_colored_component_pointer(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Recover a red pointer component that crosses the central dial area."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (((hue <= 20) | (hue >= 160)) & (saturation > 30) & (value > 20)).astype(
        np.uint8
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    image_center = np.asarray((width / 2.0, height / 2.0))
    image_area = width * height
    candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not max(40, image_area * 0.002) <= area <= image_area * 0.15:
            continue
        component = (labels == index).astype(np.uint8)
        try:
            center, tip = pointer_center_and_tip(component)
        except ValueError:
            continue
        length = float(np.linalg.norm(tip - center))
        center_distance = float(np.linalg.norm(center - image_center))
        if length < min(width, height) * 0.20:
            continue
        if center_distance > min(width, height) * 0.23:
            continue
        angle = angle_from_points(center, tip)
        score = length - center_distance * 0.75 + math.sqrt(area)
        candidates.append((score, center, tip, angle))
    if not candidates:
        raise ValueError("No central colored pointer component")
    score, center, tip, angle = max(candidates, key=lambda item: item[0])
    confidence = float(np.clip(0.45 + score / max(width, height) * 0.25, 0.45, 0.8))
    return center, tip, angle, confidence


def _recover_type_specific_pointer_results(
    image: np.ndarray,
    metadata: InstrumentTypeMetadata,
    candidates: tuple[DialCandidate, ...],
    pointer_results: tuple[GaugeResult, ...],
    visible_text: str,
    reader: EthzPaddleGaugeReader,
) -> tuple[GaugeResult, ...]:
    recovered = list(pointer_results)
    while len(recovered) < len(candidates):
        recovered.append(_empty_pointer_result(candidates[len(recovered)]))

    if metadata.type_id == "surge_arrester_monitor":
        for index, candidate in enumerate(candidates):
            existing = recovered[index]
            if existing.reading is not None or existing.reading_candidates:
                continue
            try:
                center, tip, angle = _detect_rectangular_meter_pointer(image, candidate)
            except ValueError:
                continue
            visual = replace(
                existing,
                detected=True,
                bbox=candidate.bbox,
                detection_confidence=candidate.confidence,
                pointer_found=True,
                center=(float(center[0]), float(center[1])),
                pointer_tip=(float(tip[0]), float(tip[1])),
                angle_degrees=angle,
                sweep_fraction=None,
                reading=None,
                unit=None,
                confidence=candidate.confidence,
                center_method="type-specific:rectangular-meter-hub+line",
                failure_reason=None,
                raw_reading=None,
                instrument_type_id=None,
                readout_channel_id=None,
                interpretation_method=None,
                reading_candidates=(),
            )
            interpreted = _interpret_recovered_result(reader, visual, visible_text)
            if _result_rank(interpreted) > _result_rank(existing):
                recovered[index] = interpreted

    if metadata.type_id == "shm_d_motor_drive_unit" and recovered:
        existing = recovered[0]
        if existing.reading is None and existing.pointer_found:
            try:
                label, ocr_confidence = _recognize_pointer_aligned_discrete_label(
                    image, existing, reader
                )
            except ValueError:
                pass
            else:
                reading: float | str = label
                visual = replace(
                    existing,
                    reading=reading,
                    unit=None,
                    confidence=min(existing.confidence or 1.0, ocr_confidence),
                    center_method=(
                        f"{existing.center_method}+pointer-aligned-outer-label-ocr"
                    ),
                    failure_reason=None,
                    raw_reading=None,
                    instrument_type_id=None,
                    readout_channel_id=None,
                    interpretation_method=None,
                    reading_candidates=(),
                )
                recovered[0] = _interpret_recovered_result(reader, visual, visible_text)

    if metadata.type_id == "arrester_discharge_counter" and recovered and candidates:
        existing = recovered[0]
        channel = _single_pointer_channel(metadata)
        if existing.reading is None and channel is not None and channel.allowed_values:
            x1, y1, x2, y2 = candidates[0].bbox
            try:
                center, tip, angle, color_confidence = detect_colored_component_pointer(
                    image[y1:y2, x1:x2]
                )
            except ValueError:
                pass
            else:
                step = 360.0 / len(channel.allowed_values)
                value_index = round(angle / step) % len(channel.allowed_values)
                raw_value = float(channel.allowed_values[value_index])
                offset = np.asarray((x1, y1), dtype=np.float64)
                visual = replace(
                    existing,
                    detected=True,
                    bbox=candidates[0].bbox,
                    detection_confidence=candidates[0].confidence,
                    pointer_found=True,
                    center=tuple(center + offset),
                    pointer_tip=tuple(tip + offset),
                    angle_degrees=angle,
                    reading=raw_value,
                    unit=None,
                    confidence=min(candidates[0].confidence, color_confidence),
                    center_method="type-specific:colored-component-pointer",
                    failure_reason=None,
                    raw_reading=None,
                    instrument_type_id=None,
                    readout_channel_id=None,
                    interpretation_method=None,
                    reading_candidates=(),
                )
                recovered[0] = _interpret_recovered_result(reader, visual, visible_text)
    return tuple(recovered)


def _detect_rectangular_meter_pointer(
    image: np.ndarray, candidate: DialCandidate
) -> tuple[np.ndarray, np.ndarray, float]:
    x1, y1, x2, y2 = candidate.bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Rectangular meter crop is empty")
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    face_edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 100)
    face_lines = cv2.HoughLinesP(
        face_edges,
        1,
        np.pi / 720.0,
        threshold=20,
        minLineLength=max(10, round(width * 0.25)),
        maxLineGap=15,
    )
    if face_lines is None:
        raise ValueError("Rectangular meter face was not found")
    horizontal: list[tuple[float, float, float]] = []
    for raw_line in face_lines[:, 0]:
        start = raw_line[:2].astype(np.float64)
        end = raw_line[2:].astype(np.float64)
        angle = abs(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])))
        angle = min(angle, abs(180.0 - angle))
        midpoint_y = float((start[1] + end[1]) / 2.0)
        length = float(np.linalg.norm(end - start))
        if angle <= 15.0 and 0.30 * height <= midpoint_y <= 0.82 * height:
            horizontal.append((length, float((start[0] + end[0]) / 2.0), midpoint_y))
    if not horizontal:
        raise ValueError("Rectangular meter horizontal boundary was not found")
    face_center_x = max(horizontal, key=lambda item: item[0])[1]
    lower_boundaries = [item for item in horizontal if item[2] >= 0.65 * height]
    face_bottom_y = (
        max(lower_boundaries, key=lambda item: item[2])[2] if lower_boundaries else None
    )

    roi_x1, roi_x2 = round(0.20 * width), round(0.70 * width)
    roi_y1, roi_y2 = round(0.55 * height), round(0.90 * height)
    hub_roi = cv2.GaussianBlur(gray[roi_y1:roi_y2, roi_x1:roi_x2], (3, 3), 1)
    circles = cv2.HoughCircles(
        hub_roi,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=60,
        param2=15,
        minRadius=max(4, round(width * 0.008)),
        maxRadius=max(6, round(width * 0.07)),
    )
    if circles is None:
        raise ValueError("Rectangular meter hub circle was not found")
    global_circles = circles[0].astype(np.float64)
    global_circles[:, 0] += roi_x1
    global_circles[:, 1] += roi_y1
    hub = select_meter_hub(
        global_circles,
        face_center_x=face_center_x,
        crop_shape=(height, width),
        face_bottom_y=face_bottom_y,
    )

    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    pointer_edges = cv2.Canny(cv2.GaussianBlur(enhanced, (3, 3), 0), 40, 120)
    pointer_lines = cv2.HoughLinesP(
        pointer_edges,
        1,
        np.pi / 360.0,
        threshold=25,
        minLineLength=max(10, round(min(width, height) * 0.13)),
        maxLineGap=max(4, round(min(width, height) * 0.04)),
    )
    if pointer_lines is None:
        raise ValueError("Rectangular meter pointer line was not found")
    center, tip, angle = select_meter_pointer_line(
        pointer_lines,
        hub=hub,
        crop_shape=(height, width),
    )
    offset = np.asarray((x1, y1), dtype=np.float64)
    return center + offset, tip + offset, angle


def _recognize_pointer_aligned_discrete_label(
    image: np.ndarray,
    result: GaugeResult,
    reader: EthzPaddleGaugeReader,
) -> tuple[str, float]:
    if result.bbox is None or result.center is None or result.angle_degrees is None:
        raise ValueError("Discrete dial pointer geometry is incomplete")
    x1, y1, x2, y2 = result.bbox
    dial_size = float(max(x2 - x1, y2 - y1))
    patch_width = max(24, round(dial_size * 0.27))
    patch_height = max(28, round(dial_size * 0.30))
    angle = result.angle_degrees
    radians = math.radians(angle)
    direction = np.asarray((math.sin(radians), -math.cos(radians)), dtype=np.float64)
    center = np.asarray(result.center, dtype=np.float64)
    signed_angle = (angle + 180.0) % 360.0 - 180.0
    crops: list[np.ndarray] = []
    for radius_factor in (0.98, 1.0, 1.02):
        crop_center = center + direction * dial_size * radius_factor
        patch = cv2.getRectSubPix(
            image,
            (patch_width, patch_height),
            (float(crop_center[0]), float(crop_center[1])),
        )
        for rotation_delta in range(-15, 16, 5):
            rotation = signed_angle + rotation_delta
            transform = cv2.getRotationMatrix2D(
                (patch_width / 2.0, patch_height / 2.0), rotation, 1.0
            )
            rotated = cv2.warpAffine(
                patch,
                transform,
                (patch_width, patch_height),
                borderMode=cv2.BORDER_REPLICATE,
            )
            crops.append(
                cv2.resize(
                    rotated,
                    None,
                    fx=4.0,
                    fy=4.0,
                    interpolation=cv2.INTER_CUBIC,
                )
            )
    recognition = reader.recognize_isolated_text_lines(crops)
    texts = getattr(recognition, "txts", None)
    scores = getattr(recognition, "scores", None)
    if texts is None or scores is None:
        raise ValueError("Pointer-aligned discrete label OCR returned no result")
    return select_consensus_discrete_label(
        [(str(text), float(score)) for text, score in zip(texts, scores, strict=True)]
    )


def _interpret_recovered_result(
    reader: EthzPaddleGaugeReader,
    result: GaugeResult,
    visible_text: str,
) -> GaugeResult:
    if reader.reading_interpreter is None:
        return result
    return reader.reading_interpreter.interpret(result, visible_text)


def _empty_pointer_result(candidate: DialCandidate) -> GaugeResult:
    return GaugeResult(
        detected=True,
        bbox=candidate.bbox,
        detection_confidence=candidate.confidence,
        pointer_found=False,
        center=None,
        pointer_tip=None,
        angle_degrees=None,
        sweep_fraction=None,
        reading=None,
        unit=None,
        confidence=candidate.confidence,
        center_method=None,
        timings=StageTimings(0.0, 0.0, 0.0),
    )


def normalize_counter_display(text: str) -> str | None:
    compact = "".join(text.upper().split())
    if not COUNTER_TOKEN_PATTERN.fullmatch(compact):
        return None
    return compact.translate(COUNTER_CHARACTER_MAP)


def extract_counter_candidates(
    ocr_result: object, image: np.ndarray | None = None
) -> tuple[CounterOCRCandidate, ...]:
    boxes = getattr(ocr_result, "boxes", None)
    texts = getattr(ocr_result, "txts", None)
    scores = getattr(ocr_result, "scores", None)
    if boxes is None or texts is None or scores is None:
        return ()
    candidates: list[CounterOCRCandidate] = []
    for box, raw_text, raw_score in zip(boxes, texts, scores, strict=True):
        confidence = float(raw_score)
        normalized = normalize_counter_display(str(raw_text))
        if confidence < 0.5 or normalized is None:
            continue
        box_array = np.asarray(box, dtype=np.float64)
        center = np.mean(box_array, axis=0)
        background_luma = _box_luma(image, box_array) if image is not None else None
        if background_luma is not None and background_luma > 120.0:
            continue
        candidates.append(
            CounterOCRCandidate(
                raw_text=str(raw_text).strip(),
                normalized_display=normalized,
                value=int(normalized),
                confidence=confidence,
                center=(float(center[0]), float(center[1])),
                background_luma=background_luma,
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.center[0]))


def _select_similar_dial_row(
    candidates: tuple[DialCandidate, ...],
) -> tuple[DialCandidate, ...]:
    """Keep the similarly sized horizontal row anchored by the best detection."""
    if not candidates:
        return ()
    anchor = candidates[0]
    ax1, ay1, ax2, ay2 = anchor.bbox
    anchor_width = ax2 - ax1
    anchor_height = ay2 - ay1
    anchor_center_y = (ay1 + ay2) / 2.0
    selected = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.bbox
        width = x2 - x1
        height = y2 - y1
        if not 0.65 <= width / anchor_width <= 1.45:
            continue
        if not 0.65 <= height / anchor_height <= 1.45:
            continue
        if abs((y1 + y2) / 2.0 - anchor_center_y) > 0.55 * anchor_height:
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: item.bbox[0]))


def _extract_counters_for_dials(
    reader: EthzPaddleGaugeReader,
    image: np.ndarray,
    dials: tuple[DialCandidate, ...],
) -> tuple[CounterOCRCandidate, ...]:
    """Read one dark mechanical counter window from the upper part of each dial."""
    selected: list[CounterOCRCandidate] = []
    for dial in dials:
        x1, y1, x2, y2 = dial.bbox
        dial_height = y2 - y1
        dial_width = x2 - x1
        crop_x1 = x1 + round(dial_width * 0.15)
        crop_x2 = x2 - round(dial_width * 0.15)
        crop = image[y1 : y1 + round(dial_height * 0.35), crop_x1:crop_x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        multiscale_candidates: list[tuple[CounterOCRCandidate, float]] = []
        for scale in (1.0, 3.0):
            enlarged = cv2.resize(
                enhanced,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            multiscale_candidates.extend(
                (candidate, scale)
                for candidate in extract_counter_candidates(reader.ocr(enlarged), None)
            )
        if not multiscale_candidates:
            continue
        three_digit = [
            (candidate, scale)
            for candidate, scale in multiscale_candidates
            if len(candidate.normalized_display) == 3
        ]
        pool = three_digit or multiscale_candidates
        crop_height, crop_width = enhanced.shape[:2]
        counter, selected_scale = min(
            pool,
            key=lambda item: (
                item[0].center[1] / item[1] / crop_height,
                abs(item[0].center[0] / item[1] - crop_width / 2.0) / crop_width,
                -item[0].confidence,
            ),
        )
        selected.append(
            replace(
                counter,
                center=(
                    crop_x1 + counter.center[0] / selected_scale,
                    y1 + counter.center[1] / selected_scale,
                ),
            )
        )
    return tuple(selected)


def _box_luma(image: np.ndarray, box: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x1, y1 = np.floor(np.min(box, axis=0)).astype(int)
    x2, y2 = np.ceil(np.max(box, axis=0)).astype(int)
    padding = 3
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(gray.shape[1], x2 + padding)
    y2 = min(gray.shape[0], y2 + padding)
    roi = gray[y1:y2, x1:x2]
    return float(np.asarray(roi, dtype=np.float64).mean()) if roi.size else 255.0


def _select_instance_dials(
    candidates: tuple[DialCandidate, ...], instance_count: int
) -> tuple[DialCandidate, ...]:
    selected = tuple(sorted(candidates[:instance_count], key=lambda item: item.bbox[0]))
    return selected


def _padded_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = round((x2 - x1) * 0.10)
    pad_y = round((y2 - y1) * 0.10)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


def _offset_result(result: GaugeResult, offset_x: int, offset_y: int) -> GaugeResult:
    bbox = result.bbox
    center = result.center
    pointer_tip = result.pointer_tip
    return replace(
        result,
        bbox=(
            bbox[0] + offset_x,
            bbox[1] + offset_y,
            bbox[2] + offset_x,
            bbox[3] + offset_y,
        )
        if bbox is not None
        else None,
        center=(center[0] + offset_x, center[1] + offset_y)
        if center is not None
        else None,
        pointer_tip=(pointer_tip[0] + offset_x, pointer_tip[1] + offset_y)
        if pointer_tip is not None
        else None,
    )


def _result_rank(result: GaugeResult) -> tuple[int, float]:
    if result.reading is not None:
        level = 4
    elif result.reading_candidates:
        level = 3
    elif result.pointer_found:
        level = 2
    elif result.detected:
        level = 1
    else:
        level = 0
    return (level, result.confidence or 0.0)


def _channel_analyses(
    metadata: InstrumentTypeMetadata,
    instances: tuple[str, ...],
    pointer_results: tuple[GaugeResult, ...],
    counters: tuple[CounterOCRCandidate, ...],
) -> tuple[ChannelAnalysis, ...]:
    analyses: list[ChannelAnalysis] = []
    pointer_channel = _single_pointer_channel(metadata)
    mechanical_channels = tuple(
        channel
        for channel in metadata.readout_channels
        if channel.display_type == "mechanical_counter"
    )
    for index, instance_id in enumerate(instances):
        pointer_result = (
            pointer_results[index] if index < len(pointer_results) else None
        )
        if pointer_channel is not None:
            analyses.append(
                _pointer_analysis(instance_id, pointer_channel, pointer_result)
            )
        counter = counters[index] if index < len(counters) else None
        for channel in mechanical_channels:
            analyses.append(_counter_analysis(instance_id, channel, counter))
        covered = {
            analysis.channel_id
            for analysis in analyses
            if analysis.instance_id == instance_id
        }
        for channel in metadata.readout_channels:
            if channel.channel_id not in covered:
                analyses.append(
                    ChannelAnalysis(
                        instance_id=instance_id,
                        channel_id=channel.channel_id,
                        value=None,
                        unit=channel.unit,
                        status="not_recognized",
                        method=None,
                        confidence=None,
                        note_zh="本次自动链路未获得该显示通道的可靠结果。",
                    )
                )
    return tuple(analyses)


def _single_pointer_channel(
    metadata: InstrumentTypeMetadata,
) -> ReadoutChannel | None:
    channels = tuple(
        channel
        for channel in metadata.readout_channels
        if channel.display_type in POINTER_DISPLAY_TYPES
    )
    return channels[0] if len(channels) == 1 else None


def _pointer_analysis(
    instance_id: str,
    channel: ReadoutChannel,
    result: GaugeResult | None,
) -> ChannelAnalysis:
    if result is None:
        return ChannelAnalysis(
            instance_id=instance_id,
            channel_id=channel.channel_id,
            value=None,
            unit=channel.unit,
            status="not_recognized",
            method=None,
            confidence=None,
        )
    if result.reading is not None:
        status = "recognized"
    elif result.reading_candidates:
        status = "ambiguous"
    else:
        status = "not_recognized"
    return ChannelAnalysis(
        instance_id=instance_id,
        channel_id=channel.channel_id,
        value=result.reading,
        unit=channel.unit,
        status=status,
        method=result.interpretation_method or result.center_method,
        confidence=result.confidence,
        candidates=result.reading_candidates,
        note_zh=result.failure_reason,
    )


def _counter_analysis(
    instance_id: str,
    channel: ReadoutChannel,
    candidate: CounterOCRCandidate | None,
) -> ChannelAnalysis:
    if candidate is None:
        return ChannelAnalysis(
            instance_id=instance_id,
            channel_id=channel.channel_id,
            value=None,
            unit=channel.unit,
            status="not_recognized",
            method="ocr:mechanical_counter",
            confidence=None,
        )
    normalized = candidate.raw_text != candidate.normalized_display
    return ChannelAnalysis(
        instance_id=instance_id,
        channel_id=channel.channel_id,
        value=candidate.value,
        unit=channel.unit,
        status="recognized",
        method=(
            "ocr:mechanical_counter+confusable_character_normalization"
            if normalized
            else "ocr:mechanical_counter"
        ),
        confidence=candidate.confidence,
        raw_display=candidate.normalized_display,
        raw_ocr_text=candidate.raw_text,
    )
