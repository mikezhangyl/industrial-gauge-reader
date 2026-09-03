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

from src.ethz_vision_reader import Ellipse
from src.gauge_reader import GaugeResult, StageTimings, angle_from_points
from src.instrument_metadata import (
    InstrumentMetadataCatalog,
    InstrumentTypeMetadata,
    ReadoutChannel,
)
from src.processing_stages import (
    ProcessingStageWriter,
    draw_dial_candidates,
    draw_ellipse,
)
from src.rapidocr_reader import (
    DialCandidate,
    EthzPaddleGaugeReader,
    ellipse_fits_crop,
    ellipse_rectification,
    map_bbox_between_images,
    pointer_center_and_tip,
    rectify_dial,
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
MULTI_INSTANCE_RECTANGULAR_TYPES = frozenset(
    {
        "rectangular_panel_voltmeter",
        "dual_scale_rectangular_panel_meter",
    }
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


@dataclass(frozen=True)
class HiddenPivotEstimate:
    """Robust common point of radial scale marks for an occluded meter pivot."""

    center: np.ndarray
    inlier_indices: tuple[int, ...]
    median_residual: float
    angular_spread_degrees: float
    pointer_line: np.ndarray | None = None
    right_scale_line: np.ndarray | None = None
    middle_scale_line: np.ndarray | None = None
    visible_adjustment_center: np.ndarray | None = None
    axis_offset_degrees: float | None = None
    projection_mode: str = "scale_plane_consensus"
    cross_plane_residual: float | None = None
    rectification_transform: np.ndarray | None = None
    outer_ellipse: Ellipse | None = None
    outer_ellipse_edge_support: float | None = None


class MetadataAwareImageAnalyzer:
    """Return every metadata-declared channel without mixing in human answers."""

    def __init__(
        self,
        reader: EthzPaddleGaugeReader,
        catalog: InstrumentMetadataCatalog,
    ):
        self.reader = reader
        self.catalog = catalog

    def analyze(
        self,
        image_path: Path,
        *,
        detail_image_path: Path | None = None,
        stage_writer: ProcessingStageWriter | None = None,
    ) -> InstrumentImageAnalysis:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot decode image: {image_path}")
        detail_image = (
            image
            if detail_image_path is None or detail_image_path == image_path
            else cv2.imread(str(detail_image_path), cv2.IMREAD_COLOR)
        )
        if detail_image is None:
            raise FileNotFoundError(f"Cannot decode detail image: {detail_image_path}")
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
        matches = self.catalog.find(visible_text)
        if len(matches) != 1:
            detect_generic_candidates = getattr(
                self.reader,
                "detect_all_dial_candidates",
                self.reader.detect_dial_candidates,
            )
            generic_candidates = _select_similar_dial_row(
                tuple(
                    candidate
                    for candidate in detect_generic_candidates(image_path)
                    if _bbox_area_fraction(candidate.bbox, image.shape[:2]) >= 0.01
                )
            )
            if len(generic_candidates) > 1:
                generic_results = self._read_dials(
                    image,
                    detail_image,
                    generic_candidates,
                    visible_text,
                    image_path.name,
                    stage_writer,
                )
                generic_results = _recover_generic_rectangular_pointer_results(
                    image,
                    generic_candidates,
                    generic_results,
                    stage_writer,
                )
                return _generic_pointer_analyses(
                    image_sha256,
                    visible_text,
                    generic_results,
                    metadata_match_count=len(matches),
                )
            generic_result = self._read_image(
                image_path,
                visible_text,
                detail_image_path,
                stage_writer,
                "generic-full-image",
            )
            return _generic_pointer_analysis(
                image_sha256,
                visible_text,
                generic_result,
                metadata_match_count=len(matches),
            )

        metadata = matches[0]
        if metadata.type_id in MULTI_INSTANCE_RECTANGULAR_TYPES:
            dial_candidates = self.reader.detect_all_dial_candidates(image_path)
        else:
            dial_candidates = self.reader.detect_dial_candidates(image_path)
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
        elif metadata.type_id in MULTI_INSTANCE_RECTANGULAR_TYPES:
            selected_dials = _select_similar_dial_row(dial_candidates)
            instance_count = max(len(selected_dials), 1)
        else:
            instance_count = len(counters) if mechanical_channels and counters else 1
            selected_dials = _select_instance_dials(dial_candidates, instance_count)
        selected_path_uses_type_specific_geometry = metadata.type_id in {
            "surge_arrester_monitor",
            "shm_d_motor_drive_unit",
            "arrester_discharge_counter",
            *MULTI_INSTANCE_RECTANGULAR_TYPES,
        }
        generic_stage_writer = (
            None if selected_path_uses_type_specific_geometry else stage_writer
        )
        pointer_results = self._read_dials(
            image,
            detail_image,
            selected_dials,
            visible_text,
            image_path.name,
            generic_stage_writer,
        )
        if instance_count == 1:
            full_result = self._read_image(
                image_path,
                visible_text,
                detail_image_path,
                generic_stage_writer,
                "full-image-fallback",
            )
            pointer_results = (
                max(
                    (full_result, *pointer_results),
                    key=_generic_result_rank,
                ),
            )
        pointer_results = _recover_type_specific_pointer_results(
            image,
            metadata,
            selected_dials,
            pointer_results,
            visible_text,
            self.reader,
            stage_writer,
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
        if (
            metadata.type_id == "shm_d_motor_drive_unit"
            and selected_dials
            and pointer_results
        ):
            status_analysis = _shm_mechanism_status_analysis(
                image,
                selected_dials[0],
                pointer_results[0],
                stage_writer,
            )
            if status_analysis is not None:
                channels = tuple(
                    status_analysis
                    if channel.channel_id == "mechanism_status"
                    else channel
                    for channel in channels
                )
        return InstrumentImageAnalysis(
            image_sha256=image_sha256,
            instrument_type_id=metadata.type_id,
            visible_text=visible_text,
            instances=instances,
            pointer_results=pointer_results,
            channels=channels,
        )

    def _read_image(
        self,
        image_path: Path,
        visible_text: str,
        detail_image_path: Path | None,
        stage_writer: ProcessingStageWriter | None,
        stage_group: str,
    ) -> GaugeResult:
        if stage_writer is None:
            if detail_image_path is None:
                return self.reader.read(
                    image_path,
                    visible_text_context=visible_text,
                )
            return self.reader.read(
                image_path,
                visible_text_context=visible_text,
                detail_image_path=detail_image_path,
            )
        if detail_image_path is None:
            return self.reader.read(
                image_path,
                visible_text_context=visible_text,
                stage_writer=stage_writer,
                stage_group=stage_group,
            )
        return self.reader.read(
            image_path,
            visible_text_context=visible_text,
            detail_image_path=detail_image_path,
            stage_writer=stage_writer,
            stage_group=stage_group,
        )

    def _read_dials(
        self,
        image: np.ndarray,
        detail_image: np.ndarray,
        candidates: tuple[DialCandidate, ...],
        visible_text: str,
        source_name: str,
        stage_writer: ProcessingStageWriter | None,
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
                detail_bbox = map_bbox_between_images(
                    crop_bbox,
                    source_shape=image.shape[:2],
                    target_shape=detail_image.shape[:2],
                )
                detail_x1, detail_y1, detail_x2, detail_y2 = detail_bbox
                crop_path = temporary_path / f"{Path(source_name).stem}-{index}.png"
                detail_crop_path = temporary_path / (
                    f"{Path(source_name).stem}-{index}-detail.png"
                )
                if not cv2.imwrite(str(crop_path), image[y1:y2, x1:x2]):
                    raise RuntimeError(
                        f"Failed to create temporary dial crop: {crop_path}"
                    )
                if self.reader.profile.use_high_resolution_detail:
                    if not cv2.imwrite(
                        str(detail_crop_path),
                        detail_image[detail_y1:detail_y2, detail_x1:detail_x2],
                    ):
                        raise RuntimeError(
                            "Failed to create temporary detail dial crop: "
                            f"{detail_crop_path}"
                        )
                    if stage_writer is None:
                        local_result = self.reader.read(
                            crop_path,
                            visible_text_context=visible_text,
                            detail_image_path=detail_crop_path,
                        )
                    else:
                        local_result = self.reader.read(
                            crop_path,
                            visible_text_context=visible_text,
                            detail_image_path=detail_crop_path,
                            stage_writer=stage_writer,
                            stage_group=f"dial-{index + 1}",
                        )
                else:
                    if stage_writer is None:
                        local_result = self.reader.read(
                            crop_path,
                            visible_text_context=visible_text,
                        )
                    else:
                        local_result = self.reader.read(
                            crop_path,
                            visible_text_context=visible_text,
                            stage_writer=stage_writer,
                            stage_group=f"dial-{index + 1}",
                        )
                results.append(_offset_result(local_result, x1, y1))
        return tuple(
            sorted(
                results,
                key=lambda result: result.bbox[0] if result.bbox is not None else 0,
            )
        )


def _generic_pointer_analysis(
    image_sha256: str,
    visible_text: str,
    result: GaugeResult,
    *,
    metadata_match_count: int,
) -> InstrumentImageAnalysis:
    """Keep the generic visual reading when optional type metadata is unavailable."""
    return _generic_pointer_analyses(
        image_sha256,
        visible_text,
        (result,),
        metadata_match_count=metadata_match_count,
    )


def _generic_pointer_analyses(
    image_sha256: str,
    visible_text: str,
    results: tuple[GaugeResult, ...],
    *,
    metadata_match_count: int,
) -> InstrumentImageAnalysis:
    """Keep every detected generic dial instead of discarding all but one."""
    if not results:
        return InstrumentImageAnalysis(
            image_sha256=image_sha256,
            instrument_type_id=None,
            visible_text=visible_text,
            instances=(),
            pointer_results=(),
            channels=(),
            failure_reason="Generic gauge detection returned no dial results",
        )
    if len(results) == 1 and not results[0].detected:
        result = results[0]
        return InstrumentImageAnalysis(
            image_sha256=image_sha256,
            instrument_type_id=None,
            visible_text=visible_text,
            instances=(),
            pointer_results=results,
            channels=(),
            failure_reason=result.failure_reason or "Generic gauge detection failed",
        )

    instrument_type_ids = {
        result.instrument_type_id
        for result in results
        if result.instrument_type_id is not None
    }
    instrument_type_id = (
        next(iter(instrument_type_ids)) if len(instrument_type_ids) == 1 else None
    )
    instances = tuple(f"instance_{index + 1}" for index in range(len(results)))
    channels: list[ChannelAnalysis] = []
    for instance_id, result in zip(instances, results, strict=True):
        if result.reading is not None:
            status = "recognized"
        elif result.reading_candidates:
            status = "ambiguous"
        else:
            status = "not_recognized"
        if result.instrument_type_id is not None:
            metadata_note = "整图 OCR 未命中；表盘 OCR 已匹配类型 metadata。"
        elif metadata_match_count == 0:
            metadata_note = "未匹配类型 metadata；已保留通用指针读数。"
        else:
            metadata_note = (
                "匹配到多个类型 metadata；已保留通用指针读数，未猜测具体类型。"
            )
        channels.append(
            ChannelAnalysis(
                instance_id=instance_id,
                channel_id=result.readout_channel_id or "pointer_reading",
                value=result.reading,
                unit=result.unit or "unknown",
                status=status,
                method=result.interpretation_method or "generic:analog_pointer",
                confidence=result.confidence,
                raw_display=(
                    str(result.raw_reading)
                    if result.raw_reading is not None
                    else None
                ),
                candidates=tuple(result.reading_candidates),
                note_zh=metadata_note,
            )
        )
    return InstrumentImageAnalysis(
        image_sha256=image_sha256,
        instrument_type_id=instrument_type_id,
        visible_text=visible_text,
        instances=instances,
        pointer_results=results,
        channels=tuple(channels),
    )


def _recover_generic_rectangular_pointer_results(
    image: np.ndarray,
    candidates: tuple[DialCandidate, ...],
    results: tuple[GaugeResult, ...],
    stage_writer: ProcessingStageWriter | None,
) -> tuple[GaugeResult, ...]:
    """Recover pivots hidden below wide half-dial windows."""
    recovered = list(results)
    for index, (candidate, existing) in enumerate(
        zip(candidates, recovered, strict=False)
    ):
        candidate_x1, candidate_y1, candidate_x2, candidate_y2 = candidate.bbox
        candidate_area = (candidate_x2 - candidate_x1) * (
            candidate_y2 - candidate_y1
        )
        existing_area = 0
        if existing.bbox is not None:
            x1, y1, x2, y2 = existing.bbox
            existing_area = (x2 - x1) * (y2 - y1)
        nested_detection_is_too_small = existing_area < candidate_area * 0.35
        if existing.pointer_found and not nested_detection_is_too_small:
            continue
        try:
            center, tip, angle, _ = _detect_rectangular_meter_pointer(
                image, candidate
            )
        except ValueError:
            continue
        recovered[index] = replace(
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
            center_method="generic:rectangular-meter-hub+line",
            failure_reason="Scale interpretation requires instrument metadata",
            raw_reading=None,
            instrument_type_id=None,
            readout_channel_id=None,
            interpretation_method=None,
            reading_candidates=(),
        )
        _write_selected_pointer_stages(
            stage_writer,
            image,
            candidate,
            center,
            tip,
            group=f"dial-{index + 1}-selected-rectangular-meter",
            method="rectangular_meter_hub_and_extended_line",
        )
    return tuple(recovered)


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
    """Select the visible bottom-centre circular reference on a panel meter."""
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


def select_meter_adjustment_reference(
    circles: np.ndarray,
    *,
    crop_shape: tuple[int, int],
    face_center_x: float,
) -> np.ndarray:
    """Select the zero-adjustment screw without treating it as the pivot."""

    height, width = crop_shape
    candidates: list[tuple[float, np.ndarray]] = []
    for raw_circle in np.asarray(circles, dtype=np.float64).reshape(-1, 3):
        x, y, radius = raw_circle
        if not 0.43 * width <= x <= 0.57 * width:
            continue
        if not 0.60 * height <= y <= 0.72 * height:
            continue
        if not 0.015 * width <= radius <= 0.05 * width:
            continue
        score = (
            abs(x / width - 0.50)
            + abs(y / height - 0.67)
            + 0.35 * abs(radius / width - 0.03)
            + 0.05 * abs(x - face_center_x) / width
        )
        candidates.append((score, np.asarray((x, y), dtype=np.float64)))
    if not candidates:
        raise ValueError("Meter zero-adjustment screw was not found")
    return min(candidates, key=lambda item: item[0])[1]


def infer_three_line_hidden_meter_pivot(
    pointer_line: np.ndarray,
    right_scale_line: np.ndarray,
    middle_scale_line: np.ndarray,
    *,
    crop_shape: tuple[int, int],
) -> HiddenPivotEstimate:
    """Fit the projected pivot shared by pointer, end tick, and centre tick.

    The middle scale line is expected to pass through the visible zero-adjustment
    screw.  Its deviation from image vertical records the camera-view offset; it
    must not be silently replaced with an assumed vertical line.
    """

    height, width = crop_shape
    raw_lines = np.asarray(
        (pointer_line, right_scale_line, middle_scale_line),
        dtype=np.float64,
    ).reshape(3, 4)
    normals: list[np.ndarray] = []
    rhs: list[float] = []
    for raw_line in raw_lines:
        start = raw_line[:2]
        vector = raw_line[2:] - start
        length = float(np.linalg.norm(vector))
        if length < 1.0:
            raise ValueError("Hidden-pivot calibration line is too short")
        direction = vector / length
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        normals.append(normal)
        rhs.append(float(normal @ start))

    center, _, _, _ = np.linalg.lstsq(
        np.vstack(normals),
        np.asarray(rhs),
        rcond=None,
    )
    residuals = np.asarray(
        [
            abs(float(normal @ center - intercept))
            for normal, intercept in zip(normals, rhs, strict=True)
        ]
    )
    if not (
        0.05 * width <= center[0] <= 0.95 * width
        and 0.55 * height <= center[1] <= 1.08 * height
    ):
        raise ValueError("Three calibration lines meet outside the hidden-pivot region")
    if float(np.max(residuals)) > 0.05 * width:
        raise ValueError("Three calibration lines do not share a stable pivot")

    middle = raw_lines[2]
    middle_points = middle.reshape(2, 2)
    scale_point = middle_points[int(np.argmin(middle_points[:, 1]))]
    axis_vector = center - scale_point
    axis_offset = math.degrees(
        math.atan2(float(axis_vector[0]), float(axis_vector[1]))
    )
    return HiddenPivotEstimate(
        center=center.astype(np.float64),
        inlier_indices=(),
        median_residual=float(np.median(residuals)),
        angular_spread_degrees=0.0,
        pointer_line=raw_lines[0],
        right_scale_line=raw_lines[1],
        middle_scale_line=raw_lines[2],
        axis_offset_degrees=axis_offset,
        projection_mode="three_line_coplanar",
        cross_plane_residual=float(np.median(residuals)),
    )


def infer_two_line_hidden_meter_pivot(
    pointer_line: np.ndarray,
    middle_scale_point: np.ndarray,
    visible_adjustment_center: np.ndarray,
    *,
    crop_shape: tuple[int, int],
    inlier_indices: tuple[int, ...],
    scale_residual: float,
) -> HiddenPivotEstimate:
    """Intersect the pointer with the centre-tick/adjustment-screw axis."""

    height, width = crop_shape
    pointer_line = np.asarray(pointer_line, dtype=np.float64).reshape(4)
    scale_point = np.asarray(middle_scale_point, dtype=np.float64).reshape(2)
    adjustment_center = np.asarray(
        visible_adjustment_center,
        dtype=np.float64,
    ).reshape(2)
    middle_line = np.concatenate((scale_point, adjustment_center))
    center = _line_intersection(pointer_line, middle_line)
    if center is None:
        raise ValueError("Pointer and meter centre axis are parallel")
    if not (
        0.20 * width <= center[0] <= 0.80 * width
        and 0.62 * height <= center[1] <= 1.02 * height
    ):
        raise ValueError("Two-line pivot falls outside the hidden-pivot region")
    axis_vector = center - scale_point
    axis_offset = math.degrees(
        math.atan2(float(axis_vector[0]), float(axis_vector[1]))
    )
    return HiddenPivotEstimate(
        center=center.astype(np.float64),
        inlier_indices=inlier_indices,
        median_residual=float(scale_residual),
        angular_spread_degrees=0.0,
        pointer_line=pointer_line,
        middle_scale_line=middle_line,
        visible_adjustment_center=adjustment_center,
        axis_offset_degrees=axis_offset,
        projection_mode="two_line_hidden_pivot",
    )


def detect_outer_meter_ellipse(image: np.ndarray) -> Ellipse:
    """Fit the centred outer housing rim used to normalize camera pose."""

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    edge_distance = cv2.distanceTransform(
        (edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    candidates: list[tuple[float, float, int, Ellipse]] = []
    for contour in contours:
        if len(contour) < 80:
            continue
        raw = cv2.fitEllipse(contour)
        ellipse: Ellipse = (
            (float(raw[0][0]), float(raw[0][1])),
            (float(raw[1][0]), float(raw[1][1])),
            float(raw[2]),
        )
        (center_x, center_y), axes, _ = ellipse
        minor, major = sorted(axes)
        if major < 0.70 * min(width, height):
            continue
        if minor < 0.65 * min(width, height):
            continue
        if major > 1.15 * max(width, height) or minor / major < 0.60:
            continue
        normalized_center_error = float(
            np.linalg.norm((center_x / width - 0.5, center_y / height - 0.5))
        )
        if normalized_center_error > 0.12:
            continue
        if not ellipse_fits_crop(
            ellipse,
            (height, width),
            tolerance_fraction=0.10,
        ):
            continue
        edge_support, visible_arc_fraction = _ellipse_edge_support(
            ellipse,
            edge_distance=edge_distance,
            image_shape=(height, width),
        )
        if edge_support < 0.80 or visible_arc_fraction < 0.50:
            continue
        candidates.append((major, minor, len(contour), ellipse))
    if not candidates:
        raise ValueError("Outer meter ellipse was not found")
    return max(candidates, key=lambda item: item[:3])[-1]


def _ellipse_edge_support(
    ellipse: Ellipse,
    *,
    edge_distance: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    """Measure how much of an in-frame ellipse follows observed image edges."""

    height, width = image_shape
    (center_x, center_y), (axis_a, axis_b), angle = ellipse
    perimeter = np.asarray(
        cv2.ellipse2Poly(
            (round(center_x), round(center_y)),
            (max(1, round(axis_a / 2.0)), max(1, round(axis_b / 2.0))),
            round(angle),
            0,
            359,
            2,
        ),
        dtype=np.int32,
    )
    in_bounds = (
        (perimeter[:, 0] >= 0)
        & (perimeter[:, 0] < width)
        & (perimeter[:, 1] >= 0)
        & (perimeter[:, 1] < height)
    )
    visible_arc_fraction = float(np.mean(in_bounds))
    if not np.any(in_bounds):
        return 0.0, visible_arc_fraction
    visible_points = perimeter[in_bounds]
    distances = edge_distance[visible_points[:, 1], visible_points[:, 0]]
    tolerance = max(2.0, 0.008 * min(width, height))
    return float(np.mean(distances <= tolerance)), visible_arc_fraction


def measure_outer_meter_ellipse_quality(
    image: np.ndarray,
    ellipse: Ellipse,
) -> tuple[float, float]:
    """Return edge support and visible-arc fractions for an accepted rim."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edge_distance = cv2.distanceTransform(
        (edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    return _ellipse_edge_support(
        ellipse,
        edge_distance=edge_distance,
        image_shape=image.shape[:2],
    )


def _transform_affine_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    raw_transform = np.asarray(transform, dtype=np.float64).reshape(2, 3)
    raw_point = np.asarray(point, dtype=np.float64).reshape(2)
    return raw_transform[:, :2] @ raw_point + raw_transform[:, 2]


def select_rectified_middle_scale_mark(
    tick_lines: np.ndarray,
    *,
    visible_adjustment_center: np.ndarray,
    crop_shape: tuple[int, int],
    rectification_transform: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Select the scale midpoint aligned with the screw after pose correction."""

    height, width = crop_shape
    adjustment_center = np.asarray(
        visible_adjustment_center,
        dtype=np.float64,
    ).reshape(2)
    rectified_adjustment = _transform_affine_point(
        rectification_transform,
        adjustment_center,
    )
    corners = np.asarray(
        ((0.0, 0.0), (width, 0.0), (0.0, height), (width, height)),
        dtype=np.float64,
    )
    rectified_corners = np.asarray(
        [
            _transform_affine_point(rectification_transform, corner)
            for corner in corners
        ]
    )
    rectified_width = float(np.ptp(rectified_corners[:, 0]))
    candidates: list[tuple[tuple[float, float, float], int, np.ndarray, np.ndarray]] = []
    for index, raw_line in enumerate(
        np.asarray(tick_lines, dtype=np.float64).reshape(-1, 4)
    ):
        start = raw_line[:2]
        end = raw_line[2:]
        vector = end - start
        length = float(np.linalg.norm(vector))
        midpoint = (start + end) / 2.0
        angle = math.degrees(math.atan2(vector[1], vector[0])) % 180.0
        if not max(6.0, 0.015 * width) <= length <= 0.09 * width:
            continue
        if not 0.40 * height <= midpoint[1] <= 0.52 * height:
            continue
        if not 0.25 * width <= midpoint[0] <= 0.75 * width:
            continue
        if not 25.0 <= angle <= 155.0:
            continue
        rectified_midpoint = _transform_affine_point(
            rectification_transform,
            midpoint,
        )
        horizontal_error = abs(
            float(rectified_midpoint[0] - rectified_adjustment[0])
        )
        score = (
            horizontal_error / max(rectified_width, 1.0),
            abs(float(midpoint[1] / height) - 0.46),
            -length / width,
        )
        candidates.append((score, index, raw_line, midpoint))
    if not candidates:
        raise ValueError("Middle scale mark was not found")
    score, index, raw_line, midpoint = min(candidates, key=lambda item: item[0])
    if score[0] > 0.08:
        raise ValueError("Middle scale mark does not align with adjustment screw")
    return index, raw_line, midpoint


def resolve_hidden_meter_projection(
    scale_plane_estimate: HiddenPivotEstimate,
    three_line_estimate: HiddenPivotEstimate,
    *,
    crop_shape: tuple[int, int],
) -> HiddenPivotEstimate:
    """Keep cross-plane calibration only when its image evidence is coplanar.

    The pointer, printed scale and zero-adjustment screw can sit at different
    physical depths.  Under an oblique view their projected lines need not share
    one image point.  A low three-line residual is useful evidence for a nearly
    frontal view; otherwise the scale marks, which are coplanar with each other,
    own the reading pivot and the pointer tip is mapped onto that scale plane.
    """

    _, width = crop_shape
    cross_plane_residual = float(three_line_estimate.median_residual)
    coplanar_limit = max(4.0, 0.01 * width)
    if cross_plane_residual <= coplanar_limit:
        return replace(
            three_line_estimate,
            projection_mode="three_line_coplanar",
            cross_plane_residual=cross_plane_residual,
        )
    return replace(
        three_line_estimate,
        center=np.asarray(scale_plane_estimate.center, dtype=np.float64),
        inlier_indices=scale_plane_estimate.inlier_indices,
        median_residual=scale_plane_estimate.median_residual,
        angular_spread_degrees=scale_plane_estimate.angular_spread_degrees,
        axis_offset_degrees=None,
        projection_mode="scale_plane_parallax_fallback",
        cross_plane_residual=cross_plane_residual,
    )


def _line_distance(line: np.ndarray, point: np.ndarray) -> float:
    raw_line = np.asarray(line, dtype=np.float64).reshape(4)
    start = raw_line[:2]
    vector = raw_line[2:] - start
    return abs(
        float(vector[0] * (start[1] - point[1]))
        - float(vector[1] * (start[0] - point[0]))
    ) / max(float(np.linalg.norm(vector)), 1e-9)


def _line_intersection(
    first_line: np.ndarray,
    second_line: np.ndarray,
) -> np.ndarray | None:
    first = np.asarray(first_line, dtype=np.float64).reshape(4)
    second = np.asarray(second_line, dtype=np.float64).reshape(4)
    first_start = first[:2]
    second_start = second[:2]
    matrix = np.column_stack(
        (first[2:] - first_start, -(second[2:] - second_start))
    )
    try:
        factors = np.linalg.solve(matrix, second_start - first_start)
    except np.linalg.LinAlgError:
        return None
    return first_start + factors[0] * (first[2:] - first_start)


def refine_hidden_meter_pivot_from_three_lines(
    rough_estimate: HiddenPivotEstimate,
    tick_lines: np.ndarray,
    *,
    pointer_line: np.ndarray,
    visible_adjustment_center: np.ndarray,
    crop_shape: tuple[int, int],
) -> HiddenPivotEstimate:
    """Select the end and middle scale rays, then cross-check all three rays."""

    height, width = crop_shape
    pointer_line = np.asarray(pointer_line, dtype=np.float64).reshape(4)
    adjustment_center = np.asarray(
        visible_adjustment_center,
        dtype=np.float64,
    ).reshape(2)
    pointer_points = pointer_line.reshape(2, 2)
    pointer_tip = pointer_points[
        int(
            np.argmax(
                np.linalg.norm(
                    pointer_points - rough_estimate.center,
                    axis=1,
                )
            )
        )
    ]

    candidates: list[tuple[int, np.ndarray, np.ndarray, float, float]] = []
    for index, raw_line in enumerate(
        np.asarray(tick_lines, dtype=np.float64).reshape(-1, 4)
    ):
        start = raw_line[:2]
        end = raw_line[2:]
        vector = end - start
        length = float(np.linalg.norm(vector))
        midpoint = (start + end) / 2.0
        angle = math.degrees(math.atan2(vector[1], vector[0])) % 180.0
        if not max(6.0, 0.015 * width) <= length <= 0.09 * width:
            continue
        if not 0.40 * height <= midpoint[1] <= 0.58 * height:
            continue
        if not 0.08 * width <= midpoint[0] <= 0.92 * width:
            continue
        if not 25.0 <= angle <= 155.0:
            continue
        candidates.append((index, raw_line, midpoint, angle, length))
    if len(candidates) < 2:
        raise ValueError("Too few scale marks for three-line hidden-pivot calibration")

    right_candidates: list[
        tuple[tuple[float, float, float], tuple[int, np.ndarray, np.ndarray, float, float], np.ndarray]
    ] = []
    for candidate in candidates:
        _, raw_line, midpoint, angle, length = candidate
        if midpoint[0] < rough_estimate.center[0] + 0.15 * width:
            continue
        if angle < 105.0:
            continue
        if _line_distance(raw_line, rough_estimate.center) > 0.08 * width:
            continue
        intersection = _line_intersection(pointer_line, raw_line)
        if intersection is None:
            continue
        if abs(float(intersection[0] - rough_estimate.center[0])) > 0.25 * width:
            continue
        if not 0.58 * height <= intersection[1] <= 1.05 * height:
            continue
        pointer_radius = float(np.linalg.norm(pointer_tip - intersection))
        scale_radius = float(np.linalg.norm(midpoint - intersection))
        radius_mismatch = abs(pointer_radius - scale_radius) / width
        if radius_mismatch > 0.12:
            continue
        score = (
            -float(midpoint[0]) / width,
            radius_mismatch,
            _line_distance(raw_line, rough_estimate.center) / width,
        )
        right_candidates.append((score, candidate, intersection))
    if not right_candidates:
        raise ValueError("Right-end scale ray was not found")
    _, right_candidate, provisional_center = min(
        right_candidates,
        key=lambda item: item[0],
    )
    right_index, right_line, right_midpoint, _, _ = right_candidate

    pointer_angle = angle_from_points(provisional_center, pointer_tip)
    right_angle = angle_from_points(provisional_center, right_midpoint)
    sweep = (right_angle - pointer_angle) % 360.0
    direction = 1.0
    if sweep > 180.0:
        sweep = 360.0 - sweep
        direction = -1.0
    if not 50.0 <= sweep <= 150.0:
        raise ValueError("Pointer and right scale ray do not define a plausible arc")

    middle_candidates: list[
        tuple[tuple[float, float, float], tuple[int, np.ndarray, np.ndarray, float, float], float]
    ] = []
    endpoint_radius = float(
        np.mean(
            (
                np.linalg.norm(pointer_tip - provisional_center),
                np.linalg.norm(right_midpoint - provisional_center),
            )
        )
    )
    for candidate in candidates:
        index, raw_line, midpoint, _, _ = candidate
        if index == right_index:
            continue
        candidate_angle = angle_from_points(provisional_center, midpoint)
        position = ((candidate_angle - pointer_angle) * direction) % 360.0
        if position > sweep:
            continue
        fraction = position / sweep
        if not 0.25 <= fraction <= 0.75:
            continue
        radius_mismatch = abs(
            float(np.linalg.norm(midpoint - provisional_center)) - endpoint_radius
        ) / width
        score = (
            abs(fraction - 0.5),
            _line_distance(raw_line, adjustment_center) / width,
            radius_mismatch,
        )
        middle_candidates.append((score, candidate, fraction))
    if not middle_candidates:
        raise ValueError("Middle scale ray was not found")
    _, middle_candidate, _ = min(middle_candidates, key=lambda item: item[0])
    middle_index, _, middle_midpoint, _, _ = middle_candidate
    middle_line = np.concatenate((middle_midpoint, adjustment_center))

    estimate = infer_three_line_hidden_meter_pivot(
        pointer_line,
        right_line,
        middle_line,
        crop_shape=crop_shape,
    )
    if max(
        _line_distance(pointer_line, estimate.center),
        _line_distance(right_line, estimate.center),
        _line_distance(middle_line, estimate.center),
    ) > 0.04 * width:
        raise ValueError("Three-line hidden-pivot cross-check failed")

    calibrated_indices: list[int] = []
    calibrated_angles: list[float] = []
    raw_lines = np.asarray(tick_lines, dtype=np.float64).reshape(-1, 4)
    for index, raw_line, midpoint, _, _ in candidates:
        radius = float(np.linalg.norm(midpoint - estimate.center))
        if not 0.20 * width <= radius <= 0.62 * width:
            continue
        if _line_distance(raw_line, estimate.center) > 0.035 * width:
            continue
        calibrated_indices.append(index)
        calibrated_angles.append(angle_from_points(estimate.center, midpoint))
    for required_index in (right_index, middle_index):
        if required_index not in calibrated_indices:
            calibrated_indices.append(required_index)
            raw_line = raw_lines[required_index]
            midpoint = (raw_line[:2] + raw_line[2:]) / 2.0
            calibrated_angles.append(angle_from_points(estimate.center, midpoint))
    if len(calibrated_indices) < 3:
        raise ValueError("Calibrated pivot has too little scale support")

    unwrapped_angles = np.unwrap(np.radians(calibrated_angles))
    angular_spread = math.degrees(float(np.ptp(unwrapped_angles)))
    return replace(
        estimate,
        inlier_indices=tuple(calibrated_indices),
        angular_spread_degrees=angular_spread,
        visible_adjustment_center=adjustment_center,
    )


def infer_hidden_meter_pivot(
    lines: np.ndarray,
    *,
    crop_shape: tuple[int, int],
    face_center_x: float,
    face_bottom_y: float,
    visible_hub: np.ndarray | None = None,
) -> HiddenPivotEstimate:
    """Infer an off-window pivot from short radial scale marks.

    The visible bottom-centre circle on many panel meters is a zero-adjustment
    screw rather than the movement pivot.  Radial tick extensions still meet at
    the projected pivot, even when that pivot is hidden below the dial window.
    """

    height, width = crop_shape
    raw_lines = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    candidates: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    for index, raw_line in enumerate(raw_lines):
        start = raw_line[:2]
        end = raw_line[2:]
        vector = end - start
        length = float(np.linalg.norm(vector))
        midpoint = (start + end) / 2.0
        if not max(6.0, 0.015 * width) <= length <= 0.09 * width:
            continue
        if not 0.36 * height <= midpoint[1] <= face_bottom_y - 0.10 * height:
            continue
        if not 0.08 * width <= midpoint[0] <= 0.92 * width:
            continue
        direction = vector / max(length, 1e-9)
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        angle = math.degrees(math.atan2(direction[1], direction[0])) % 180.0
        if not 25.0 <= angle <= 155.0:
            continue
        candidates.append((index, midpoint, normal, angle))

    if len(candidates) < 3:
        raise ValueError("Too few radial scale marks to infer a hidden pivot")

    minimum_y = face_bottom_y - 0.02 * height
    if visible_hub is not None:
        minimum_y = max(minimum_y, float(visible_hub[1]) + 0.035 * height)
    maximum_y = min(1.10 * height, face_bottom_y + 0.25 * height)
    minimum_x = face_center_x - 0.30 * width
    maximum_x = face_center_x + 0.30 * width
    residual_limit = max(3.0, 0.018 * width)

    best: (
        tuple[tuple[int, float, float, float, float], np.ndarray, tuple[int, ...]]
        | None
    ) = None
    for left_index in range(len(candidates)):
        _, left_point, left_normal, left_angle = candidates[left_index]
        for right_index in range(left_index + 1, len(candidates)):
            _, right_point, right_normal, right_angle = candidates[right_index]
            angle_separation = abs(left_angle - right_angle)
            angle_separation = min(angle_separation, 180.0 - angle_separation)
            if angle_separation < 8.0:
                continue
            matrix = np.vstack((left_normal, right_normal))
            rhs = np.asarray(
                (float(left_normal @ left_point), float(right_normal @ right_point))
            )
            try:
                intersection = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                continue
            if not (
                minimum_x <= intersection[0] <= maximum_x
                and minimum_y <= intersection[1] <= maximum_y
            ):
                continue
            residuals = np.asarray(
                [abs(float(normal @ (intersection - point))) for _, point, normal, _ in candidates]
            )
            radii = np.asarray(
                [float(np.linalg.norm(point - intersection)) for _, point, _, _ in candidates]
            )
            inlier_positions = tuple(
                int(position)
                for position in np.flatnonzero(
                    (residuals <= residual_limit)
                    & (radii >= 0.22 * width)
                    & (radii <= 0.62 * width)
                )
            )
            if len(inlier_positions) < 3:
                continue
            angles = np.asarray(
                [candidates[position][3] for position in inlier_positions]
            )
            spread = float(np.ptp(angles))
            distinct_angle_bins = len(set((angles // 6.0).astype(int)))
            median_residual = float(np.median(residuals[list(inlier_positions)]))
            inlier_radii = radii[list(inlier_positions)]
            median_radius = float(np.median(inlier_radii))
            radius_mad = float(np.median(np.abs(inlier_radii - median_radius)))
            if median_radius > 0.55 * width or radius_mad > 0.07 * width:
                continue
            score = (
                distinct_angle_bins,
                spread,
                -radius_mad,
                -median_residual,
                -abs(float(intersection[0]) - face_center_x),
            )
            if best is None or score > best[0]:
                best = (score, intersection, inlier_positions)

    if best is None:
        raise ValueError("Radial scale marks have no plausible common pivot")

    _, intersection, inlier_positions = best
    for _ in range(2):
        current_radii = np.asarray(
            [
                float(np.linalg.norm(candidates[position][1] - intersection))
                for position in inlier_positions
            ]
        )
        median_radius = float(np.median(current_radii))
        radius_mad = float(np.median(np.abs(current_radii - median_radius)))
        radius_tolerance = max(
            0.05 * width,
            min(0.09 * width, 2.5 * radius_mad),
        )
        inlier_candidates = [candidates[position] for position in inlier_positions]
        matrix = np.vstack([normal for _, _, normal, _ in inlier_candidates])
        rhs = np.asarray(
            [float(normal @ point) for _, point, normal, _ in inlier_candidates]
        )
        intersection, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        residuals = np.asarray(
            [abs(float(normal @ (intersection - point))) for _, point, normal, _ in candidates]
        )
        radii = np.asarray(
            [float(np.linalg.norm(point - intersection)) for _, point, _, _ in candidates]
        )
        inlier_positions = tuple(
            int(position)
            for position in np.flatnonzero(
                (residuals <= residual_limit)
                & (np.abs(radii - median_radius) <= radius_tolerance)
            )
        )
        if len(inlier_positions) < 3:
            raise ValueError("Hidden pivot refinement lost radial support")

    inlier_candidates = [candidates[position] for position in inlier_positions]
    angles = np.asarray([angle for _, _, _, angle in inlier_candidates])
    angular_spread = float(np.ptp(angles))
    if angular_spread < 20.0:
        raise ValueError("Radial scale marks do not span enough angles")
    residuals = np.asarray(
        [abs(float(normal @ (intersection - point))) for _, point, normal, _ in inlier_candidates]
    )
    return HiddenPivotEstimate(
        center=intersection.astype(np.float64),
        inlier_indices=tuple(candidates[position][0] for position in inlier_positions),
        median_residual=float(np.median(residuals)),
        angular_spread_degrees=angular_spread,
    )


def detect_hidden_meter_pivot(
    image: np.ndarray,
) -> tuple[HiddenPivotEstimate, np.ndarray | None, np.ndarray]:
    """Detect radial scale marks and infer a hidden panel-meter pivot."""

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    face_center_x, face_bottom_y = select_meter_face_geometry(
        face_lines,
        crop_shape=(height, width),
    )
    if face_bottom_y is None:
        face_bottom_y = 0.75 * height

    roi_x1, roi_x2 = round(0.20 * width), round(0.70 * width)
    roi_y1, roi_y2 = round(0.55 * height), round(0.90 * height)
    hub_roi = cv2.GaussianBlur(gray[roi_y1:roi_y2, roi_x1:roi_x2], (3, 3), 1)
    circles = cv2.HoughCircles(
        hub_roi,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=60,
        param2=13,
        minRadius=max(4, round(width * 0.008)),
        maxRadius=max(6, round(width * 0.07)),
    )
    visible_hub: np.ndarray | None = None
    if circles is not None:
        global_circles = circles[0].astype(np.float64)
        global_circles[:, 0] += roi_x1
        global_circles[:, 1] += roi_y1
        try:
            visible_hub = select_meter_adjustment_reference(
                global_circles,
                face_center_x=face_center_x,
                crop_shape=(height, width),
            )
        except ValueError:
            pass

    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    tick_edges = cv2.Canny(cv2.GaussianBlur(enhanced, (3, 3), 0), 40, 120)
    tick_lines = cv2.HoughLinesP(
        tick_edges,
        1,
        np.pi / 720.0,
        threshold=10,
        minLineLength=max(6, round(width * 0.015)),
        maxLineGap=2,
    )
    if tick_lines is None:
        raise ValueError("Rectangular meter scale marks were not found")
    estimate = infer_hidden_meter_pivot(
        tick_lines,
        crop_shape=(height, width),
        face_center_x=face_center_x,
        face_bottom_y=face_bottom_y,
        visible_hub=None,
    )
    return estimate, visible_hub, tick_lines


def detect_three_line_hidden_meter_geometry(
    image: np.ndarray,
) -> tuple[HiddenPivotEstimate, np.ndarray, np.ndarray]:
    """Detect the pointer/end/middle calibration triad on a hidden-pivot meter."""

    height, width = image.shape[:2]
    rough_estimate, adjustment_center, tick_lines = detect_hidden_meter_pivot(image)
    if adjustment_center is None:
        raise ValueError("Zero-adjustment screw is required for three-line calibration")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    pointer_line, _, _ = _select_meter_pointer_segment(
        pointer_lines,
        hub=rough_estimate.center,
        adjustment_center=adjustment_center,
        crop_shape=(height, width),
    )
    three_line_estimate = refine_hidden_meter_pivot_from_three_lines(
        rough_estimate,
        tick_lines,
        pointer_line=pointer_line,
        visible_adjustment_center=adjustment_center,
        crop_shape=(height, width),
    )
    estimate = resolve_hidden_meter_projection(
        rough_estimate,
        three_line_estimate,
        crop_shape=(height, width),
    )
    pointer_points = pointer_line.reshape(2, 2)
    pointer_tip = pointer_points[
        int(np.argmax(np.linalg.norm(pointer_points - estimate.center, axis=1)))
    ]
    return estimate, tick_lines, pointer_tip


def detect_two_line_hidden_meter_geometry(
    image: np.ndarray,
) -> tuple[HiddenPivotEstimate, np.ndarray, np.ndarray]:
    """Detect an occluded pivot from the pointer and the dial centre axis.

    The visible circular part below the scale is the zero-adjustment screw, not
    the pointer pivot.  The outer housing ellipse is used to undo camera pose
    when selecting the printed middle tick.  The hidden pivot is then the
    intersection of the measured pointer line and the line through that middle
    tick and the adjustment screw.
    """

    height, width = image.shape[:2]
    rough_estimate, adjustment_center, tick_lines = detect_hidden_meter_pivot(image)
    if adjustment_center is None:
        raise ValueError("Zero-adjustment screw is required for two-line calibration")

    outer_ellipse = detect_outer_meter_ellipse(image)
    outer_edge_support, _ = measure_outer_meter_ellipse_quality(
        image,
        outer_ellipse,
    )
    transform = ellipse_rectification(outer_ellipse)
    _, _, middle_scale_point = select_rectified_middle_scale_mark(
        tick_lines,
        visible_adjustment_center=adjustment_center,
        crop_shape=(height, width),
        rectification_transform=transform,
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    pointer_line, _, _ = _select_meter_pointer_segment(
        pointer_lines,
        hub=rough_estimate.center,
        adjustment_center=adjustment_center,
        crop_shape=(height, width),
    )
    estimate = infer_two_line_hidden_meter_pivot(
        pointer_line,
        middle_scale_point,
        adjustment_center,
        crop_shape=(height, width),
        inlier_indices=rough_estimate.inlier_indices,
        scale_residual=rough_estimate.median_residual,
    )
    rectified_center = _transform_affine_point(transform, estimate.center)
    rectified_middle = _transform_affine_point(transform, middle_scale_point)
    axis_vector = rectified_center - rectified_middle
    axis_offset = math.degrees(
        math.atan2(float(axis_vector[0]), float(axis_vector[1]))
    )
    estimate = replace(
        estimate,
        angular_spread_degrees=rough_estimate.angular_spread_degrees,
        axis_offset_degrees=axis_offset,
        rectification_transform=transform,
        outer_ellipse=outer_ellipse,
        outer_ellipse_edge_support=outer_edge_support,
    )
    pointer_points = pointer_line.reshape(2, 2)
    pointer_tip = pointer_points[
        int(np.argmax(np.linalg.norm(pointer_points - estimate.center, axis=1)))
    ]
    return estimate, tick_lines, pointer_tip


def infer_hidden_meter_sweep_fraction(
    estimate: HiddenPivotEstimate,
    tick_lines: np.ndarray,
    *,
    pointer_tip: np.ndarray,
) -> float:
    """Map a pointer to the scale using the observed radial tick fan."""

    raw_lines = np.asarray(tick_lines, dtype=np.float64).reshape(-1, 4)
    center = estimate.center
    tip = np.asarray(pointer_tip, dtype=np.float64)
    if estimate.rectification_transform is not None:
        center = _transform_affine_point(
            estimate.rectification_transform,
            center,
        )
        tip = _transform_affine_point(
            estimate.rectification_transform,
            tip,
        )
    directions: list[float] = []
    for line_index in estimate.inlier_indices:
        raw_line = raw_lines[line_index]
        midpoint = (raw_line[:2] + raw_line[2:]) / 2.0
        if estimate.rectification_transform is not None:
            midpoint = _transform_affine_point(
                estimate.rectification_transform,
                midpoint,
            )
        angle = angle_from_points(center, midpoint)
        directions.append((angle + 180.0) % 360.0 - 180.0)
    if len(directions) < 3:
        raise ValueError("Too few scale directions to map the hidden pointer")

    start_angle = min(directions)
    end_angle = max(directions)
    sweep = end_angle - start_angle
    if not 50.0 <= sweep <= 150.0:
        raise ValueError("Hidden-pivot scale fan has an implausible sweep")

    pointer_angle = angle_from_points(center, tip)
    pointer_angle = (pointer_angle + 180.0) % 360.0 - 180.0
    endpoint_tolerance = max(3.0, 0.05 * sweep)
    if pointer_angle < start_angle - endpoint_tolerance:
        raise ValueError("Pointer falls before the observed hidden-pivot scale")
    if pointer_angle > end_angle + endpoint_tolerance:
        raise ValueError("Pointer falls after the observed hidden-pivot scale")
    if pointer_angle - start_angle <= endpoint_tolerance:
        return 0.0
    if end_angle - pointer_angle <= endpoint_tolerance:
        return 1.0
    return float(np.clip((pointer_angle - start_angle) / sweep, 0.0, 1.0))


def select_meter_face_geometry(
    lines: np.ndarray,
    *,
    crop_shape: tuple[int, int],
) -> tuple[float, float | None]:
    """Locate a rectangular meter from its lower frame, ignoring scale strokes."""
    height, _ = crop_shape
    horizontal: list[tuple[float, float, float]] = []
    for raw_line in np.asarray(lines).reshape(-1, 4):
        start = raw_line[:2].astype(np.float64)
        end = raw_line[2:].astype(np.float64)
        angle = abs(
            math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
        )
        angle = min(angle, abs(180.0 - angle))
        midpoint_y = float((start[1] + end[1]) / 2.0)
        length = float(np.linalg.norm(end - start))
        if angle <= 15.0 and 0.30 * height <= midpoint_y <= 0.82 * height:
            horizontal.append(
                (length, float((start[0] + end[0]) / 2.0), midpoint_y)
            )
    if not horizontal:
        raise ValueError("Rectangular meter horizontal boundary was not found")
    lower_boundaries = [item for item in horizontal if item[2] >= 0.65 * height]
    if lower_boundaries:
        lower_boundary = max(lower_boundaries, key=lambda item: item[0])
        return lower_boundary[1], max(item[2] for item in lower_boundaries)
    return max(horizontal, key=lambda item: item[0])[1], None


def select_meter_pointer_line(
    lines: np.ndarray,
    *,
    hub: np.ndarray,
    adjustment_center: np.ndarray | None = None,
    crop_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select the longest radial line that reaches the panel-meter hub."""
    _, tip, angle = _select_meter_pointer_segment(
        lines,
        hub=hub,
        adjustment_center=adjustment_center,
        crop_shape=crop_shape,
    )
    return np.asarray(hub, dtype=np.float64), tip, angle


def _select_meter_pointer_segment(
    lines: np.ndarray,
    *,
    hub: np.ndarray,
    adjustment_center: np.ndarray | None = None,
    crop_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the observed pointer segment as well as its reading endpoint."""

    height, width = crop_shape
    hub = np.asarray(hub, dtype=np.float64)
    candidates: list[
        tuple[float, float, float, np.ndarray, np.ndarray, float]
    ] = []
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
        if perpendicular_distance > 0.15 * width:
            continue
        adjustment_distance = 0.0
        if adjustment_center is not None:
            adjustment_distance = float(
                np.linalg.norm(near - np.asarray(adjustment_center, dtype=np.float64))
            )
            if adjustment_distance > 0.16 * width:
                continue
        candidates.append(
            (
                adjustment_distance,
                -length,
                perpendicular_distance,
                raw_line.astype(np.float64),
                tip,
                angle,
            )
        )
    if not candidates:
        raise ValueError("Rectangular meter pointer line was not found")
    _, _, _, pointer_line, tip, angle = min(
        candidates,
        key=lambda item: (item[1], item[2], item[0]),
    )
    return pointer_line, tip, angle


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


def red_chroma_mask(image: np.ndarray) -> np.ndarray:
    """Keep genuinely red pixels while rejecting low-chroma shadows and glass tint."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue, green, red = cv2.split(image)
    red_dominance = red.astype(np.int16) - np.maximum(blue, green).astype(np.int16)
    return (
        ((hue <= 20) | (hue >= 160))
        & (saturation >= 45)
        & (value >= 30)
        & (red_dominance >= 15)
    ).astype(np.uint8)


def _selected_red_pointer_mask(
    image: np.ndarray,
    center: np.ndarray,
    tip: np.ndarray,
) -> np.ndarray:
    """Select the red component whose line, or extension, passes through the hub."""
    mask = red_chroma_mask(image)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    direction = tip - center
    pointer_length = float(np.linalg.norm(direction))
    if pointer_length <= 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    direction /= pointer_length
    normal = np.asarray((-direction[1], direction[0]))
    line_tolerance = max(3.0, min(image.shape[:2]) * 0.035)
    hub_tolerance = max(5.0, min(image.shape[:2]) * 0.12)
    minimum_area = max(10, round(mask.size * 0.0002))
    candidates: list[tuple[int, int, int]] = []
    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        ys, xs = np.where(labels == index)
        points = np.column_stack((xs, ys)).astype(np.float64)
        offsets = points - center
        line_support = int(np.count_nonzero(np.abs(offsets @ normal) <= line_tolerance))
        nearest_hub = float(np.min(np.linalg.norm(offsets, axis=1)))
        if line_support < minimum_area or nearest_hub > hub_tolerance:
            continue
        candidates.append((line_support, area, index))
    if not candidates:
        return np.zeros(mask.shape, dtype=np.uint8)
    selected = max(candidates)[2]
    return (labels == selected).astype(np.uint8)


def detect_colored_component_pointer(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Recover a red pointer component that crosses the central dial area."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = (
        ((hue <= 20) | (hue >= 160)) & (saturation > 30) & (value > 20)
    ).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    image_center = np.asarray((width / 2.0, height / 2.0))
    image_area = width * height
    candidates: list[tuple[float, np.ndarray, np.ndarray, float, np.ndarray]] = []
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
        candidates.append((score, center, tip, angle, component))
    if not candidates:
        raise ValueError("No central colored pointer component")
    score, center, tip, angle, component = max(
        candidates, key=lambda item: item[0]
    )
    pointer_lines = cv2.HoughLinesP(
        component * 255,
        1,
        np.pi / 360.0,
        threshold=max(10, round(min(width, height) * 0.08)),
        minLineLength=max(10, round(min(width, height) * 0.13)),
        maxLineGap=max(4, round(min(width, height) * 0.07)),
    )
    if pointer_lines is not None:
        try:
            center, tip, angle = select_meter_pointer_line(
                pointer_lines,
                hub=center,
                crop_shape=(height, width),
            )
        except ValueError:
            pass
    confidence = float(np.clip(0.45 + score / max(width, height) * 0.25, 0.45, 0.8))
    return center, tip, angle, confidence


def _write_selected_pointer_stages(
    writer: ProcessingStageWriter | None,
    image: np.ndarray,
    candidate: DialCandidate,
    center: np.ndarray,
    tip: np.ndarray,
    *,
    group: str,
    method: str,
    include_red_mask: bool = False,
    include_outer_label_ring: bool = False,
    include_hidden_pivot_evidence: bool = False,
    outer_label_extent_factor: float = 0.68,
) -> None:
    if writer is None:
        return
    x1, y1, x2, y2 = candidate.bbox
    if include_outer_label_ring:
        inner_size = max(x2 - x1, y2 - y1)
        # Radius from the pointer hub, not a per-side padding.  Different dial
        # constructions place their readable labels at different radii.
        outer_extent = inner_size * outer_label_extent_factor
        x1 = max(0, round(float(center[0]) - outer_extent))
        y1 = max(0, round(float(center[1]) - outer_extent))
        x2 = min(image.shape[1], round(float(center[0]) + outer_extent))
        y2 = min(image.shape[0], round(float(center[1]) + outer_extent))
    selected_bbox = (x1, y1, x2, y2)
    crop = image[y1:y2, x1:x2]
    writer.write(
        group,
        "selected-detection",
        draw_dial_candidates(
            image,
            [(selected_bbox, candidate.confidence)],
            selected_bbox=selected_bbox,
        ),
        title_zh="最终采用的仪表区域",
        operation="type_specific_candidate_selection",
        source_stage="analysis-image",
        preserves_aspect_ratio=True,
        note_zh="该绿色框是最终读数路径采用的完整仪表区域。",
    )
    writer.write(
        group,
        "selected-crop",
        crop,
        title_zh="最终几何读取裁剪",
        operation=f"crop_bbox={list(selected_bbox)}",
        source_stage="selected-detection",
        preserves_aspect_ratio=True,
        note_zh="保留完整仪表外圈和内部读数窗口，未拉伸。",
    )
    hidden_estimate: HiddenPivotEstimate | None = None
    if include_hidden_pivot_evidence:
        try:
            estimate, _, detected_pointer_tip = (
                detect_two_line_hidden_meter_geometry(crop)
            )
        except ValueError:
            pass
        else:
            hidden_estimate = estimate
            if estimate.outer_ellipse is not None:
                writer.write(
                    group,
                    "outer-ellipse-fit",
                    draw_ellipse(crop, estimate.outer_ellipse),
                    title_zh="外圈椭圆姿态基准",
                    operation="outer_housing_edge_ellipse_fit",
                    source_stage="selected-crop",
                    preserves_aspect_ratio=True,
                    note_zh=(
                        "绿色椭圆拟合仪表外壳完整圆环，只用于校正斜拍姿态；"
                        "不会把调零螺钉当作指针轴心。"
                        + (
                            "可见圆弧边缘贴合率 "
                            f"{estimate.outer_ellipse_edge_support:.1%}。"
                            if estimate.outer_ellipse_edge_support is not None
                            else ""
                        )
                    ),
                )
            evidence_overlay = crop.copy()
            pivot_point = tuple(np.rint(estimate.center).astype(int))
            middle_line = np.asarray(estimate.middle_scale_line).reshape(4)
            middle_scale_point = tuple(np.rint(middle_line[:2]).astype(int))
            adjustment_point = tuple(np.rint(middle_line[2:]).astype(int))
            pointer_tip_point = tuple(np.rint(detected_pointer_tip).astype(int))
            _draw_extended_calibration_ray(
                evidence_overlay,
                np.asarray(estimate.pointer_line).reshape(4),
                pointer_tip_point,
                pivot_point,
                (0, 0, 255),
            )
            _draw_extended_calibration_ray(
                evidence_overlay,
                middle_line,
                middle_scale_point,
                pivot_point,
                (0, 190, 255),
            )
            cv2.circle(
                evidence_overlay,
                adjustment_point,
                7,
                (255, 120, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                evidence_overlay,
                pivot_point,
                8,
                (255, 0, 255),
                3,
                cv2.LINE_AA,
            )
            writer.write(
                group,
                "hidden-pivot-evidence",
                evidence_overlay,
                title_zh="两线隐藏轴心交叉验证",
                operation="pointer_line+middle_tick_adjustment_axis",
                source_stage=(
                    "outer-ellipse-fit"
                    if estimate.outer_ellipse is not None
                    else "selected-crop"
                ),
                preserves_aspect_ratio=True,
                note_zh=(
                    "红线是实测指针延长线；橙线经过椭圆校正后选中的中间"
                    "刻度和蓝色调零螺钉；紫点是两线交点，即隐藏指针轴心。"
                    "最右端刻度不参与轴心求解，只由完整刻度扇面复核读数。"
                    f"校正后中心轴偏移 {estimate.axis_offset_degrees:.2f}°，"
                    f"刻度扇面残差 {estimate.median_residual:.2f}px。"
                ),
            )
            if estimate.outer_ellipse is not None:
                rectified, transform = rectify_dial(crop, estimate.outer_ellipse)
                rectified_center = _transform_affine_point(transform, estimate.center)
                rectified_tip = _transform_affine_point(transform, detected_pointer_tip)
                rectified_middle = _transform_affine_point(transform, middle_line[:2])
                rectified_adjustment = _transform_affine_point(
                    transform,
                    middle_line[2:],
                )
                rectified_overlay = rectified.copy()
                _draw_extended_calibration_ray(
                    rectified_overlay,
                    np.concatenate((rectified_center, rectified_tip)),
                    tuple(np.rint(rectified_tip).astype(int)),
                    tuple(np.rint(rectified_center).astype(int)),
                    (0, 0, 255),
                )
                _draw_extended_calibration_ray(
                    rectified_overlay,
                    np.concatenate((rectified_middle, rectified_adjustment)),
                    tuple(np.rint(rectified_middle).astype(int)),
                    tuple(np.rint(rectified_center).astype(int)),
                    (0, 190, 255),
                )
                cv2.circle(
                    rectified_overlay,
                    tuple(np.rint(rectified_adjustment).astype(int)),
                    7,
                    (255, 120, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    rectified_overlay,
                    tuple(np.rint(rectified_center).astype(int)),
                    8,
                    (255, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
                writer.write(
                    group,
                    "rectified-two-line-geometry",
                    rectified_overlay,
                    title_zh="姿态校正后的两线几何",
                    operation="outer_ellipse_affine_rectification+two_line_projection",
                    source_stage="hidden-pivot-evidence",
                    preserves_aspect_ratio=False,
                    note_zh=(
                        "外圈椭圆被恢复为圆后，红色指针线与橙色中轴线仍在"
                        "紫色隐藏轴心相交；该坐标系用于角度和刻度扇面映射。"
                    ),
                )
    if include_red_mask:
        local_center = center - np.asarray((x1, y1), dtype=np.float64)
        local_tip = tip - np.asarray((x1, y1), dtype=np.float64)
        red_mask = _selected_red_pointer_mask(crop, local_center, local_tip)
        writer.write(
            group,
            "colored-pointer-mask",
            red_mask,
            title_zh="筛选后的红色指针组件",
            operation="red_chroma_and_hub_line_component",
            source_stage="selected-crop",
            preserves_aspect_ratio=True,
            note_zh="仅保留其直线或延长线穿过轴心的红色连通组件。",
        )
    overlay = crop.copy()
    local_center = np.rint(center - np.asarray((x1, y1))).astype(int)
    local_tip = np.rint(tip - np.asarray((x1, y1))).astype(int)
    line_width = max(2, round(max(crop.shape[:2]) / 180))
    cv2.circle(
        overlay,
        tuple(local_center),
        max(4, line_width * 2),
        (40, 180, 40),
        line_width,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(
        overlay,
        tuple(local_center),
        tuple(local_tip),
        (0, 0, 255),
        line_width,
        cv2.LINE_AA,
        tipLength=0.08,
    )
    two_line_mapping = (
        hidden_estimate is not None
        and hidden_estimate.projection_mode == "two_line_hidden_pivot"
    )
    writer.write(
        group,
        "selected-pointer-geometry",
        overlay,
        title_zh=(
            "最终采用的两线刻度映射几何"
            if two_line_mapping
            else "最终采用的指针几何"
        ),
        operation=method,
        source_stage=(
            "colored-pointer-mask"
            if include_red_mask
            else "hidden-pivot-evidence"
            if include_hidden_pivot_evidence
            else "selected-crop"
        ),
        preserves_aspect_ratio=True,
        note_zh=(
            "红箭头从两线交点得到的隐藏轴心沿实测指针方向指向读数端；"
            "读数角度已在外圈椭圆校正坐标中计算。"
            if two_line_mapping
            else "红箭头从轴心指向实际读数端；这是最终读数采用的方向。"
        ),
    )


def _draw_extended_calibration_ray(
    image: np.ndarray,
    line: np.ndarray,
    anchor: tuple[int, int],
    toward: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a measured line direction past its feature without forcing a hit."""

    raw_line = np.asarray(line, dtype=np.float64).reshape(4)
    direction = raw_line[2:] - raw_line[:2]
    if float(np.dot(direction, np.asarray(toward) - np.asarray(anchor))) < 0.0:
        direction *= -1.0
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    endpoint = np.asarray(anchor, dtype=np.float64) + direction * (
        2.0 * max(image.shape[:2])
    )
    intersects, clipped_start, clipped_end = cv2.clipLine(
        (0, 0, image.shape[1], image.shape[0]),
        tuple(np.rint(anchor).astype(int)),
        tuple(np.rint(endpoint).astype(int)),
    )
    if intersects:
        cv2.line(
            image,
            clipped_start,
            clipped_end,
            color,
            2,
            cv2.LINE_AA,
        )


def _shm_mechanism_status_analysis(
    image: np.ndarray,
    candidate: DialCandidate,
    outer_pointer: GaugeResult,
    stage_writer: ProcessingStageWriter | None,
) -> ChannelAnalysis | None:
    """Read the independent short red SHM-D status arm near the pointer hub."""
    if outer_pointer.center is None:
        return None
    center = np.asarray(outer_pointer.center, dtype=np.float64)
    x1, y1, x2, y2 = candidate.bbox
    inner_size = float(max(x2 - x1, y2 - y1))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = ((hue <= 20) | (hue >= 160)) & (saturation > 30) & (value > 20)
    ys, xs = np.nonzero(red)
    if len(xs) == 0:
        return None
    global_x = xs.astype(np.float64) + x1
    global_y = ys.astype(np.float64) + y1
    delta_x = global_x - center[0]
    delta_y = global_y - center[1]
    radii = np.hypot(delta_x, delta_y)
    radial_band = (radii >= inner_size * 0.08) & (radii <= inner_size * 0.30)
    if int(np.count_nonzero(radial_band)) < 20:
        return None
    angles = (
        np.degrees(np.arctan2(delta_x[radial_band], -delta_y[radial_band]))
        + 360.0
    ) % 360.0
    if outer_pointer.angle_degrees is not None:
        outer_distance = np.abs(
            (angles - outer_pointer.angle_degrees + 180.0) % 360.0 - 180.0
        )
        angles = angles[outer_distance > 25.0]
    if len(angles) < 20:
        return None
    histogram, _edges = np.histogram(angles, bins=np.arange(361, dtype=np.float64))
    extended = np.concatenate((histogram[-5:], histogram, histogram[:5]))
    smoothed = np.convolve(extended, np.ones(11, dtype=np.float64), mode="valid")
    status_angle = float(np.argmax(smoothed) % 360)
    peak_support = float(np.max(smoothed))
    if peak_support < 20.0:
        return None
    bottom_distance = abs((status_angle - 180.0 + 180.0) % 360.0 - 180.0)
    status = "at_position" if bottom_distance <= 30.0 else "in_transition"
    tip_radius = inner_size * 0.30
    radians = math.radians(status_angle)
    tip = center + np.asarray(
        (math.sin(radians) * tip_radius, -math.cos(radians) * tip_radius)
    )
    _write_selected_pointer_stages(
        stage_writer,
        image,
        candidate,
        center,
        tip,
        group="dial-1-selected-inner-status",
        method="type-specific:shm-short-red-status-arm",
        include_red_mask=True,
    )
    support_fraction = peak_support / max(float(len(angles)), 1.0)
    return ChannelAnalysis(
        instance_id="instance_1",
        channel_id="mechanism_status",
        value=status,
        unit="state",
        status="recognized",
        method="type-specific:shm-short-red-status-arm",
        confidence=float(np.clip(support_fraction, 0.45, 0.95)),
        note_zh=(
            "内圈短红指针位于底部到位分区。"
            if status == "at_position"
            else "内圈短红指针离开底部到位分区。"
        ),
    )


def _hidden_meter_geometry_candidates(
    detector_candidate: DialCandidate,
    existing: GaugeResult,
    *,
    image_shape: tuple[int, int],
) -> tuple[DialCandidate, ...]:
    """Generate conservative inner crops around the nested reader's dial box.

    The detector often includes a narrow strip of the metal rim.  Those edges
    can dominate Hough line detection, while a large generic padding can cut
    the scale window too loosely.  Try small, deterministic insets first and
    keep the unmodified boxes as fallbacks.
    """

    image_height, image_width = image_shape
    base_bbox = (
        existing.bbox
        if existing.detected and existing.bbox is not None
        else detector_candidate.bbox
    )
    bases = (base_bbox, detector_candidate.bbox)
    scales = (0.94, 0.90, 0.86, 1.0, 0.82, 0.98, 1.02, 1.06)
    candidates: list[DialCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    for bbox in bases:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        for scale in scales:
            scaled_bbox = (
                max(0, round(center_x - width * scale / 2.0)),
                max(0, round(center_y - height * scale / 2.0)),
                min(image_width, round(center_x + width * scale / 2.0)),
                min(image_height, round(center_y + height * scale / 2.0)),
            )
            if scaled_bbox in seen:
                continue
            seen.add(scaled_bbox)
            candidates.append(
                DialCandidate(scaled_bbox, detector_candidate.confidence)
            )
    return tuple(candidates)


def _recover_type_specific_pointer_results(
    image: np.ndarray,
    metadata: InstrumentTypeMetadata,
    candidates: tuple[DialCandidate, ...],
    pointer_results: tuple[GaugeResult, ...],
    visible_text: str,
    reader: EthzPaddleGaugeReader,
    stage_writer: ProcessingStageWriter | None = None,
) -> tuple[GaugeResult, ...]:
    recovered = list(pointer_results)
    while len(recovered) < len(candidates):
        recovered.append(_empty_pointer_result(candidates[len(recovered)]))

    if metadata.type_id == "surge_arrester_monitor":
        for index, candidate in enumerate(candidates):
            existing = recovered[index]
            geometry_candidate: DialCandidate | None = None
            geometry: tuple[np.ndarray, np.ndarray, float, float | None] | None = None
            for trial_candidate in _hidden_meter_geometry_candidates(
                candidate,
                existing,
                image_shape=image.shape[:2],
            ):
                try:
                    geometry = _detect_rectangular_meter_pointer(
                        image,
                        trial_candidate,
                        require_hidden_pivot=True,
                    )
                except ValueError:
                    continue
                geometry_candidate = trial_candidate
                break
            if geometry_candidate is None or geometry is None:
                continue
            center, tip, angle, sweep_fraction = geometry
            visual = replace(
                existing,
                detected=True,
                bbox=geometry_candidate.bbox,
                detection_confidence=candidate.confidence,
                pointer_found=True,
                center=(float(center[0]), float(center[1])),
                pointer_tip=(float(tip[0]), float(tip[1])),
                angle_degrees=angle,
                sweep_fraction=sweep_fraction,
                reading=None,
                unit=None,
                confidence=candidate.confidence,
                center_method=(
                    "type-specific:ellipse-rectified-two-line-hidden-pivot+tick-scale"
                ),
                failure_reason=None,
                raw_reading=None,
                instrument_type_id=None,
                readout_channel_id=None,
                interpretation_method=None,
                reading_candidates=(),
            )
            interpreted = _interpret_recovered_result(reader, visual, visible_text)
            if interpreted.pointer_found and interpreted.sweep_fraction is not None:
                recovered[index] = interpreted
                _write_selected_pointer_stages(
                    stage_writer,
                    image,
                    geometry_candidate,
                    center,
                    tip,
                    group=f"dial-{index + 1}-selected-meter",
                    method="ellipse_rectified_two_line_hidden_pivot",
                    include_hidden_pivot_evidence=True,
                )

    if metadata.type_id in MULTI_INSTANCE_RECTANGULAR_TYPES:
        for index, candidate in enumerate(candidates):
            existing = recovered[index]
            try:
                center, tip, angle, _ = _detect_rectangular_meter_pointer(
                    image,
                    candidate,
                )
            except ValueError:
                if existing.pointer_found and existing.center and existing.pointer_tip:
                    _write_selected_pointer_stages(
                        stage_writer,
                        image,
                        candidate,
                        np.asarray(existing.center, dtype=np.float64),
                        np.asarray(existing.pointer_tip, dtype=np.float64),
                        group=f"dial-{index + 1}-selected-rectangular-meter",
                        method=existing.center_method or "accepted_pointer_geometry",
                    )
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
            recovered[index] = _interpret_recovered_result(
                reader,
                visual,
                visible_text,
            )
            _write_selected_pointer_stages(
                stage_writer,
                image,
                candidate,
                center,
                tip,
                group=f"dial-{index + 1}-selected-rectangular-meter",
                method="rectangular_meter_hub_and_extended_line",
            )

    if metadata.type_id == "shm_d_motor_drive_unit" and recovered:
        existing = recovered[0]
        if candidates:
            x1, y1, x2, y2 = candidates[0].bbox
            try:
                center, tip, angle, color_confidence = (
                    detect_colored_component_pointer(image[y1:y2, x1:x2])
                )
            except ValueError:
                pass
            else:
                offset = np.asarray((x1, y1), dtype=np.float64)
                existing = replace(
                    existing,
                    detected=True,
                    bbox=candidates[0].bbox,
                    detection_confidence=candidates[0].confidence,
                    pointer_found=True,
                    center=tuple(center + offset),
                    pointer_tip=tuple(tip + offset),
                    angle_degrees=angle,
                    reading=None,
                    unit=None,
                    confidence=min(
                        candidates[0].confidence,
                        color_confidence,
                    ),
                    center_method="type-specific:colored-component-pointer",
                    failure_reason=None,
                    raw_reading=None,
                    instrument_type_id=None,
                    readout_channel_id=None,
                    interpretation_method=None,
                    reading_candidates=(),
                )
                recovered[0] = _interpret_recovered_result(
                    reader, existing, visible_text
                )
                _write_selected_pointer_stages(
                    stage_writer,
                    image,
                    candidates[0],
                    center + offset,
                    tip + offset,
                    group="dial-1-selected-outer-pointer",
                    method="colored_component_and_hough_line",
                    include_red_mask=True,
                    include_outer_label_ring=True,
                    outer_label_extent_factor=1.2,
                )
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
        if channel is not None and channel.allowed_values:
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
                _write_selected_pointer_stages(
                    stage_writer,
                    image,
                    candidates[0],
                    center + offset,
                    tip + offset,
                    group="dial-1-selected-discharge-pointer",
                    method="colored_component_and_hough_line",
                    include_red_mask=True,
                    include_outer_label_ring=True,
                )
    return tuple(recovered)


def _detect_rectangular_meter_pointer(
    image: np.ndarray,
    candidate: DialCandidate,
    *,
    require_hidden_pivot: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float | None]:
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
    face_center_x, face_bottom_y = select_meter_face_geometry(
        face_lines,
        crop_shape=(height, width),
    )

    if require_hidden_pivot:
        estimate, tick_lines, tip = detect_two_line_hidden_meter_geometry(crop)
        center = estimate.center
        angle_center = center
        angle_tip = tip
        if estimate.rectification_transform is not None:
            angle_center = _transform_affine_point(
                estimate.rectification_transform,
                center,
            )
            angle_tip = _transform_affine_point(
                estimate.rectification_transform,
                tip,
            )
        angle = angle_from_points(angle_center, angle_tip)
        sweep_fraction = infer_hidden_meter_sweep_fraction(
            estimate,
            tick_lines,
            pointer_tip=tip,
        )
        offset = np.asarray((x1, y1), dtype=np.float64)
        return center + offset, tip + offset, angle, sweep_fraction

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
    return center + offset, tip + offset, angle, None


def _recognize_pointer_aligned_discrete_label(
    image: np.ndarray,
    result: GaugeResult,
    reader: EthzPaddleGaugeReader,
) -> tuple[str, float]:
    if result.bbox is None or result.center is None or result.angle_degrees is None:
        raise ValueError("Discrete dial pointer geometry is incomplete")
    x1, y1, x2, y2 = result.bbox
    dial_size = float(max(x2 - x1, y2 - y1))
    patch_width = max(24, round(dial_size * 0.18))
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
    groups: list[list[DialCandidate]] = []
    for anchor in candidates:
        ax1, ay1, ax2, ay2 = anchor.bbox
        anchor_width = ax2 - ax1
        anchor_height = ay2 - ay1
        anchor_center_y = (ay1 + ay2) / 2.0
        group = []
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
            group.append(candidate)
        groups.append(group)

    def group_rank(group: list[DialCandidate]) -> tuple[int, float, float]:
        median_area = float(
            np.median(
                [
                    (item.bbox[2] - item.bbox[0])
                    * (item.bbox[3] - item.bbox[1])
                    for item in group
                ]
            )
        )
        return (len(group), median_area, sum(item.confidence for item in group))

    selected = max(groups, key=group_rank)
    return tuple(sorted(selected, key=lambda item: item.bbox[0]))


def _bbox_area_fraction(
    bbox: tuple[int, int, int, int], image_shape: tuple[int, int]
) -> float:
    x1, y1, x2, y2 = bbox
    image_height, image_width = image_shape
    return (x2 - x1) * (y2 - y1) / float(image_height * image_width)


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
    # Detector boxes often follow the printed face while useful perspective
    # geometry lives on the surrounding bezel.  Keep enough context for the
    # nested reader to see that complete outer rim.
    pad_x = round((x2 - x1) * 0.20)
    pad_y = round((y2 - y1) * 0.20)
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


def _generic_result_rank(result: GaugeResult) -> tuple[int, int, float]:
    """Prefer validated rectification only when comparing generic read paths."""
    level, confidence = _result_rank(result)
    method = result.center_method or ""
    geometry_level = int(
        "edge-ellipse" in method and "unrectified-pointer-fallback" not in method
    )
    return (level, geometry_level, confidence)


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
