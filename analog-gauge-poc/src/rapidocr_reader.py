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
from src.pipeline_profile import GaugePipelineProfile, get_gauge_pipeline_profile
from src.processing_stages import (
    ProcessingStageWriter,
    draw_dial_candidates,
    draw_ellipse,
    draw_pointer_geometry,
)

RECTIFIED_SIZE = 640
DIAL_RADIUS = RECTIFIED_SIZE * 0.47
SECTOR_STEP_DEGREES = 15
NUMBER_PATTERN = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
MINIMUM_SCALE_LABELS = 3
MINIMUM_DIAL_ELLIPSE_EDGE_SUPPORT = 0.80
MINIMUM_DIAL_ELLIPSE_VISIBLE_ARC = 0.50
MINIMUM_PROJECTIVE_DIAL_DIMENSION = 800
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


@dataclass(frozen=True)
class PointerSegmentationTrace:
    """Selected segmentation path plus the exact intermediate pixel arrays."""

    candidate: DialCandidate
    mask: np.ndarray
    confidence: float
    method: str
    crop: np.ndarray
    canvas: np.ndarray
    model_input: np.ndarray | None
    model_output_mask: np.ndarray | None
    content_bbox: tuple[int, int, int, int]


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


def detect_rectangular_gauge_candidates(image: np.ndarray) -> list[DialCandidate]:
    """Find large dark instrument frames missed by the circular-gauge detector."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
        iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[DialCandidate] = []
    image_area = float(height * width)
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area_fraction = box_width * box_height / image_area
        aspect_ratio = box_width / box_height
        if not 0.01 <= area_fraction <= 0.40:
            continue
        if not 0.60 <= aspect_ratio <= 2.70:
            continue
        if box_width < width * 0.10 or box_height < height * 0.12:
            continue
        rectangularity = cv2.contourArea(contour) / max(1.0, box_width * box_height)
        if rectangularity < 0.75:
            continue
        crop = gray[y : y + box_height, x : x + box_width]
        band = max(3, round(min(box_width, box_height) * 0.08))
        inner = crop[band:-band, band:-band]
        if inner.size == 0:
            continue
        border = np.concatenate(
            (
                crop[:band, :].ravel(),
                crop[-band:, :].ravel(),
                crop[:, :band].ravel(),
                crop[:, -band:].ravel(),
            )
        )
        border_contrast = float(inner.mean() - border.mean())
        if border_contrast < 20.0:
            continue
        confidence = float(
            np.clip(0.55 + border_contrast / 400.0 + rectangularity * 0.05, 0.55, 0.9)
        )
        candidate_bottom = y + box_height
        if aspect_ratio >= 1.45:
            # Wide half-dial windows commonly place the pivot in the dark lower
            # housing.  Include that housing so the pointer line and its
            # extension can be validated against the real pivot.
            candidate_bottom = min(height, y + round(box_height * 1.75))
        candidates.append(
            DialCandidate((x, y, x + box_width, candidate_bottom), confidence)
        )
    return deduplicate_dial_candidates(candidates)


def map_bbox_between_images(
    bbox: tuple[int, int, int, int],
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map one detection box between aligned images without clipping its content."""

    source_height, source_width = source_shape
    target_height, target_width = target_shape
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("Image dimensions must be positive")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(target_width, math.floor(x1 * scale_x))),
        max(0, min(target_height, math.floor(y1 * scale_y))),
        max(0, min(target_width, math.ceil(x2 * scale_x))),
        max(0, min(target_height, math.ceil(y2 * scale_y))),
    )


def expand_candidate_and_pointer_mask(
    candidate: DialCandidate,
    pointer_mask: np.ndarray,
    *,
    image_shape: tuple[int, int],
    margin_fraction: float,
) -> tuple[DialCandidate, np.ndarray]:
    """Add scene context around a detected face without moving its pointer mask."""
    if margin_fraction <= 0:
        return candidate, pointer_mask
    image_height, image_width = image_shape
    x1, y1, x2, y2 = candidate.bbox
    width, height = x2 - x1, y2 - y1
    expanded_bbox = (
        max(0, round(x1 - width * margin_fraction)),
        max(0, round(y1 - height * margin_fraction)),
        min(image_width, round(x2 + width * margin_fraction)),
        min(image_height, round(y2 + height * margin_fraction)),
    )
    expanded_x1, expanded_y1, expanded_x2, expanded_y2 = expanded_bbox
    if pointer_mask.shape != (height, width):
        pointer_mask = cv2.resize(
            pointer_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    expanded_mask = np.zeros(
        (expanded_y2 - expanded_y1, expanded_x2 - expanded_x1),
        dtype=pointer_mask.dtype,
    )
    offset_x, offset_y = x1 - expanded_x1, y1 - expanded_y1
    expanded_mask[offset_y : offset_y + height, offset_x : offset_x + width] = (
        pointer_mask
    )
    return DialCandidate(expanded_bbox, candidate.confidence), expanded_mask


def pad_dial_crop_to_square(
    crop: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pad a dial crop to a square while preserving all original pixel geometry."""

    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Dial crop is empty")
    side = max(height, width)
    left = (side - width) // 2
    right = side - width - left
    top = (side - height) // 2
    bottom = side - height - top
    padded = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_REPLICATE,
    )
    return padded, (left, top, left + width, top + height)


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


def select_pointer_segmentation_trace(
    traces: list[PointerSegmentationTrace],
    image_shape: tuple[int, int],
) -> PointerSegmentationTrace:
    """Prefer a local dial crop over a scene-sized box for the same pointer.

    The detector can emit both a real dial candidate and a near-full-frame object
    box.  A pointer segmenter may score the latter more highly even though that
    crop cannot support reliable rim fitting or scale OCR.  We only suppress such
    a frame when another compact candidate independently locates the same hub.
    """
    if not traces:
        raise ValueError("No pointer segmentation traces to select")
    image_height, image_width = image_shape
    image_area = float(image_height * image_width)

    def bbox_area(trace: PointerSegmentationTrace) -> float:
        x1, y1, x2, y2 = trace.candidate.bbox
        return float((x2 - x1) * (y2 - y1))

    def global_hub(trace: PointerSegmentationTrace) -> np.ndarray:
        hub, _ = pointer_center_and_tip(trace.mask)
        x1, y1, _, _ = trace.candidate.bbox
        return hub + np.asarray((x1, y1), dtype=np.float64)

    hubs = {id(trace): global_hub(trace) for trace in traces}
    local_traces: list[PointerSegmentationTrace] = []
    for trace in traces:
        x1, y1, x2, y2 = trace.candidate.bbox
        width, height = x2 - x1, y2 - y1
        edge_margin = max(3, round(min(image_height, image_width) * 0.01))
        touched_edges = sum(
            (
                x1 <= edge_margin,
                y1 <= edge_margin,
                x2 >= image_width - edge_margin,
                y2 >= image_height - edge_margin,
            )
        )
        scene_sized = bbox_area(trace) >= image_area * 0.55 and touched_edges >= 2
        same_pointer_in_compact_box = any(
            other is not trace
            and bbox_area(other) <= image_area * 0.30
            and np.linalg.norm(hubs[id(trace)] - hubs[id(other)])
            <= min(width, height) * 0.08
            for other in traces
        )
        if not (scene_sized and same_pointer_in_compact_box):
            local_traces.append(trace)

    pool = local_traces or traces
    return max(
        pool,
        key=lambda trace: (trace.confidence, trace.candidate.confidence),
    )


def _dial_edge_distance(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    return cv2.distanceTransform(
        (edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )


def ellipse_edge_support(
    ellipse: Ellipse,
    *,
    edge_distance: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    """Measure observed edge support for the visible part of an ellipse."""

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


def measure_dial_ellipse_quality(
    crop: np.ndarray,
    ellipse: Ellipse,
) -> tuple[float, float]:
    """Return edge-support and visible-arc fractions for a dial ellipse."""

    return ellipse_edge_support(
        ellipse,
        edge_distance=_dial_edge_distance(crop),
        image_shape=crop.shape[:2],
    )


def _tight_ellipse_edge_score(
    ellipse: Ellipse,
    *,
    edge_distance: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    """Score pixel-level alignment without a resolution-sized tolerance band."""

    height, width = image_shape
    (center_x, center_y), (axis_a, axis_b), angle = ellipse
    perimeter = np.asarray(
        cv2.ellipse2Poly(
            (round(center_x), round(center_y)),
            (max(1, round(axis_a / 2.0)), max(1, round(axis_b / 2.0))),
            round(angle),
            0,
            359,
            1,
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
    sigma = max(1.5, 0.002 * min(height, width))
    alignment = np.exp(-0.5 * np.square(distances / sigma))
    return float(np.mean(alignment)), visible_arc_fraction


def projective_ellipse_rectification(
    ellipse: Ellipse,
    physical_center: np.ndarray,
) -> np.ndarray:
    """Map a projected physical circle and its true centre to a front-view circle.

    Under perspective projection the centre of an ellipse is generally not the
    image of the physical circle centre.  The polar of the physical centre with
    respect to the ellipse is the dial plane's vanishing line.  Removing that
    projective component before the metric ellipse correction keeps the pointer
    pivot and every genuinely concentric ring at the same rectified centre.
    """

    (center_x, center_y), (axis_a, axis_b), angle = ellipse
    if min(axis_a, axis_b) <= 0:
        raise ValueError("Ellipse axes must be positive")
    theta = math.radians(angle)
    rotation = np.asarray(
        (
            (math.cos(theta), -math.sin(theta)),
            (math.sin(theta), math.cos(theta)),
        ),
        dtype=np.float64,
    )
    quadratic = rotation @ np.diag(
        (4.0 / axis_a**2, 4.0 / axis_b**2)
    ) @ rotation.T
    ellipse_center = np.asarray((center_x, center_y), dtype=np.float64)
    conic = np.zeros((3, 3), dtype=np.float64)
    conic[:2, :2] = quadratic
    conic[:2, 2] = -quadratic @ ellipse_center
    conic[2, :2] = -ellipse_center @ quadratic
    conic[2, 2] = ellipse_center @ quadratic @ ellipse_center - 1.0

    homogeneous_center = np.append(
        np.asarray(physical_center, dtype=np.float64).reshape(2),
        1.0,
    )
    vanishing_line = conic @ homogeneous_center
    if abs(vanishing_line[2]) < 1e-10:
        raise ValueError("Physical centre produces an unstable vanishing line")
    vanishing_line /= vanishing_line[2]
    projective_to_affine = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (vanishing_line[0], vanishing_line[1], 1.0),
        ),
        dtype=np.float64,
    )

    inverse_projective = np.linalg.inv(projective_to_affine)
    affine_conic = inverse_projective.T @ conic @ inverse_projective
    affine_quadratic = 0.5 * (
        affine_conic[:2, :2] + affine_conic[:2, :2].T
    )
    affine_linear = affine_conic[:2, 2]
    affine_center = -np.linalg.solve(affine_quadratic, affine_linear)
    centered_constant = float(
        affine_conic[2, 2] + affine_linear @ affine_center
    )
    normalized_quadratic = affine_quadratic / -centered_constant
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_quadratic)
    if np.any(eigenvalues <= 0):
        raise ValueError("Rectified ellipse is not positive definite")
    square_root = (
        eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    )
    linear = DIAL_RADIUS * square_root
    target_center = np.asarray(
        (RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2),
        dtype=np.float64,
    )
    affine_to_circle = np.eye(3, dtype=np.float64)
    affine_to_circle[:2, :2] = linear
    affine_to_circle[:2, 2] = target_center - linear @ affine_center
    transform = affine_to_circle @ projective_to_affine

    # Circle rectification is ambiguous up to an in-plane rotation.  Preserve
    # the orientation convention of the existing affine rectifier so metadata
    # dial angles keep the same zero direction while perspective is removed.
    center_x, center_y = homogeneous_center[:2]
    denominator = float(
        transform[2, 0] * center_x
        + transform[2, 1] * center_y
        + transform[2, 2]
    )
    numerator = transform[:2, :] @ homogeneous_center
    jacobian = (
        transform[:2, :2] * denominator
        - numerator[:, None] * transform[2, :2]
    ) / denominator**2
    affine_reference = ellipse_rectification(ellipse)[:, :2].astype(np.float64)
    rotation_target = affine_reference @ np.linalg.inv(jacobian)
    left, _, right_transpose = np.linalg.svd(rotation_target)
    residual_rotation = left @ right_transpose
    if np.linalg.det(residual_rotation) < 0:
        left[:, -1] *= -1
        residual_rotation = left @ right_transpose
    rotate_about_center = np.eye(3, dtype=np.float64)
    rotate_about_center[:2, :2] = residual_rotation
    rotate_about_center[:2, 2] = target_center - residual_rotation @ target_center
    transform = rotate_about_center @ transform

    transformed_center = transform @ homogeneous_center
    transformed_center = transformed_center[:2] / transformed_center[2]
    if np.linalg.norm(transformed_center - target_center) > 1.0:
        raise ValueError("Projective rectification did not preserve the pivot")
    return transform.astype(np.float32)


def _projective_ring_edge_quality(
    transform: np.ndarray,
    radius: float,
    *,
    edge_distance: np.ndarray,
    image_shape: tuple[int, int],
    angular_step: int = 2,
) -> float:
    """Measure distributed source-edge support for one rectified circle."""

    height, width = image_shape
    phases = np.arange(0, 360, angular_step, dtype=np.float64)
    radians = np.deg2rad(phases)
    target_center = np.asarray(
        (RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2),
        dtype=np.float64,
    )
    rectified_points = target_center + radius * np.column_stack(
        (np.cos(radians), np.sin(radians))
    )
    inverse = np.linalg.inv(np.asarray(transform, dtype=np.float64).reshape(3, 3))
    homogeneous = np.column_stack(
        (rectified_points, np.ones(len(rectified_points), dtype=np.float64))
    )
    source_homogeneous = homogeneous @ inverse.T
    source_points = source_homogeneous[:, :2] / source_homogeneous[:, 2, None]
    in_bounds = (
        (source_points[:, 0] >= 0)
        & (source_points[:, 0] < width - 1)
        & (source_points[:, 1] >= 0)
        & (source_points[:, 1] < height - 1)
    )
    distances = np.full(len(source_points), 1e6, dtype=np.float32)
    if np.any(in_bounds):
        distances[in_bounds] = cv2.remap(
            edge_distance,
            source_points[in_bounds, 0].astype(np.float32),
            source_points[in_bounds, 1].astype(np.float32),
            cv2.INTER_LINEAR,
        ).reshape(-1)
    sigma = max(1.5, 0.002 * min(height, width))
    alignment = np.exp(-0.5 * np.square(distances / sigma))
    sector_means = np.asarray(
        [
            float(np.mean(alignment[(phases >= start) & (phases < start + 15)]))
            for start in range(0, 360, 15)
        ],
        dtype=np.float64,
    )
    # Ignore at most five occluded sectors, but require the remaining support to
    # be distributed.  A shadow arc cannot win merely by being locally strong.
    distributed_support = float(np.mean(np.sort(sector_means)[5:]))
    coverage = float(np.mean(sector_means > 0.12))
    return (
        0.65 * distributed_support
        + 0.35 * float(np.mean(alignment))
        + 0.10 * coverage
    )


def _projectively_concentric_ring_support(
    edge_distance: np.ndarray,
    image_shape: tuple[int, int],
    ellipse: Ellipse,
    physical_center: np.ndarray,
    *,
    angular_step: int,
) -> tuple[float, float, float, np.ndarray]:
    transform = projective_ellipse_rectification(ellipse, physical_center)
    outer_support = _projective_ring_edge_quality(
        transform,
        DIAL_RADIUS,
        edge_distance=edge_distance,
        image_shape=image_shape,
        angular_step=angular_step,
    )
    inner_candidates = [
        (
            _projective_ring_edge_quality(
                transform,
                DIAL_RADIUS * fraction,
                edge_distance=edge_distance,
                image_shape=image_shape,
                angular_step=angular_step,
            ),
            fraction,
        )
        for fraction in np.arange(0.68, 0.881, 0.025)
    ]
    inner_support, inner_radius_fraction = max(inner_candidates)
    return (
        outer_support,
        inner_support,
        float(inner_radius_fraction),
        transform,
    )


def measure_projectively_concentric_ring_support(
    crop: np.ndarray,
    ellipse: Ellipse,
    physical_center: np.ndarray,
) -> tuple[float, float, float]:
    """Return support for distinct outer/inner rings under one camera pose."""

    outer, inner, fraction, _ = _projectively_concentric_ring_support(
        _dial_edge_distance(crop),
        (crop.shape[0], crop.shape[1]),
        ellipse,
        physical_center,
        angular_step=1,
    )
    return outer, inner, fraction


def refine_dial_ellipse_from_concentric_rings(
    crop: np.ndarray,
    pointer_mask: np.ndarray,
    pose_prior: Ellipse,
) -> Ellipse:
    """Recover an outer rim only when a second concentric ring confirms it."""

    edge_distance = _dial_edge_distance(crop)
    height, width = crop.shape[:2]
    image_scale = float(min(height, width))
    physical_center, has_visible_hub = _detect_visible_pointer_hub_evidence(
        crop,
        pointer_mask,
        pose_prior,
    )
    if not has_visible_hub:
        return pose_prior
    prior_center = np.asarray(pose_prior[0], dtype=np.float64)
    prior_minor, prior_major = sorted(pose_prior[1])
    prior_ratio = prior_minor / max(prior_major, 1.0)
    prior_major_orientation = (
        pose_prior[2]
        if pose_prior[1][0] >= pose_prior[1][1]
        else pose_prior[2] + 90.0
    ) % 180.0

    def ellipse_from_parameters(parameters: np.ndarray) -> Ellipse:
        center_x, center_y, ratio, orientation, major = parameters
        return (
            (float(center_x), float(center_y)),
            (float(major * ratio), float(major)),
            float((orientation - 90.0) % 180.0),
        )

    def evaluate(parameters: np.ndarray, *, angular_step: int) -> float:
        center_x, center_y, ratio, orientation, major = parameters
        ellipse = ellipse_from_parameters(parameters)
        orientation_delta = abs(
            (orientation - prior_major_orientation + 90.0) % 180.0 - 90.0
        )
        if (
            not 0.60 <= ratio <= 1.0
            or not 0.88 * prior_major <= major <= 1.25 * prior_major
            or orientation_delta > 24.0
            or np.linalg.norm(
                np.asarray((center_x, center_y), dtype=np.float64)
                - physical_center
            )
            > 0.15 * major
            or not ellipse_fits_crop(
                ellipse,
                (height, width),
                tolerance_fraction=0.06,
            )
        ):
            return -1.0
        try:
            outer, inner, _, _ = _projectively_concentric_ring_support(
                edge_distance,
                (height, width),
                ellipse,
                physical_center,
                angular_step=angular_step,
            )
        except (ValueError, np.linalg.LinAlgError):
            return -1.0
        return min(outer, inner) + 0.35 * (outer + inner)

    center_delta = physical_center - prior_center
    seeds: list[tuple[float, np.ndarray]] = []
    for interpolation in (0.25, 0.40, 0.55, 0.70):
        interpolated_center = prior_center + interpolation * center_delta
        for offset_x in (-0.01, 0.0, 0.01):
            for offset_y in (-0.01, 0.0, 0.01):
                for ratio_delta in (-0.09, -0.06, -0.03, 0.0, 0.03):
                    for orientation_delta in (-18, -12, -6, 0, 6, 12, 18):
                        for major_scale in (0.96, 1.02, 1.08, 1.14, 1.20):
                            parameters = np.asarray(
                                (
                                    interpolated_center[0]
                                    + offset_x * image_scale,
                                    interpolated_center[1]
                                    + offset_y * image_scale,
                                    prior_ratio + ratio_delta,
                                    prior_major_orientation + orientation_delta,
                                    prior_major * major_scale,
                                ),
                                dtype=np.float64,
                            )
                            score = evaluate(parameters, angular_step=3)
                            if score >= 0:
                                seeds.append((score, parameters))
    if not seeds:
        return pose_prior
    seeds.sort(key=lambda item: item[0], reverse=True)

    refined: list[tuple[float, np.ndarray]] = []
    step_levels = (
        (0.008 * image_scale, 0.008 * image_scale, 0.015, 2.0, 0.025 * prior_major),
        (0.003 * image_scale, 0.003 * image_scale, 0.006, 0.8, 0.010 * prior_major),
        (0.001 * image_scale, 0.001 * image_scale, 0.002, 0.3, 0.004 * prior_major),
    )
    for _, seed in seeds[:8]:
        parameters = seed.copy()
        best_score = evaluate(parameters, angular_step=1)
        for steps in step_levels:
            for _ in range(24):
                improved = False
                for index, step in enumerate(steps):
                    for direction in (-1.0, 1.0):
                        candidate = parameters.copy()
                        candidate[index] += direction * step
                        score = evaluate(candidate, angular_step=1)
                        if score > best_score + 1e-9:
                            parameters = candidate
                            best_score = score
                            improved = True
                if not improved:
                    break
        refined.append((best_score, parameters))

    best_score, best_parameters = max(refined, key=lambda item: item[0])
    prior_score = evaluate(
        np.asarray(
            (
                prior_center[0],
                prior_center[1],
                prior_ratio,
                prior_major_orientation,
                prior_major,
            ),
            dtype=np.float64,
        ),
        angular_step=1,
    )
    best_ellipse = ellipse_from_parameters(best_parameters)
    outer, inner, _, _ = _projectively_concentric_ring_support(
        edge_distance,
        (height, width),
        best_ellipse,
        physical_center,
        angular_step=1,
    )
    if (
        best_score < max(0.55, prior_score + 0.08)
        or outer < 0.30
        or inner < 0.25
    ):
        return pose_prior
    return best_ellipse


def refine_dial_ellipse_from_pose_prior(
    crop: np.ndarray,
    pointer_mask: np.ndarray,
    pose_prior: Ellipse,
) -> Ellipse:
    """Snap a coherent contour pose to one physical rim near the pointer hub.

    A broad edge search can combine fragments from several concentric rings and
    nearby housing edges.  The long contour supplies the camera-pose signature
    (axis ratio and orientation); this refinement may adjust its centre and size,
    but cannot jump to an unrelated, larger ellipse with a different pose.
    """

    edge_distance = _dial_edge_distance(crop)
    height, width = crop.shape[:2]
    scale = float(min(height, width))
    pointer_center, _ = pointer_center_and_tip(pointer_mask)
    (_, _), (prior_axis_a, prior_axis_b), prior_angle = pose_prior

    def evaluate(parameters: np.ndarray) -> tuple[float, float, Ellipse]:
        center_x, center_y, common_scale, anisotropy, angle_delta = parameters
        axis_a = prior_axis_a * common_scale * (1.0 + anisotropy)
        axis_b = prior_axis_b * common_scale * (1.0 - anisotropy)
        ellipse: Ellipse = (
            (float(center_x), float(center_y)),
            (float(axis_a), float(axis_b)),
            float((prior_angle + angle_delta) % 180.0),
        )
        minor, major = sorted((axis_a, axis_b))
        if (
            minor < 0.25 * scale
            or major > 1.30 * max(height, width)
            or minor / max(major, 1.0) < 0.30
            or np.linalg.norm(np.asarray((center_x, center_y)) - pointer_center)
            > 0.025 * scale
            or not ellipse_fits_crop(
                ellipse,
                (height, width),
                tolerance_fraction=0.06,
            )
        ):
            return -1.0, 0.0, ellipse
        score, visible_arc = _tight_ellipse_edge_score(
            ellipse,
            edge_distance=edge_distance,
            image_shape=(height, width),
        )
        return score, visible_arc, ellipse

    seeds: list[tuple[float, np.ndarray]] = []
    for center_x_offset in (-0.01, 0.0, 0.01):
        for center_y_offset in (-0.01, 0.0, 0.01):
            # Refinement may snap inward to a cleaner concentric ring, but it
            # must not expand a valid contour pose into the surrounding housing.
            for common_scale in np.arange(0.84, 1.021, 0.02):
                for anisotropy in (-0.04, -0.02, 0.0, 0.02, 0.04):
                    for angle_delta in range(-10, 11, 2):
                        parameters = np.asarray(
                            (
                                pointer_center[0] + center_x_offset * scale,
                                pointer_center[1] + center_y_offset * scale,
                                common_scale,
                                anisotropy,
                                float(angle_delta),
                            ),
                            dtype=np.float64,
                        )
                        score, visible_arc, _ = evaluate(parameters)
                        if visible_arc >= MINIMUM_DIAL_ELLIPSE_VISIBLE_ARC:
                            seeds.append((score, parameters))
    seeds.sort(key=lambda item: item[0], reverse=True)
    if not seeds:
        return pose_prior

    best_score, best_parameters = seeds[0]
    step_levels = (
        (0.005 * scale, 0.005 * scale, 0.010, 0.010, 1.5),
        (0.002 * scale, 0.002 * scale, 0.004, 0.004, 0.6),
    )
    for steps in step_levels:
        for _ in range(30):
            improved = False
            for index, step in enumerate(steps):
                for direction in (-1.0, 1.0):
                    candidate = best_parameters.copy()
                    candidate[index] += direction * step
                    score, visible_arc, _ = evaluate(candidate)
                    if (
                        visible_arc >= MINIMUM_DIAL_ELLIPSE_VISIBLE_ARC
                        and score > best_score + 1e-9
                    ):
                        best_score = score
                        best_parameters = candidate
                        improved = True
            if not improved:
                break

    prior_score, _, _ = evaluate(
        np.asarray(
            (
                pose_prior[0][0],
                pose_prior[0][1],
                1.0,
                0.0,
                0.0,
            ),
            dtype=np.float64,
        )
    )
    _, _, best_ellipse = evaluate(best_parameters)
    if best_score < max(0.18, prior_score + 0.015):
        return pose_prior
    return best_ellipse


def _detect_visible_pointer_hub_evidence(
    crop: np.ndarray,
    pointer_mask: np.ndarray,
    dial_ellipse: Ellipse,
) -> tuple[np.ndarray, bool]:
    """Prefer a visible metallic pointer hub over a nearby shadow blob.

    Pointer segmentation can merge the dark hub shadow into the pointer and
    shift its component centre.  A real exposed hub usually supplies a compact
    circular edge with a specular highlight; a shadow circle is dark inside.
    Only circles very near the segmentation centre are considered, and the
    segmentation centre remains the fallback for matte or hidden pivots.
    """

    segmentation_center, _ = pointer_center_and_tip(pointer_mask)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    minimum_dimension = min(gray.shape)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 1.0),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, round(minimum_dimension * 0.03)),
        param1=80,
        param2=20,
        minRadius=max(3, round(minimum_dimension * 0.008)),
        maxRadius=max(6, round(minimum_dimension * 0.06)),
    )
    if circles is None:
        return segmentation_center, False

    major_axis = max(dial_ellipse[1])
    candidates: list[tuple[float, np.ndarray]] = []
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    for center_x, center_y, radius in circles[0]:
        center = np.asarray((center_x, center_y), dtype=np.float64)
        distance = float(np.linalg.norm(center - segmentation_center))
        if distance > 0.035 * major_axis:
            continue
        interior = (
            np.square(xx - center_x) + np.square(yy - center_y)
            <= np.square(0.70 * radius)
        )
        values = gray[interior]
        if values.size < 10:
            continue
        bright_percentile = float(np.percentile(values, 90))
        if bright_percentile < 170.0:
            continue
        variation = float(np.std(values))
        score = bright_percentile + 0.5 * variation - 0.25 * distance
        candidates.append((score, center))
    if not candidates:
        return segmentation_center, False
    return max(candidates, key=lambda item: item[0])[1], True


def detect_visible_pointer_hub(
    crop: np.ndarray,
    pointer_mask: np.ndarray,
    dial_ellipse: Ellipse,
) -> np.ndarray:
    """Return the visible hub centre, or the segmentation fallback."""

    return _detect_visible_pointer_hub_evidence(
        crop,
        pointer_mask,
        dial_ellipse,
    )[0]


def _search_edge_supported_dial_ellipse(
    crop: np.ndarray,
    pointer_mask: np.ndarray,
    *,
    edge_distance: np.ndarray,
) -> Ellipse:
    """Recover a broken outer rim by optimizing only measurable edge support."""

    height, width = crop.shape[:2]
    scale = float(min(height, width))
    center_prior, _ = pointer_center_and_tip(pointer_mask)

    def evaluate(parameters: np.ndarray) -> tuple[float, float, Ellipse]:
        center_x, center_y, axis_a, axis_b, angle = parameters
        ellipse: Ellipse = (
            (float(center_x), float(center_y)),
            (float(axis_a), float(axis_b)),
            float(angle % 180.0),
        )
        minor, major = sorted((axis_a, axis_b))
        center_error = float(
            np.linalg.norm(np.asarray((center_x, center_y)) - center_prior) / scale
        )
        if (
            minor < 0.55 * scale
            or major > 1.20 * max(height, width)
            or minor / max(major, 1.0) < 0.60
            or center_error > 0.08
            or not ellipse_fits_crop(
                ellipse,
                (height, width),
                tolerance_fraction=0.10,
            )
        ):
            return -1.0, 0.0, ellipse
        support, visible_arc = ellipse_edge_support(
            ellipse,
            edge_distance=edge_distance,
            image_shape=(height, width),
        )
        return support, visible_arc, ellipse

    seeds: list[tuple[float, np.ndarray]] = []
    for center_x_offset in (-0.04, -0.02, 0.0, 0.02, 0.04):
        for center_y_offset in (-0.04, -0.02, 0.0, 0.02, 0.04):
            for minor_fraction in (0.62, 0.70, 0.78, 0.86):
                minor = minor_fraction * scale
                for axis_ratio in (0.72, 0.78, 0.84, 0.90, 0.96, 1.0):
                    major = minor / axis_ratio
                    for angle in range(0, 180, 15):
                        parameters = np.asarray(
                            (
                                center_prior[0] + center_x_offset * scale,
                                center_prior[1] + center_y_offset * scale,
                                minor,
                                major,
                                float(angle),
                            ),
                            dtype=np.float64,
                        )
                        support, visible_arc, _ = evaluate(parameters)
                        if visible_arc >= MINIMUM_DIAL_ELLIPSE_VISIBLE_ARC:
                            seeds.append((support, parameters))
    seeds.sort(key=lambda item: item[0], reverse=True)

    refined: list[tuple[float, float, Ellipse]] = []
    step_levels = (
        (0.015 * scale, 0.015 * scale, 0.025 * scale, 0.025 * scale, 5.0),
        (0.006 * scale, 0.006 * scale, 0.010 * scale, 0.010 * scale, 2.0),
        (0.002 * scale, 0.002 * scale, 0.004 * scale, 0.004 * scale, 0.7),
    )
    for initial_support, initial_parameters in seeds[:24]:
        parameters = initial_parameters.copy()
        best_support = initial_support
        for steps in step_levels:
            for _ in range(40):
                improved = False
                for index, step in enumerate(steps):
                    for direction in (-1.0, 1.0):
                        candidate = parameters.copy()
                        candidate[index] += direction * step
                        support, visible_arc, _ = evaluate(candidate)
                        if (
                            visible_arc >= MINIMUM_DIAL_ELLIPSE_VISIBLE_ARC
                            and support > best_support + 1e-9
                        ):
                            parameters = candidate
                            best_support = support
                            improved = True
                if not improved:
                    break
        support, visible_arc, ellipse = evaluate(parameters)
        if (
            support >= MINIMUM_DIAL_ELLIPSE_EDGE_SUPPORT
            and visible_arc >= MINIMUM_DIAL_ELLIPSE_VISIBLE_ARC
        ):
            refined.append((min(ellipse[1]), support, ellipse))
    if not refined:
        raise ValueError("Could not find an edge-supported dial ellipse")
    return max(refined, key=lambda item: (item[0], item[1]))[-1]


def detect_dial_ellipse(crop: np.ndarray, pointer_mask: np.ndarray) -> Ellipse:
    """Find and validate a physical dial rim before pose rectification."""
    center_prior, _ = pointer_center_and_tip(pointer_mask)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edge_distance = cv2.distanceTransform(
        (edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    height, width = crop.shape[:2]
    candidates: list[tuple[float, float, Ellipse]] = []
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
        # A tight detector crop may clip one side of an otherwise long physical
        # rim.  Keep its ratio/orientation as a pose prior; the later hub-centred
        # refinement must still fit the actual crop before it can be returned.
        if not ellipse_fits_crop(ellipse, crop.shape[:2], tolerance_fraction=0.10):
            continue
        points = contour[:, 0, :].astype(np.float64)
        contour_error = float(np.median(ellipse_residual(ellipse, points)))
        center_error = float(
            np.linalg.norm(np.asarray((center_x, center_y)) - center_prior)
            / max(height, width)
        )
        geometry_error = contour_error + center_error
        outer_rim_penalty = 0.08 * (
            1.0 - min(1.0, minor / max(1.0, min(height, width)))
        )
        candidates.append(
            (geometry_error + outer_rim_penalty, geometry_error, ellipse)
        )
    if not candidates:
        return _search_edge_supported_dial_ellipse(
            crop,
            pointer_mask,
            edge_distance=edge_distance,
        )
    _, geometry_error, ellipse = select_consensus_dial_ellipse(
        candidates,
        pointer_mask,
    )
    if geometry_error > 0.15:
        return _search_edge_supported_dial_ellipse(
            crop,
            pointer_mask,
            edge_distance=edge_distance,
        )
    refined = refine_dial_ellipse_center(crop, pointer_mask, ellipse)
    center_was_repaired = (
        np.linalg.norm(
            np.asarray(refined[0], dtype=np.float64)
            - np.asarray(ellipse[0], dtype=np.float64)
        )
        > 1.0
    )
    if (
        center_was_repaired
        and min(crop.shape[:2]) >= MINIMUM_PROJECTIVE_DIAL_DIMENSION
    ):
        paired_rim = refine_dial_ellipse_from_concentric_rings(
            crop,
            pointer_mask,
            ellipse,
        )
        if paired_rim != ellipse:
            return paired_rim
        return refine_dial_ellipse_from_pose_prior(crop, pointer_mask, refined)

    reliable: list[tuple[float, float, Ellipse]] = []
    for candidate_ellipse in (refined, ellipse, *(item[2] for item in candidates)):
        support, visible_arc = ellipse_edge_support(
            candidate_ellipse,
            edge_distance=edge_distance,
            image_shape=crop.shape[:2],
        )
        if (
            support >= MINIMUM_DIAL_ELLIPSE_EDGE_SUPPORT
            and visible_arc >= MINIMUM_DIAL_ELLIPSE_VISIBLE_ARC
        ):
            reliable.append((min(candidate_ellipse[1]), support, candidate_ellipse))
    if reliable:
        return max(reliable, key=lambda item: (item[0], item[1]))[-1]
    try:
        return _search_edge_supported_dial_ellipse(
            crop,
            pointer_mask,
            edge_distance=edge_distance,
        )
    except ValueError:
        return refined


def select_hub_consistent_ellipse_center(
    ellipse: Ellipse,
    pointer_center: np.ndarray,
    hub_circles: np.ndarray,
    crop_shape: tuple[int, int],
) -> Ellipse:
    """Repair a moderately shifted rim only when circular hub evidence is decisive."""
    ellipse_center = np.asarray(ellipse[0], dtype=np.float64)
    major_axis = max(ellipse[1])
    center_offset = float(np.linalg.norm(pointer_center - ellipse_center))
    if not 0.07 <= center_offset / max(major_axis, 1.0) <= 0.14:
        return ellipse
    circles = np.asarray(hub_circles, dtype=np.float64).reshape(-1, 3)
    if len(circles) == 0:
        return ellipse
    circle_centers = circles[:, :2]
    if float(np.min(np.linalg.norm(circle_centers - ellipse_center, axis=1))) <= (
        major_axis * 0.04
    ):
        return ellipse
    distances = np.linalg.norm(circle_centers - pointer_center, axis=1)
    selected_index = int(np.argmin(distances))
    if float(distances[selected_index]) > major_axis * 0.03:
        return ellipse
    selected_center = circle_centers[selected_index]
    refined: Ellipse = (
        (float(selected_center[0]), float(selected_center[1])),
        ellipse[1],
        ellipse[2],
    )
    if not ellipse_fits_crop(refined, crop_shape, tolerance_fraction=0.03):
        return ellipse
    return refined


def refine_dial_ellipse_center(
    crop: np.ndarray,
    pointer_mask: np.ndarray,
    ellipse: Ellipse,
) -> Ellipse:
    """Use a visible hub circle to correct a partial-rim ellipse centre."""
    pointer_center, _ = pointer_center_and_tip(pointer_mask)
    major_axis = max(ellipse[1])
    center_offset = float(
        np.linalg.norm(pointer_center - np.asarray(ellipse[0], dtype=np.float64))
    )
    if not 0.07 <= center_offset / max(major_axis, 1.0) <= 0.14:
        return ellipse
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    minimum_dimension = min(gray.shape)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 1.0),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, round(minimum_dimension * 0.03)),
        param1=80,
        param2=20,
        minRadius=max(3, round(minimum_dimension * 0.008)),
        maxRadius=max(6, round(minimum_dimension * 0.06)),
    )
    if circles is None:
        return ellipse
    return select_hub_consistent_ellipse_center(
        ellipse,
        pointer_center,
        circles[0],
        crop.shape[:2],
    )


def ellipse_fits_crop(
    ellipse: Ellipse,
    crop_shape: tuple[int, int],
    *,
    tolerance_fraction: float,
) -> bool:
    """Reject a rim that would require border replication outside its source crop."""
    (center_x, center_y), (axis_a, axis_b), angle = ellipse
    radians = math.radians(angle)
    half_width = 0.5 * math.sqrt(
        (axis_a * math.cos(radians)) ** 2
        + (axis_b * math.sin(radians)) ** 2
    )
    half_height = 0.5 * math.sqrt(
        (axis_a * math.sin(radians)) ** 2
        + (axis_b * math.cos(radians)) ** 2
    )
    height, width = crop_shape
    tolerance = max(height, width) * tolerance_fraction
    return (
        center_x - half_width >= -tolerance
        and center_y - half_height >= -tolerance
        and center_x + half_width <= width + tolerance
        and center_y + half_height <= height + tolerance
    )


def select_consensus_dial_ellipse(
    candidates: list[tuple[float, float, Ellipse]],
    pointer_mask: np.ndarray,
) -> tuple[float, float, Ellipse]:
    """Prefer a concentric rim whose rectified pointer angle agrees with its peers.

    Concentric face, glass and housing rims describe the same perspective plane, so
    their centres, axis ratios, orientations and rectified pointer angles should
    agree.  A large partial scene edge can produce a similar pointer angle while
    having a shifted centre, so establish geometric consensus before selecting the
    outermost rim in that group.
    """
    usable = [candidate for candidate in candidates if candidate[1] <= 0.15]
    if len(usable) < 3:
        return min(candidates, key=lambda candidate: candidate[0])

    angled: list[tuple[float, tuple[float, float, Ellipse]]] = []
    for candidate in usable:
        try:
            transform = ellipse_rectification(candidate[2])
            rectified_mask = cv2.warpAffine(
                pointer_mask,
                transform,
                (RECTIFIED_SIZE, RECTIFIED_SIZE),
                flags=cv2.INTER_NEAREST,
            )
            direction, _ = pointer_from_rectified_mask(rectified_mask)
        except ValueError:
            continue
        angle = angle_from_points((0.0, 0.0), direction)
        angled.append((angle, candidate))
    if len(angled) < 3:
        return min(candidates, key=lambda candidate: candidate[0])

    def ellipse_signature(
        candidate: tuple[float, float, Ellipse],
    ) -> tuple[np.ndarray, float, float, float]:
        center, (axis_a, axis_b), angle = candidate[2]
        minor, major = sorted((axis_a, axis_b))
        major_angle = angle if axis_a >= axis_b else angle + 90.0
        return (
            np.asarray(center, dtype=np.float64),
            minor / max(major, 1.0),
            major_angle % 180.0,
            major,
        )

    def perspective_matches(
        left: tuple[float, tuple[float, float, Ellipse]],
        right: tuple[float, tuple[float, float, Ellipse]],
    ) -> bool:
        left_center, left_ratio, left_orientation, left_major = ellipse_signature(
            left[1]
        )
        right_center, right_ratio, right_orientation, right_major = ellipse_signature(
            right[1]
        )
        center_distance = float(np.linalg.norm(left_center - right_center))
        orientation_delta = abs(
            (left_orientation - right_orientation + 90.0) % 180.0 - 90.0
        )
        pointer_delta = abs((left[0] - right[0] + 180.0) % 360.0 - 180.0)
        return (
            center_distance <= 0.065 * min(left_major, right_major)
            and abs(left_ratio - right_ratio) <= 0.06
            and orientation_delta <= 8.0
            and pointer_delta <= 5.0
        )

    geometry_groups = [
        [other for other in angled if perspective_matches(seed, other)]
        for seed in angled
    ]
    geometry_group = max(
        geometry_groups,
        key=lambda group: (
            len(group),
            -sum(item[1][0] for item in group),
        ),
    )
    if len(geometry_group) >= 2:
        angled = geometry_group

    def consensus_cost(item: tuple[float, tuple[float, float, Ellipse]]) -> float:
        angle = item[0]
        return sum(
            abs((angle - other_angle + 180.0) % 360.0 - 180.0)
            for other_angle, _ in angled
        )

    medoid_angle, _ = min(
        angled,
        key=lambda item: (consensus_cost(item), item[1][0]),
    )
    consensus_group = [
        candidate
        for angle, candidate in angled
        if abs((angle - medoid_angle + 180.0) % 360.0 - 180.0) <= 3.0
    ]
    return max(
        consensus_group,
        key=lambda candidate: (min(candidate[2][1]), -candidate[0]),
    )


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


def projected_concentric_ellipse(
    transform: np.ndarray,
    radius_fraction: float,
) -> Ellipse:
    """Project a rectified concentric circle back into the source image."""

    phases = np.deg2rad(np.arange(0, 360, 2, dtype=np.float32))
    center = RECTIFIED_SIZE / 2
    radius = DIAL_RADIUS * radius_fraction
    rectified_points = np.column_stack(
        (center + radius * np.cos(phases), center + radius * np.sin(phases))
    ).astype(np.float32)
    inverse = np.linalg.inv(np.asarray(transform, dtype=np.float64).reshape(3, 3))
    source_points = cv2.perspectiveTransform(
        rectified_points.reshape(-1, 1, 2),
        inverse,
    )
    raw = cv2.fitEllipse(source_points)
    return (
        (float(raw[0][0]), float(raw[0][1])),
        (float(raw[1][0]), float(raw[1][1])),
        float(raw[2]),
    )


def _warp_rectified(
    image: np.ndarray,
    transform: np.ndarray,
    *,
    flags: int,
) -> np.ndarray:
    if np.asarray(transform).shape == (3, 3):
        return cv2.warpPerspective(
            image,
            transform,
            (RECTIFIED_SIZE, RECTIFIED_SIZE),
            flags=flags,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return cv2.warpAffine(
        image,
        transform,
        (RECTIFIED_SIZE, RECTIFIED_SIZE),
        flags=flags,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _inverse_rectified_point(
    transform: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    raw_transform = np.asarray(transform, dtype=np.float64)
    raw_point = np.asarray(point, dtype=np.float64).reshape(2)
    if raw_transform.shape == (3, 3):
        inverse = np.linalg.inv(raw_transform)
        homogeneous = inverse @ np.append(raw_point, 1.0)
        return homogeneous[:2] / homogeneous[2]
    inverse = cv2.invertAffineTransform(raw_transform.reshape(2, 3))
    return inverse @ np.append(raw_point, 1.0)


def rectify_dial(
    crop: np.ndarray,
    ellipse: Ellipse,
    *,
    physical_center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    transform = (
        ellipse_rectification(ellipse)
        if physical_center is None
        else projective_ellipse_rectification(ellipse, physical_center)
    )
    rectified = _warp_rectified(crop, transform, flags=cv2.INTER_CUBIC)
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
    """Make angular crops across inner and outer numeric scale bands."""
    center = RECTIFIED_SIZE / 2
    crops: list[np.ndarray] = []
    phases: list[float] = []
    for radial_fraction in (0.45, 0.55):
        for phase in range(0, 360, SECTOR_STEP_DEGREES):
            radians = math.radians(phase)
            crop_center = (
                center
                + DIAL_RADIUS * radial_fraction * math.cos(radians),
                center
                + DIAL_RADIUS * radial_fraction * math.sin(radians),
            )
            sector = cv2.getRectSubPix(rectified, (100, 70), crop_center)
            sector = cv2.resize(
                sector, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
            )
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
    if len(labels) < MINIMUM_SCALE_LABELS:
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
                    if len(selected) < MINIMUM_SCALE_LABELS:
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
                    if len(final) < MINIMUM_SCALE_LABELS:
                        continue
                    final_phases = phases[final]
                    final_values = values[final]
                    if float(np.ptp(final_phases)) < 45.0:
                        continue
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
        for cluster in clusters:
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
    center = np.asarray((RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2))
    hub, tip = pointer_center_and_tip(mask)
    direction = tip - hub
    length = float(np.linalg.norm(direction))
    if length <= 0:
        raise ValueError("Pointer direction could not be estimated")
    direction /= length
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
        profile: GaugePipelineProfile | None = None,
    ):
        if units_per_major_segment <= 0:
            raise ValueError("units_per_major_segment must be positive")
        self.device = device
        self.units_per_major_segment = units_per_major_segment
        self.reading_interpreter = reading_interpreter
        self.profile = profile or get_gauge_pipeline_profile("448")
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
            image,
            imgsz=self.profile.detection_size,
            conf=0.20,
            device=self.device,
            verbose=False,
        )[0]
        candidates = self._result_candidates(detection, (0, 0), image.shape[:2])
        return tuple(deduplicate_dial_candidates(candidates))

    def detect_all_dial_candidates(self, image_path: Path) -> tuple[DialCandidate, ...]:
        """Collect every plausible dial across the full image and adaptive tiles."""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot decode image: {image_path}")
        detection = self.detector.predict(
            image,
            imgsz=self.profile.detection_size,
            conf=0.20,
            device=self.device,
            verbose=False,
        )[0]
        candidates = self._result_candidates(detection, (0, 0), image.shape[:2])
        for tiles in adaptive_tile_levels(image.shape[:2]):
            candidates.extend(self._tile_candidates(image, tiles))
        candidates.extend(detect_rectangular_gauge_candidates(image))
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
    ) -> PointerSegmentationTrace | None:
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
        padded_crops = [
            (
                pad_dial_crop_to_square(crop)
                if self.profile.preserve_canvas_aspect_ratio
                else (crop, (0, 0, crop.shape[1], crop.shape[0]))
            )
            for crop in crops
        ]
        inputs = [
            cv2.resize(
                padded_crop,
                (self.profile.dial_canvas_size, self.profile.dial_canvas_size),
            )
            for padded_crop, _ in padded_crops
        ]
        segmentations = self.segmenter.predict(
            inputs,
            imgsz=self.profile.segmentation_inference_size,
            conf=self.profile.segmentation_confidence,
            device=self.device,
            verbose=False,
        )
        traces: list[PointerSegmentationTrace] = []
        for candidate, crop, padded_crop, model_input, segmentation in zip(
            selected, crops, padded_crops, inputs, segmentations, strict=True
        ):
            if segmentation.masks is None or len(segmentation.masks.data) == 0:
                continue
            pointer_index = int(torch.argmax(segmentation.boxes.conf).item())
            pointer_confidence = float(
                segmentation.boxes.conf[pointer_index].detach().cpu()
            )
            model_output_mask = (
                segmentation.masks.data[pointer_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            square, content_bbox = padded_crop
            mask = cv2.resize(
                model_output_mask,
                (square.shape[1], square.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            content_x1, content_y1, content_x2, content_y2 = content_bbox
            mask = mask[content_y1:content_y2, content_x1:content_x2]
            if mask.shape != crop.shape[:2]:
                continue
            try:
                pointer_center, _ = pointer_center_and_tip(mask)
            except ValueError:
                continue
            normalized_center = pointer_center / np.asarray(
                (crop.shape[1], crop.shape[0]), dtype=np.float64
            )
            if np.any(normalized_center < 0.10) or np.any(normalized_center > 0.90):
                continue
            trace = PointerSegmentationTrace(
                candidate=candidate,
                mask=mask,
                confidence=pointer_confidence,
                method="model-segmentation",
                crop=crop,
                canvas=square,
                model_input=model_input,
                model_output_mask=model_output_mask,
                content_bbox=content_bbox,
            )
            traces.append(trace)
        if not traces:
            return None
        return select_pointer_segmentation_trace(traces, image.shape[:2])

    @staticmethod
    def _classical_candidates(
        image: np.ndarray, candidates: list[DialCandidate]
    ) -> PointerSegmentationTrace | None:
        traces: list[PointerSegmentationTrace] = []
        for candidate in sorted(
            candidates, key=lambda item: item.confidence, reverse=True
        )[:8]:
            x1, y1, x2, y2 = candidate.bbox
            crop = image[y1:y2, x1:x2]
            try:
                pointer = detect_classical_pointer(crop)
            except ValueError:
                continue
            trace = PointerSegmentationTrace(
                candidate=candidate,
                mask=pointer.mask,
                confidence=pointer.confidence,
                method="colored-hub+line-segment",
                crop=crop,
                canvas=crop,
                model_input=None,
                model_output_mask=None,
                content_bbox=(0, 0, crop.shape[1], crop.shape[0]),
            )
            traces.append(trace)
        if not traces:
            return None
        return select_pointer_segmentation_trace(traces, image.shape[:2])

    def _tile_candidates(
        self,
        image: np.ndarray,
        tiles: list[tuple[int, int, int, int]],
    ) -> list[DialCandidate]:
        crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
        detections = self.detector.predict(
            crops,
            imgsz=self.profile.detection_size,
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

    def _write_processing_stages(
        self,
        writer: ProcessingStageWriter,
        group: str,
        *,
        image: np.ndarray,
        detail_image: np.ndarray,
        candidates: list[DialCandidate],
        candidate: DialCandidate,
        detail_candidate: DialCandidate,
        segmentation_trace: PointerSegmentationTrace,
        geometry_crop: np.ndarray,
        geometry_mask: np.ndarray,
    ) -> None:
        writer.write(
            group,
            "dial-detection",
            draw_dial_candidates(
                image,
                [(item.bbox, item.confidence) for item in candidates],
                selected_bbox=candidate.bbox,
            ),
            title_zh="表盘候选检测结果",
            operation="draw_detection_candidates",
            source_stage="analysis-image",
            preserves_aspect_ratio=True,
            note_zh="橙框为候选，绿框和星号为本次读取采用的候选。",
        )
        writer.write(
            group,
            "segmentation-crop",
            segmentation_trace.crop,
            title_zh="指针分割前的表盘裁剪",
            operation=f"crop_bbox={list(segmentation_trace.candidate.bbox)}",
            source_stage=(
                "detail-image"
                if self.profile.segment_on_high_resolution_detail
                and detail_image is not image
                else "analysis-image"
            ),
            preserves_aspect_ratio=True,
            note_zh="只做矩形裁剪，保留裁剪区域的原始像素比例。",
        )
        model_segmentation_used = segmentation_trace.model_input is not None
        padding_enabled = (
            self.profile.preserve_canvas_aspect_ratio and model_segmentation_used
        )
        writer.write(
            group,
            "segmentation-canvas",
            segmentation_trace.canvas,
            title_zh=(
                "保持比例补边后的分割画布" if padding_enabled else "未补边的分割画布"
            ),
            operation=(
                f"replicate_pad_content_bbox={list(segmentation_trace.content_bbox)}"
                if padding_enabled
                else "identity_no_padding"
            ),
            source_stage="segmentation-crop",
            preserves_aspect_ratio=True,
            note_zh=(
                "补边只增加边缘像素，不拉伸原裁剪内容。"
                if padding_enabled
                else (
                    "经典几何回退直接使用原裁剪，不经过模型补边。"
                    if not model_segmentation_used
                    else "当前 profile 未启用补边，因此尺寸与裁剪图相同。"
                )
            ),
        )
        if segmentation_trace.model_input is not None:
            canvas_height, canvas_width = segmentation_trace.canvas.shape[:2]
            preserves_ratio = canvas_width == canvas_height
            writer.write(
                group,
                "segmentation-model-input",
                segmentation_trace.model_input,
                title_zh="指针模型实际输入图",
                operation=(
                    f"resize_{canvas_width}x{canvas_height}_to_"
                    f"{self.profile.dial_canvas_size}x{self.profile.dial_canvas_size}"
                ),
                source_stage="segmentation-canvas",
                preserves_aspect_ratio=preserves_ratio,
                note_zh=(
                    "画布为正方形，缩放未改变宽高比。"
                    if preserves_ratio
                    else "非正方形画布被缩放为正方形，会改变几何比例。"
                ),
            )
        if segmentation_trace.model_output_mask is not None:
            writer.write(
                group,
                "segmentation-model-mask",
                segmentation_trace.model_output_mask,
                title_zh="指针模型原始输出掩膜",
                operation="model_segmentation_mask",
                source_stage="segmentation-model-input",
                preserves_aspect_ratio=True,
                note_zh="保存模型直接返回的掩膜尺寸，未回映射。",
            )
        writer.write(
            group,
            "segmentation-restored-mask",
            segmentation_trace.mask,
            title_zh=(
                "回映射到分割裁剪的指针掩膜"
                if segmentation_trace.model_output_mask is not None
                else "经典几何回退生成的指针掩膜"
            ),
            operation=(
                "nearest_resize+remove_padding"
                if segmentation_trace.model_output_mask is not None
                else "classical_pointer_geometry_mask"
            ),
            source_stage=(
                "segmentation-model-mask"
                if segmentation_trace.model_output_mask is not None
                else "segmentation-crop"
            ),
            preserves_aspect_ratio=True,
            note_zh=(
                "掩膜恢复到分割裁剪的宽高，并移除可能存在的补边。"
                if segmentation_trace.model_output_mask is not None
                else "该路径未调用分割模型，掩膜直接来自经典线段几何。"
            ),
        )
        if detail_image is not image:
            writer.write(
                group,
                "detail-detection",
                draw_dial_candidates(
                    detail_image,
                    [(detail_candidate.bbox, detail_candidate.confidence)],
                    selected_bbox=detail_candidate.bbox,
                ),
                title_zh="映射到高分辨率图的表盘框",
                operation="map_bbox_analysis_to_detail",
                source_stage="detail-image",
                preserves_aspect_ratio=True,
                note_zh="检测坐标按低分辨率图与细节图的比例映射。",
            )
        writer.write(
            group,
            "geometry-crop",
            geometry_crop,
            title_zh="用于椭圆与刻度几何的表盘裁剪",
            operation=f"crop_bbox={list(detail_candidate.bbox)}",
            source_stage="detail-image"
            if detail_image is not image
            else "analysis-image",
            preserves_aspect_ratio=True,
            note_zh="高分辨率 profile 使用原图细节裁剪；默认 profile 使用分析图裁剪。",
        )
        geometry_mapping_changed = geometry_mask.shape != segmentation_trace.mask.shape
        writer.write(
            group,
            "geometry-mask",
            geometry_mask,
            title_zh="用于几何读取的指针掩膜",
            operation=(
                "nearest_map_segmentation_mask_to_detail_crop"
                if geometry_mapping_changed
                else "identity_from_restored_mask"
            ),
            source_stage="segmentation-restored-mask",
            preserves_aspect_ratio=True,
            note_zh=(
                "掩膜按表盘框坐标映射到高分辨率裁剪。"
                if geometry_mapping_changed
                else "分割与几何读取使用相同裁剪尺寸，未再次缩放。"
            ),
        )

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

    def read(
        self,
        image_path: Path,
        *,
        visible_text_context: str = "",
        detail_image_path: Path | None = None,
        stage_writer: ProcessingStageWriter | None = None,
        stage_group: str = "full-image",
    ) -> GaugeResult:
        preprocess_start = time.perf_counter_ns()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot decode image: {image_path}")
        detail_image = image
        if (
            self.profile.use_high_resolution_detail
            and detail_image_path is not None
            and detail_image_path != image_path
        ):
            detail_image = cv2.imread(str(detail_image_path), cv2.IMREAD_COLOR)
        if detail_image is None:
            raise FileNotFoundError(f"Cannot decode detail image: {detail_image_path}")
        analysis_shape = image.shape[:2]
        detail_shape = detail_image.shape[:2]
        preprocess_end = time.perf_counter_ns()

        def segment_at_detail(
            analysis_candidates: list[DialCandidate],
            *,
            classical: bool = False,
        ) -> (
            tuple[
                DialCandidate,
                DialCandidate,
                np.ndarray,
                PointerSegmentationTrace,
            ]
            | None
        ):
            detail_candidates = [
                DialCandidate(
                    map_bbox_between_images(
                        candidate.bbox,
                        source_shape=analysis_shape,
                        target_shape=detail_shape,
                    ),
                    candidate.confidence,
                )
                for candidate in analysis_candidates
            ]
            use_detail_segmentation = (
                self.profile.segment_on_high_resolution_detail
                and detail_image is not image
            )
            segmentation_image = detail_image if use_detail_segmentation else image
            segmentation_candidates = (
                detail_candidates if use_detail_segmentation else analysis_candidates
            )
            detail_result = (
                self._classical_candidates(segmentation_image, segmentation_candidates)
                if classical
                else self._segment_candidates(
                    segmentation_image, segmentation_candidates
                )
            )
            if detail_result is None:
                return None
            selected_segmentation = detail_result.candidate
            mask = detail_result.mask
            try:
                selected_index = segmentation_candidates.index(selected_segmentation)
            except ValueError:
                return None
            selected_detail = detail_candidates[selected_index]
            if not use_detail_segmentation and detail_shape != analysis_shape:
                detail_x1, detail_y1, detail_x2, detail_y2 = selected_detail.bbox
                mask = cv2.resize(
                    mask,
                    (detail_x2 - detail_x1, detail_y2 - detail_y1),
                    interpolation=cv2.INTER_NEAREST,
                )
            selected_detail, mask = expand_candidate_and_pointer_mask(
                selected_detail,
                mask,
                image_shape=detail_shape,
                margin_fraction=self.profile.geometry_crop_margin_fraction,
            )
            return (
                analysis_candidates[selected_index],
                selected_detail,
                mask,
                detail_result,
            )

        synchronize(self.device)
        inference_start = time.perf_counter_ns()
        image_visible_text = self._full_image_visible_text(image_path, image)
        visible_text_context = f"{visible_text_context} {image_visible_text}".strip()
        detection = self.detector.predict(
            image,
            imgsz=self.profile.detection_size,
            conf=0.25,
            device=self.device,
            verbose=False,
        )[0]
        has_raw_detection = detection.boxes is not None and len(detection.boxes) > 0
        candidates = self._result_candidates(detection, (0, 0), image.shape[:2])
        segmented = segment_at_detail(candidates)
        if segmented is None:
            segmented = segment_at_detail(candidates, classical=True)
        if segmented is None and not has_raw_detection:
            try:
                circle_detection = detect_circular_gauge(image)
                circle_candidate = DialCandidate(
                    circle_detection.bbox, circle_detection.confidence
                )
                if is_plausible_dial_bbox(circle_candidate.bbox, image.shape[:2]):
                    candidates.append(circle_candidate)
                    segmented = segment_at_detail([circle_candidate])
                    if segmented is None:
                        segmented = segment_at_detail(
                            [circle_candidate], classical=True
                        )
            except ValueError:
                pass
        if segmented is None:
            for tiles in adaptive_tile_levels(image.shape[:2]):
                tiled_candidates = self._tile_candidates(image, tiles)
                candidates.extend(tiled_candidates)
                segmented = segment_at_detail(tiled_candidates)
                if segmented is None:
                    segmented = segment_at_detail(tiled_candidates, classical=True)
                if segmented is not None:
                    break
        if segmented is None:
            synchronize(self.device)
            inference_end = time.perf_counter_ns()
            candidate = max(candidates, key=lambda item: item.confidence, default=None)
            if stage_writer is not None:
                stage_writer.write(
                    stage_group,
                    "dial-detection",
                    draw_dial_candidates(
                        image,
                        [(item.bbox, item.confidence) for item in candidates],
                        selected_bbox=candidate.bbox if candidate is not None else None,
                    ),
                    title_zh="表盘候选检测结果",
                    operation="draw_detection_candidates",
                    source_stage="analysis-image",
                    preserves_aspect_ratio=True,
                    note_zh="橙框为候选，绿框和星号为最终保留候选。",
                )
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
        candidate, detail_candidate, mask, segmentation_trace = segmented
        pointer_confidence = segmentation_trace.confidence
        pointer_method = segmentation_trace.method
        bbox = candidate.bbox
        detection_confidence = candidate.confidence
        detail_x1, detail_y1, detail_x2, detail_y2 = detail_candidate.bbox
        crop = detail_image[detail_y1:detail_y2, detail_x1:detail_x2]
        if stage_writer is not None:
            self._write_processing_stages(
                stage_writer,
                stage_group,
                image=image,
                detail_image=detail_image,
                candidates=candidates,
                candidate=candidate,
                detail_candidate=detail_candidate,
                segmentation_trace=segmentation_trace,
                geometry_crop=crop,
                geometry_mask=mask,
            )

        def analysis_point(detail_point: np.ndarray) -> np.ndarray:
            return np.asarray(
                (
                    detail_point[0] * image.shape[1] / detail_image.shape[1],
                    detail_point[1] * image.shape[0] / detail_image.shape[0],
                ),
                dtype=np.float64,
            )

        try:
            ellipse = detect_dial_ellipse(crop, mask)
            physical_center, has_visible_hub = _detect_visible_pointer_hub_evidence(
                crop,
                mask,
                ellipse,
            )
            ellipse_edge_support_fraction, ellipse_visible_arc_fraction = (
                measure_dial_ellipse_quality(crop, ellipse)
            )
            paired_ring_evidence = False
            projective_transform: np.ndarray | None = None
            inner_ring_support = 0.0
            inner_radius_fraction = 0.0
            try:
                (
                    outer_ring_support,
                    inner_ring_support,
                    inner_radius_fraction,
                    candidate_projective_transform,
                ) = _projectively_concentric_ring_support(
                    _dial_edge_distance(crop),
                    (crop.shape[0], crop.shape[1]),
                    ellipse,
                    physical_center,
                    angular_step=1,
                )
                paired_ring_evidence = (
                    min(crop.shape[:2]) >= MINIMUM_PROJECTIVE_DIAL_DIMENSION
                    and has_visible_hub
                    and outer_ring_support >= 0.30
                    and inner_ring_support >= 0.25
                )
                center_offset_fraction = float(
                    np.linalg.norm(np.asarray(ellipse[0]) - physical_center)
                    / max(ellipse[1])
                )
                if paired_ring_evidence and center_offset_fraction >= 0.015:
                    projective_transform = candidate_projective_transform
            except (ValueError, np.linalg.LinAlgError):
                pass
            ellipse_evidence = (
                "透视同心双环路径"
                if paired_ring_evidence
                else (
                    "强边缘路径"
                    if ellipse_edge_support_fraction
                    >= MINIMUM_DIAL_ELLIPSE_EDGE_SUPPORT
                    else "轮廓姿态约束路径"
                )
            )
            if stage_writer is not None:
                ellipse_overlay = draw_ellipse(crop, ellipse)
                if paired_ring_evidence and projective_transform is not None:
                    inner_ellipse = projected_concentric_ellipse(
                        projective_transform,
                        inner_radius_fraction,
                    )
                    cv2.ellipse(
                        ellipse_overlay,
                        inner_ellipse,
                        (255, 0, 255),
                        max(2, round(min(crop.shape[:2]) * 0.003)),
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        ellipse_overlay,
                        tuple(np.rint(physical_center).astype(int)),
                        max(4, round(min(crop.shape[:2]) * 0.008)),
                        (255, 120, 0),
                        3,
                        cv2.LINE_AA,
                    )
                stage_writer.write(
                    stage_group,
                    "fitted-ellipse",
                    ellipse_overlay,
                    title_zh=(
                        "表盘透视同心双环拟合结果"
                        if paired_ring_evidence
                        else "表盘椭圆拟合结果"
                    ),
                    operation=(
                        "projectively_concentric_outer+inner_ring_overlay"
                        if paired_ring_evidence
                        else "edge_ellipse_fit_overlay"
                    ),
                    source_stage="geometry-crop",
                    preserves_aspect_ratio=True,
                    note_zh=(
                        "底图像素尺寸未改变。绿色为物理外圈；"
                        + (
                            "紫色为经同一透视姿态验证的内圈，蓝点为指针轴心；"
                            f"内圈支持 {inner_ring_support:.1%}，"
                            f"校正后半径比例 {inner_radius_fraction:.3f}。"
                            if paired_ring_evidence
                            else "当前没有足够的独立内圈证据；"
                        )
                        +
                        f"可见圆弧边缘贴合率 "
                        f"{ellipse_edge_support_fraction:.1%}，"
                        f"可见圆弧比例 {ellipse_visible_arc_fraction:.1%}；"
                        f"采用{ellipse_evidence}。"
                    ),
                )
            rectified, transform = rectify_dial(
                crop,
                ellipse,
                physical_center=(
                    physical_center if projective_transform is not None else None
                ),
            )
            rectified_mask = _warp_rectified(
                mask, transform, flags=cv2.INTER_NEAREST
            )
            rectified_center = np.asarray(
                (RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2), dtype=np.float64
            )
            direction, rectified_tip = pointer_from_rectified_mask(rectified_mask)
            full_dial_recognition = self.ocr(rectified)
            visible_text = " ".join(
                (
                    visible_text_context,
                    visible_text_from_ocr_result(full_dial_recognition),
                )
            ).strip()
            labels = recognize_numeric_sectors(self.ocr, rectified)
            color_result: ColorScaleResult | None = None
            color_result_source = "geometry"
            selected_color_rectified: np.ndarray | None = None
            selected_color_center: np.ndarray | None = None
            selected_color_tip: np.ndarray | None = None
            consolidated_label_count = len(
                consolidate_duplicate_labels(
                    labels,
                    (
                        (RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2),
                        (2 * DIAL_RADIUS, 2 * DIAL_RADIUS),
                        0.0,
                    ),
                )
            )
            try:
                color_result = read_color_segments(
                    rectified, direction, self.units_per_major_segment
                )
            except ValueError:
                if consolidated_label_count < MINIMUM_SCALE_LABELS:
                    labels = recognize_full_dial(
                        self.ocr, rectified, full_dial_recognition
                    )
            if self.profile.use_high_resolution_detail and detail_shape != analysis_shape:
                try:
                    analysis_ellipse = detect_dial_ellipse(
                        segmentation_trace.crop,
                        segmentation_trace.mask,
                    )
                    (
                        analysis_physical_center,
                        analysis_has_visible_hub,
                    ) = _detect_visible_pointer_hub_evidence(
                        segmentation_trace.crop,
                        segmentation_trace.mask,
                        analysis_ellipse,
                    )
                    analysis_projective = False
                    try:
                        analysis_outer, analysis_inner, _, _ = (
                            _projectively_concentric_ring_support(
                                _dial_edge_distance(segmentation_trace.crop),
                                (
                                    segmentation_trace.crop.shape[0],
                                    segmentation_trace.crop.shape[1],
                                ),
                                analysis_ellipse,
                                analysis_physical_center,
                                angular_step=2,
                            )
                        )
                        analysis_projective = (
                            min(segmentation_trace.crop.shape[:2])
                            >= MINIMUM_PROJECTIVE_DIAL_DIMENSION
                            and analysis_has_visible_hub
                            and analysis_outer >= 0.30
                            and analysis_inner >= 0.25
                            and np.linalg.norm(
                                np.asarray(analysis_ellipse[0])
                                - analysis_physical_center
                            )
                            / max(analysis_ellipse[1])
                            >= 0.015
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        pass
                    analysis_rectified, analysis_transform = rectify_dial(
                        segmentation_trace.crop,
                        analysis_ellipse,
                        physical_center=(
                            analysis_physical_center if analysis_projective else None
                        ),
                    )
                    analysis_rectified_mask = _warp_rectified(
                        segmentation_trace.mask,
                        analysis_transform,
                        flags=cv2.INTER_NEAREST,
                    )
                    analysis_center = np.asarray(
                        (RECTIFIED_SIZE / 2, RECTIFIED_SIZE / 2), dtype=np.float64
                    )
                    analysis_direction, analysis_tip = pointer_from_rectified_mask(
                        analysis_rectified_mask
                    )
                    analysis_color_result = read_color_segments(
                        analysis_rectified,
                        analysis_direction,
                        self.units_per_major_segment,
                    )
                except ValueError:
                    pass
                else:
                    if (
                        color_result is None
                        or analysis_color_result.confidence > color_result.confidence
                    ):
                        color_result = analysis_color_result
                        color_result_source = "analysis"
                        selected_color_rectified = analysis_rectified
                        selected_color_center = analysis_center
                        selected_color_tip = analysis_tip
            self.last_rectified = rectified
            self.last_ring = unwrap_scale_ring(rectified)
            if stage_writer is not None:
                stage_writer.write(
                    stage_group,
                    "rectified-dial",
                    rectified,
                    title_zh="椭圆校正后的表盘",
                    operation=(
                        f"projective_warp_to_{RECTIFIED_SIZE}x{RECTIFIED_SIZE}"
                        if projective_transform is not None
                        else f"affine_warp_to_{RECTIFIED_SIZE}x{RECTIFIED_SIZE}"
                    ),
                    source_stage="geometry-crop",
                    preserves_aspect_ratio=False,
                    note_zh=(
                        "用同一透视变换将内外物理圆环和指针轴心校正到"
                        "同心正视坐标。"
                        if projective_transform is not None
                        else "将斜拍椭圆表盘校正到固定正方形坐标系。"
                    ),
                )
                stage_writer.write(
                    stage_group,
                    "rectified-pointer-mask",
                    rectified_mask,
                    title_zh="校正后的指针掩膜",
                    operation=(
                        f"projective_mask_warp_to_{RECTIFIED_SIZE}x{RECTIFIED_SIZE}"
                        if projective_transform is not None
                        else f"affine_mask_warp_to_{RECTIFIED_SIZE}x{RECTIFIED_SIZE}"
                    ),
                    source_stage="geometry-mask",
                    preserves_aspect_ratio=False,
                    note_zh="使用与表盘相同的校正矩阵，最近邻插值。",
                )
                stage_writer.write(
                    stage_group,
                    "rectified-pointer-geometry",
                    draw_pointer_geometry(
                        rectified,
                        rectified_center,
                        rectified_tip,
                    ),
                    title_zh="透视校正后采用的最终指针几何",
                    operation="rectified_hub_to_reading_end",
                    source_stage="rectified-pointer-mask",
                    preserves_aspect_ratio=True,
                    note_zh="红箭头从轴心指向最终采用的读数端。",
                )
                stage_writer.write(
                    stage_group,
                    "unwrapped-scale-ring",
                    self.last_ring,
                    title_zh="展开后的刻度环",
                    operation="polar_scale_ring_unwrap",
                    source_stage="rectified-dial",
                    preserves_aspect_ratio=False,
                    note_zh="用于观察刻度与 OCR 输入；展开会改变几何比例。",
                )
                if (
                    selected_color_rectified is not None
                    and selected_color_center is not None
                    and selected_color_tip is not None
                ):
                    stage_writer.write(
                        stage_group,
                        "selected-color-scale-dial",
                        selected_color_rectified,
                        title_zh="最终采用的低分辨率彩色刻度校正图",
                        operation="analysis_crop_edge_ellipse_affine_rectification",
                        source_stage="segmentation-restored-mask",
                        preserves_aspect_ratio=False,
                        note_zh=(
                            "高、低分辨率彩色刻度分别复算；此图的颜色分段"
                            "几何一致性更高，因此只用于彩色刻度换算。"
                        ),
                    )
                    stage_writer.write(
                        stage_group,
                        "selected-color-scale-pointer-geometry",
                        draw_pointer_geometry(
                            selected_color_rectified,
                            selected_color_center,
                            selected_color_tip,
                        ),
                        title_zh="彩色刻度分支采用的指针几何",
                        operation="analysis_color_scale_geometry_consensus",
                        source_stage="selected-color-scale-dial",
                        preserves_aspect_ratio=True,
                        note_zh="该箭头用于最终彩色分段读数，并非额外仪表通道。",
                    )
        except ValueError as error:
            synchronize(self.device)
            inference_end = time.perf_counter_ns()
            fallback_center, fallback_tip = pointer_center_and_tip(mask)
            if stage_writer is not None:
                stage_writer.write(
                    stage_group,
                    "unrectified-pointer-geometry",
                    draw_pointer_geometry(crop, fallback_center, fallback_tip),
                    title_zh="未做椭圆校正的最终指针几何",
                    operation="pointer_mask_hub_to_reading_end",
                    source_stage="geometry-mask",
                    preserves_aspect_ratio=True,
                    note_zh=(
                        "表盘椭圆未通过完整性检查；红箭头仍展示最终采用的"
                        "轴心到读数端方向。"
                    ),
                )
            detail_global_center = fallback_center + np.asarray((detail_x1, detail_y1))
            detail_global_tip = fallback_tip + np.asarray((detail_x1, detail_y1))
            global_center = analysis_point(detail_global_center)
            global_tip = analysis_point(detail_global_tip)
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
        crop_center = _inverse_rectified_point(transform, rectified_center)
        crop_tip = _inverse_rectified_point(transform, rectified_tip)
        detail_global_center = crop_center + np.asarray((detail_x1, detail_y1))
        detail_global_tip = crop_tip + np.asarray((detail_x1, detail_y1))
        global_center = analysis_point(detail_global_center)
        global_tip = analysis_point(detail_global_tip)
        angle = angle_from_points(rectified_center, rectified_tip)
        try:
            fit, inliers, rejected = robust_scale_fit(labels, circle)
            if color_result is not None and len(inliers) <= MINIMUM_SCALE_LABELS:
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
                    center_method=(
                        f"{pointer_method}+edge-ellipse+{color_result_source}"
                        "-color-segment-scale"
                    ),
                    timings=StageTimings(
                        (preprocess_end - preprocess_start) / 1e6,
                        (inference_end - inference_start) / 1e6,
                        (postprocess_end - postprocess_start) / 1e6,
                    ),
                )
                return self._interpret_result(result, visible_text)
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
