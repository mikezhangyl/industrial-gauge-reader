"""Lossless, auditable image artifacts for each gauge-processing stage."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ProcessingStageArtifact:
    """One exact-size image produced or consumed by the processing pipeline."""

    group: str
    stage_id: str
    title_zh: str
    path: str
    dimensions: tuple[int, int]
    aspect_ratio: float
    operation: str
    source_stage: str | None
    preserves_aspect_ratio: bool | None
    note_zh: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dimensions"] = list(self.dimensions)
        return payload


class ProcessingStageWriter:
    """Write stage images beside a batch report without changing pipeline arrays."""

    def __init__(self, report_directory: Path, relative_directory: Path):
        if relative_directory.is_absolute() or ".." in relative_directory.parts:
            raise ValueError("Stage output directory must be report-relative")
        self.report_directory = report_directory.resolve()
        self.relative_directory = relative_directory
        self.output_directory = self.report_directory / relative_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._group_counts: dict[str, int] = defaultdict(int)
        self._artifacts: list[ProcessingStageArtifact] = []

    def write_oriented_source(
        self,
        source_path: Path,
        *,
        group: str = "batch-input",
    ) -> ProcessingStageArtifact:
        """Persist EXIF-oriented source pixels without resizing or cropping."""

        with Image.open(source_path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.asarray(oriented)
        return self.write(
            group,
            "oriented-source",
            rgb[:, :, ::-1],
            title_zh="EXIF 方向校正后的原图",
            operation="exif_transpose+rgb",
            source_stage=None,
            preserves_aspect_ratio=True,
            note_zh="保持方向校正后的完整像素尺寸；未裁剪、未缩放。",
        )

    def write(
        self,
        group: str,
        stage_id: str,
        image: np.ndarray,
        *,
        title_zh: str,
        operation: str,
        source_stage: str | None,
        preserves_aspect_ratio: bool | None,
        note_zh: str | None = None,
    ) -> ProcessingStageArtifact:
        """Write an ndarray as lossless PNG and record its exact pixel geometry."""

        if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("Processing stage image must have positive dimensions")
        group_key = _safe_component(group)
        stage_key = _safe_component(stage_id)
        self._group_counts[group_key] += 1
        sequence = self._group_counts[group_key]
        relative_path = (
            self.relative_directory / group_key / f"{sequence:02d}-{stage_key}.png"
        )
        destination = self.report_directory / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        writable = _png_writable_image(image)
        if not cv2.imwrite(str(destination), writable):
            raise RuntimeError(f"Failed to write processing stage image: {destination}")
        height, width = writable.shape[:2]
        artifact = ProcessingStageArtifact(
            group=group,
            stage_id=stage_id,
            title_zh=title_zh,
            path=relative_path.as_posix(),
            dimensions=(width, height),
            aspect_ratio=round(width / height, 8),
            operation=operation,
            source_stage=source_stage,
            preserves_aspect_ratio=preserves_aspect_ratio,
            note_zh=note_zh,
        )
        self._artifacts.append(artifact)
        return artifact

    def records(self) -> list[dict[str, Any]]:
        return [artifact.to_dict() for artifact in self._artifacts]


def draw_dial_candidates(
    image: np.ndarray,
    candidates: list[tuple[tuple[int, int, int, int], float]],
    *,
    selected_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Overlay all dial candidates while making the selected candidate explicit."""

    overlay = image.copy()
    line_width = max(2, round(max(image.shape[:2]) / 500))
    for index, (bbox, _confidence) in enumerate(candidates, start=1):
        x1, y1, x2, y2 = bbox
        selected = selected_bbox == bbox
        color = (40, 180, 40) if selected else (0, 180, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, line_width)
        cv2.putText(
            overlay,
            f"{index}{'*' if selected else ''}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, line_width * 0.22),
            color,
            max(1, line_width // 2),
            cv2.LINE_AA,
        )
    return overlay


def draw_ellipse(image: np.ndarray, ellipse: Any) -> np.ndarray:
    """Overlay the fitted dial ellipse without changing the source image geometry."""

    overlay = image.copy()
    center, axes, angle = ellipse
    cv2.ellipse(
        overlay,
        (
            (float(center[0]), float(center[1])),
            (float(axes[0]), float(axes[1])),
            float(angle),
        ),
        (40, 180, 40),
        max(2, round(max(image.shape[:2]) / 400)),
        cv2.LINE_AA,
    )
    return overlay


def draw_pointer_geometry(
    image: np.ndarray,
    center: np.ndarray | tuple[float, float],
    tip: np.ndarray | tuple[float, float],
) -> np.ndarray:
    """Overlay the accepted hub-to-reading-end direction on its source crop."""
    overlay = image.copy()
    center_point = tuple(np.rint(center).astype(int))
    tip_point = tuple(np.rint(tip).astype(int))
    line_width = max(2, round(max(image.shape[:2]) / 180))
    cv2.circle(
        overlay,
        center_point,
        max(4, line_width * 2),
        (40, 180, 40),
        line_width,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(
        overlay,
        center_point,
        tip_point,
        (0, 0, 255),
        line_width,
        cv2.LINE_AA,
        tipLength=0.08,
    )
    return overlay


def _png_writable_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255
    if image.dtype == np.uint8:
        if image.ndim == 2 and image.max(initial=0) <= 1:
            return image * 255
        return image
    if np.issubdtype(image.dtype, np.floating):
        finite = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        if finite.max(initial=0.0) <= 1.0 and finite.min(initial=0.0) >= 0.0:
            return np.rint(finite * 255).astype(np.uint8)
        return np.clip(finite, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return safe or "stage"
