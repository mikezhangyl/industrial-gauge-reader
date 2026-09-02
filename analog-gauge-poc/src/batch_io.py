"""Deterministic folder input, image normalization, and report previews."""

from __future__ import annotations

import base64
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)
NORMALIZED_MAX_EDGE = 1920
PREVIEW_SIZE = (960, 640)


@dataclass(frozen=True)
class NormalizedBatchImage:
    source_path: Path
    analysis_path: Path
    source_sha256: str
    source_size: tuple[int, int]
    oriented_size: tuple[int, int]
    normalized_size: tuple[int, int]


def discover_input_images(inputs: list[Path]) -> tuple[list[Path], Path | None]:
    """Resolve either one input directory or an explicit ordered image list."""
    if not inputs:
        raise ValueError("At least one input image or directory is required")
    if len(inputs) == 1 and inputs[0].is_dir():
        input_directory = inputs[0]
        images = sorted(
            (
                path
                for path in input_directory.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
            ),
            key=_natural_path_key,
        )
        if not images:
            raise ValueError(
                f"No supported images found in directory: {input_directory}"
            )
        return images, input_directory

    directories = [path for path in inputs if path.is_dir()]
    if directories:
        raise ValueError(
            "Use exactly one directory, or provide image files without directories"
        )
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise ValueError(f"Input images do not exist: {missing}")
    unsupported = [
        path
        for path in inputs
        if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if unsupported:
        raise ValueError(f"Unsupported image files: {unsupported}")
    resolved = [path.resolve() for path in inputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate input images are not allowed")
    return inputs, None


def default_report_path(project_root: Path, input_directory: Path | None) -> Path:
    """Choose a stable output path without exposing an absolute input path."""
    if input_directory is None:
        return project_root / "output" / "user-instrument-batch.json"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", input_directory.name).strip("-.")
    batch_name = safe_name or "batch"
    return project_root / "output" / batch_name / "instrument-report.json"


def normalize_batch_images(
    image_paths: list[Path],
    destination_directory: Path,
    *,
    max_edge: int = NORMALIZED_MAX_EDGE,
) -> list[NormalizedBatchImage]:
    """Apply EXIF orientation and a deterministic maximum edge to every image."""
    if max_edge <= 0:
        raise ValueError("max_edge must be positive")
    destination_directory.mkdir(parents=True, exist_ok=True)
    normalized: list[NormalizedBatchImage] = []
    for index, source_path in enumerate(image_paths):
        try:
            with Image.open(source_path) as source:
                source_size = source.size
                oriented = ImageOps.exif_transpose(source).convert("RGB")
        except OSError as error:
            raise ValueError(f"Cannot decode image: {source_path}") from error
        oriented_size = oriented.size
        if max(oriented.size) > max_edge:
            oriented.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        analysis_path = destination_directory / f"{index + 1:04d}.png"
        oriented.save(analysis_path, format="PNG", compress_level=6)
        normalized.append(
            NormalizedBatchImage(
                source_path=source_path,
                analysis_path=analysis_path,
                source_sha256=_sha256(source_path),
                source_size=source_size,
                oriented_size=oriented_size,
                normalized_size=oriented.size,
            )
        )
    return normalized


def preview_crop_bbox(
    detections: list[dict[str, Any]], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Return a padded union crop around detected instruments, or the full image."""
    width, height = image_size
    boxes = [
        tuple(int(value) for value in detection["bbox"])
        for detection in detections
        if detection.get("bbox") is not None and len(detection["bbox"]) == 4
    ]
    if not boxes:
        return (0, 0, width, height)
    x1 = max(0, min(box[0] for box in boxes))
    y1 = max(0, min(box[1] for box in boxes))
    x2 = min(width, max(box[2] for box in boxes))
    y2 = min(height, max(box[3] for box in boxes))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, width, height)
    padding_x = round((x2 - x1) * 0.12)
    padding_y = round((y2 - y1) * 0.12)
    return (
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(width, x2 + padding_x),
        min(height, y2 + padding_y),
    )


def annotated_preview_data_uri(
    image_path: Path,
    detections: list[dict[str, Any]],
    *,
    preview_size: tuple[int, int] = PREVIEW_SIZE,
) -> str:
    """Create one fixed-size, self-contained preview with detected gauges boxed."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    crop_bbox = preview_crop_bbox(detections, image.size)
    cropped = image.crop(crop_bbox)
    draw = ImageDraw.Draw(cropped)
    line_width = max(3, round(max(cropped.size) / 320))
    for detection in detections:
        raw_bbox = detection.get("bbox")
        if raw_bbox is None or len(raw_bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in raw_bbox)
        box = (
            x1 - crop_bbox[0],
            y1 - crop_bbox[1],
            x2 - crop_bbox[0],
            y2 - crop_bbox[1],
        )
        draw.rectangle(box, outline="#20a464", width=line_width)
        label = str(detection.get("instance_id") or "instrument")
        label_bbox = draw.textbbox((box[0], box[1]), label)
        label_height = label_bbox[3] - label_bbox[1] + 8
        label_width = label_bbox[2] - label_bbox[0] + 10
        label_top = max(0, box[1] - label_height)
        draw.rectangle(
            (box[0], label_top, box[0] + label_width, label_top + label_height),
            fill="#20a464",
        )
        draw.text((box[0] + 5, label_top + 3), label, fill="white")

    fitted = ImageOps.contain(cropped, preview_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", preview_size, "#eef1ee")
    offset = (
        (preview_size[0] - fitted.width) // 2,
        (preview_size[1] - fitted.height) // 2,
    )
    canvas.paste(fitted, offset)
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_path_key(path: Path) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", path.name.casefold())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)
