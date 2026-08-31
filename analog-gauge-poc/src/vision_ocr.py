"""Compile and call the local macOS Vision text recognizer."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OCRObservation:
    text: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


def ensure_vision_ocr(project_root: Path) -> Path:
    source = project_root / "src/vision_ocr.swift"
    executable = project_root / ".cache/vision_ocr"
    executable.parent.mkdir(parents=True, exist_ok=True)
    if not executable.exists() or executable.stat().st_mtime < source.stat().st_mtime:
        print("Compiling local macOS Vision OCR helper")
        subprocess.run(
            ["swiftc", "-O", str(source), "-o", str(executable)],
            check=True,
            timeout=180,
        )
    return executable


class VisionOCR:
    def __init__(self, executable: Path):
        self.executable = executable

    def recognize(self, image_path: Path) -> list[OCRObservation]:
        completed = subprocess.run(
            [str(self.executable), str(image_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
        return [OCRObservation(**item) for item in payload]
