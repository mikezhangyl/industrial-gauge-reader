"""Auditable JSONL and event-screenshot session recorder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from .types import Observation


class SessionRecord(TypedDict):
    timestamp: str
    detected: bool
    pointer_found: bool
    bbox: list[int] | None
    angle_degrees: float | None
    seconds: float | None
    confidence: float | None
    alarm_state: str
    expected_seconds: float | None
    failure_reason: str | None
    processing_ms: float
    method: str
    screenshot: str | None
    raw_screenshot: str | None
    raw_frame: str | None
    privacy_applied: bool
    tilt_degrees: float | None
    perspective_rectified: bool
    scale_reference_labels: int
    scale_reference_rotation_degrees: float | None


class SessionRecorder:
    def __init__(
        self,
        session_dir: Path,
        snapshot_interval_seconds: float = 5.0,
        archive_all_raw_frames: bool = False,
        privacy_applied: bool = False,
    ):
        self.session_dir = session_dir
        self.screenshot_dir = session_dir / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.archive_all_raw_frames = archive_all_raw_frames
        self.privacy_applied = privacy_applied
        self.frame_dir = session_dir / "frames"
        if archive_all_raw_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = session_dir / "records.jsonl"
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.records: list[SessionRecord] = []
        self._last_snapshot_timestamp: float | None = None
        self._previous_detected: bool | None = None
        self._previous_alarm: str | None = None
        self._previous_failure: str | None = None

    def record(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        observation: Observation,
        alarm_state: str,
        alarm_changed: bool,
        expected_seconds: float | None,
        manual_snapshot: bool = False,
        privacy_applied: bool | None = None,
    ) -> SessionRecord:
        timestamp = observation.captured_at.timestamp()
        interval_elapsed = (
            self._last_snapshot_timestamp is None
            or timestamp - self._last_snapshot_timestamp
            >= self.snapshot_interval_seconds
        )
        state_changed = (
            self._previous_detected != observation.detected
            or self._previous_alarm != alarm_state
            or self._previous_failure != observation.failure_reason
            or alarm_changed
        )
        screenshot_path: Path | None = None
        raw_screenshot_path: Path | None = None
        stem = observation.captured_at.strftime("%Y%m%d-%H%M%S-%f")
        raw_frame_path: Path | None = None
        frame_privacy_applied = (
            self.privacy_applied if privacy_applied is None else privacy_applied
        )
        frame_prefix = "privacy" if frame_privacy_applied else "unprotected"
        if self.archive_all_raw_frames:
            raw_frame_path = self.frame_dir / f"{frame_prefix}-{stem}.jpg"
            cv2.imwrite(str(raw_frame_path), raw_frame)
        if interval_elapsed or state_changed or manual_snapshot:
            screenshot_path = self.screenshot_dir / f"annotated-{stem}.jpg"
            raw_screenshot_path = self.screenshot_dir / f"{frame_prefix}-{stem}.jpg"
            cv2.imwrite(str(screenshot_path), annotated_frame)
            cv2.imwrite(str(raw_screenshot_path), raw_frame)
            self._last_snapshot_timestamp = timestamp

        record: SessionRecord = {
            "timestamp": observation.captured_at.isoformat(timespec="milliseconds"),
            "detected": observation.detected,
            "pointer_found": observation.pointer_found,
            "bbox": list(observation.bbox) if observation.bbox else None,
            "angle_degrees": observation.angle_degrees,
            "seconds": observation.value,
            "confidence": observation.confidence,
            "alarm_state": alarm_state,
            "expected_seconds": expected_seconds,
            "failure_reason": observation.failure_reason,
            "processing_ms": observation.processing_ms,
            "method": observation.method,
            "screenshot": (
                str(screenshot_path.relative_to(self.session_dir))
                if screenshot_path
                else None
            ),
            "raw_screenshot": (
                str(raw_screenshot_path.relative_to(self.session_dir))
                if raw_screenshot_path
                else None
            ),
            "raw_frame": (
                str(raw_frame_path.relative_to(self.session_dir))
                if raw_frame_path
                else None
            ),
            "privacy_applied": frame_privacy_applied,
            "tilt_degrees": observation.tilt_degrees,
            "perspective_rectified": observation.perspective_rectified,
            "scale_reference_labels": observation.scale_reference_labels,
            "scale_reference_rotation_degrees": (
                observation.scale_reference_rotation_degrees
            ),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        self._previous_detected = observation.detected
        self._previous_alarm = alarm_state
        self._previous_failure = observation.failure_reason
        return record
