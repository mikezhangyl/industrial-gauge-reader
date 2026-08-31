from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

from camera_clock_poc.clock_demo.reader import (
    ClockDetection,
    ClockSecondHandReader,
    angle_to_seconds,
    circular_seconds_error,
)


def synthetic_clock(seconds: float) -> np.ndarray:
    image = np.full((600, 800, 3), 235, dtype=np.uint8)
    center = (400, 300)
    radius = 190
    cv2.circle(image, center, radius, (20, 20, 20), 8, cv2.LINE_AA)
    cv2.circle(image, center, 10, (20, 20, 20), -1, cv2.LINE_AA)
    for index in range(60):
        angle = math.radians(index * 6.0 - 90.0)
        length = 18 if index % 5 == 0 else 9
        outer = (
            round(center[0] + math.cos(angle) * (radius - 8)),
            round(center[1] + math.sin(angle) * (radius - 8)),
        )
        inner = (
            round(center[0] + math.cos(angle) * (radius - 8 - length)),
            round(center[1] + math.sin(angle) * (radius - 8 - length)),
        )
        cv2.line(image, inner, outer, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.line(image, center, (325, 355), (25, 25, 25), 12, cv2.LINE_AA)
    cv2.line(image, center, (470, 360), (25, 25, 25), 8, cv2.LINE_AA)
    angle = math.radians(seconds * 6.0 - 90.0)
    second_tip = (
        round(center[0] + math.cos(angle) * 170),
        round(center[1] + math.sin(angle) * 170),
    )
    cv2.line(image, center, second_tip, (0, 0, 210), 3, cv2.LINE_AA)
    return image


def foreshortened_clock(seconds: float) -> np.ndarray:
    """Simulate a circular dial viewed about 52 degrees off-axis."""

    transform = np.asarray(((0.62, 0.0, 152.0), (0.0, 1.0, 0.0)))
    return cv2.warpAffine(
        synthetic_clock(seconds),
        transform,
        (800, 600),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(235, 235, 235),
    )


class _ForeshortenedClockDetector:
    def detect(self, image: np.ndarray) -> list[ClockDetection]:
        del image
        return [ClockDetection((260, 100, 540, 500), 0.95)]


def perspective_clock(seconds: float) -> np.ndarray:
    """Simulate a dial whose far and near edges have different sizes."""

    source = np.asarray(
        ((200.0, 100.0), (600.0, 100.0), (600.0, 500.0), (200.0, 500.0)),
        dtype=np.float32,
    )
    destination = np.asarray(
        ((275.0, 145.0), (600.0, 115.0), (670.0, 490.0), (205.0, 520.0)),
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        synthetic_clock(seconds),
        transform,
        (800, 600),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(235, 235, 235),
    )


class _PerspectiveClockDetector:
    def detect(self, image: np.ndarray) -> list[ClockDetection]:
        del image
        return [ClockDetection((185, 95, 690, 540), 0.95)]


class ClockReaderTests(unittest.TestCase):
    def test_angle_to_seconds_wraps_at_twelve(self) -> None:
        self.assertAlmostEqual(angle_to_seconds(0.0), 0.0)
        self.assertAlmostEqual(angle_to_seconds(300.0), 50.0)
        self.assertAlmostEqual(angle_to_seconds(360.0), 0.0)

    def test_circular_seconds_error_handles_wrap(self) -> None:
        self.assertAlmostEqual(circular_seconds_error(59.5, 0.5), 1.0)

    def test_full_frame_synthetic_clock_reaches_level_two(self) -> None:
        reader = ClockSecondHandReader()
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)
        observations = [
            reader.read(
                synthetic_clock(51.0 + index * 0.5),
                started + timedelta(seconds=index * 0.5),
            )
            for index in range(6)
        ]
        first = observations[0]
        second = observations[-1]
        self.assertTrue(first.detected)
        self.assertTrue(second.pointer_found)
        self.assertIsNotNone(second.value)
        assert second.value is not None
        self.assertLess(circular_seconds_error(second.value, 53.5), 3.0)

    def test_foreshortened_dial_is_rectified_before_reading_seconds(self) -> None:
        reader = ClockSecondHandReader(_ForeshortenedClockDetector())
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)
        observations = [
            reader.read(
                foreshortened_clock(5.0 + index * 0.5),
                started + timedelta(seconds=index * 0.5),
            )
            for index in range(16)
        ]

        final = observations[-1]
        self.assertTrue(final.pointer_found)
        self.assertIsNotNone(final.value)
        assert final.value is not None
        self.assertLess(circular_seconds_error(final.value, 12.5), 1.0)
        self.assertTrue(final.perspective_rectified)
        self.assertIsNotNone(final.tilt_degrees)
        assert final.tilt_degrees is not None
        self.assertAlmostEqual(final.tilt_degrees, 52.0, delta=2.0)

    def test_projective_dial_uses_center_and_scale_geometry_before_reading(
        self,
    ) -> None:
        reader = ClockSecondHandReader(_PerspectiveClockDetector())
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)
        observations = [
            reader.read(
                perspective_clock(5.0 + index * 0.5),
                started + timedelta(seconds=index * 0.5),
            )
            for index in range(16)
        ]

        final = observations[-1]
        self.assertTrue(final.pointer_found)
        self.assertIsNotNone(final.value)
        assert final.value is not None
        self.assertLess(circular_seconds_error(final.value, 12.5), 1.0)
        self.assertTrue(final.perspective_rectified)


if __name__ == "__main__":
    unittest.main()
