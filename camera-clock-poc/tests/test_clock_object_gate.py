from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np

from camera_clock_poc.clock_demo.reader import (
    ClockDetection,
    ClockReferenceCorrection,
    ClockSecondHandReader,
    HandCandidate,
)


class FakeDetector:
    def __init__(self, detections: list[ClockDetection]) -> None:
        self.detections = detections

    def detect(self, image: np.ndarray) -> list[ClockDetection]:
        return self.detections


class FakeClockFaceReference:
    def estimate(
        self,
        image: np.ndarray,
        center: tuple[float, float],
        radius: float,
    ) -> ClockReferenceCorrection:
        return ClockReferenceCorrection(np.eye(3, dtype=np.float32), 0.0, 8)


class ClockObjectGateTests(unittest.TestCase):
    def test_empty_object_detection_cannot_become_a_clock(self) -> None:
        reader = ClockSecondHandReader(FakeDetector([]))
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)

        observation = reader.read(frame, datetime(2026, 8, 20, tzinfo=timezone.utc))

        self.assertFalse(observation.detected)
        self.assertFalse(observation.pointer_found)
        self.assertIsNone(observation.value)

    def test_clock_box_without_motion_is_detected_but_not_read(self) -> None:
        reader = ClockSecondHandReader(
            FakeDetector([ClockDetection((180, 80, 460, 360), 0.8)])
        )
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)

        observation = reader.read(frame, datetime(2026, 8, 20, tzinfo=timezone.utc))

        self.assertTrue(observation.detected)
        self.assertFalse(observation.pointer_found)
        self.assertIsNone(observation.value)

    def test_frisbee_candidate_without_clock_face_evidence_is_rejected(self) -> None:
        reader = ClockSecondHandReader(
            FakeDetector([ClockDetection((180, 80, 460, 360), 0.8, "frisbee")])
        )
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)

        observation = reader.read(frame, datetime(2026, 8, 20, tzinfo=timezone.utc))

        self.assertFalse(observation.detected)

    def test_frisbee_candidate_with_clock_numbers_is_accepted(self) -> None:
        reader = ClockSecondHandReader(
            FakeDetector([ClockDetection((180, 80, 460, 360), 0.8, "frisbee")]),
            FakeClockFaceReference(),
        )
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)

        observation = reader.read(frame, datetime(2026, 8, 20, tzinfo=timezone.utc))

        self.assertTrue(observation.detected)
        self.assertEqual(observation.scale_reference_labels, 8)

    def test_locked_second_hand_survives_one_blurred_frame(self) -> None:
        reader = ClockSecondHandReader(
            FakeDetector([ClockDetection((180, 80, 460, 360), 0.8)])
        )
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)

        def hand(angle: float) -> HandCandidate:
            return HandCandidate(
                tip=(256.0, 60.0),
                angle_degrees=angle,
                length_ratio=0.86,
                line_score=1.5,
            )

        per_frame_candidates = [
            [hand(300.0)],
            [hand(303.0)],
            [hand(306.0)],
            [],
        ]
        candidates = [
            channel_candidates
            for frame_candidates in per_frame_candidates
            for channel_candidates in (frame_candidates, frame_candidates)
        ]
        with patch(
            "camera_clock_poc.clock_demo.reader.detect_motion_hand_candidates",
            side_effect=candidates,
        ):
            observations = [
                reader.read(frame, started + timedelta(seconds=index * 0.5))
                for index in range(5)
            ]

        self.assertTrue(observations[3].pointer_found)
        self.assertTrue(observations[4].pointer_found)
        self.assertIsNotNone(observations[4].value)
        self.assertIn("短时预测保持", observations[4].method)


if __name__ == "__main__":
    unittest.main()
