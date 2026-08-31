from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

from camera_clock_poc.clock_demo.reader import (
    AdaptiveMotionHandTracker,
    HandCandidate,
    NormalizedDial,
    TrackedHand,
    detect_motion_hand_candidates,
)


def cluttered_face(angle_degrees: float, shift: tuple[float, float]) -> NormalizedDial:
    size = 512
    center = (256.0, 256.0)
    radius = 205.0
    image = np.full((size, size, 3), 232, dtype=np.uint8)
    cv2.circle(image, (256, 256), round(radius), (45, 45, 45), 6, cv2.LINE_AA)
    for angle in (45.0, 118.0, 205.0, 302.0):
        radians = math.radians(angle)
        tip = (
            round(center[0] + math.sin(radians) * radius * 0.78),
            round(center[1] - math.cos(radians) * radius * 0.78),
        )
        cv2.line(image, (256, 256), tip, (70, 70, 70), 12, cv2.LINE_AA)
    cv2.putText(
        image,
        "88",
        (170, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.8,
        (55, 55, 55),
        14,
        cv2.LINE_AA,
    )
    radians = math.radians(angle_degrees)
    second_tip = (
        round(center[0] + math.sin(radians) * radius * 0.90),
        round(center[1] - math.cos(radians) * radius * 0.90),
    )
    cv2.line(image, (256, 256), second_tip, (20, 20, 120), 3, cv2.LINE_AA)
    transform = np.asarray(((1.0, 0.0, shift[0]), (0.0, 1.0, shift[1])))
    image = cv2.warpAffine(
        image,
        transform,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return NormalizedDial(image, gray, center, radius, (0, 0, size, size))


class MotionHandDetectorTests(unittest.TestCase):
    def test_moving_thin_hand_wins_over_static_radial_clutter_and_frame_jitter(
        self,
    ) -> None:
        previous = cluttered_face(60.0, (0.0, 0.0))
        current = cluttered_face(63.0, (2.0, -1.0))

        candidates = detect_motion_hand_candidates(previous, current)

        self.assertTrue(candidates)
        error = abs((candidates[0].angle_degrees - 63.0 + 180.0) % 360.0 - 180.0)
        self.assertLess(error, 2.5)

    def test_adaptive_tracker_locks_a_motion_channel_without_delaying_output(
        self,
    ) -> None:
        tracker = AdaptiveMotionHandTracker(calibration_hits=3)
        started = datetime(2026, 8, 21, tzinfo=timezone.utc)
        readings = []
        for index in range(7):
            previous = cluttered_face(60.0 + index * 3.0, (0.0, 0.0))
            current = cluttered_face(63.0 + index * 3.0, (0.0, 0.0))
            readings.append(
                tracker.update(
                    previous,
                    current,
                    started + timedelta(seconds=(index + 1) * 0.5),
                )
            )

        self.assertTrue(any(reading is not None for reading in readings[:4]))
        self.assertIn(tracker.selected_channel, {"gray", "local_contrast"})

    def test_adaptive_tracker_rejects_an_inconsistent_channel_handoff(self) -> None:
        tracker = AdaptiveMotionHandTracker()
        started = datetime(2026, 8, 21, tzinfo=timezone.utc)

        def tracked(angle: float) -> TrackedHand:
            return TrackedHand(
                HandCandidate((256.0, 60.0), angle, 0.9, 1.0),
                velocity_degrees_per_second=6.0,
                confidence=0.9,
            )

        self.assertIsNotNone(tracker._remember(tracked(10.0), started))
        self.assertIsNone(
            tracker._remember(tracked(40.0), started + timedelta(seconds=1.0))
        )
        self.assertIsNotNone(
            tracker._remember(tracked(16.0), started + timedelta(seconds=1.0))
        )


if __name__ == "__main__":
    unittest.main()
