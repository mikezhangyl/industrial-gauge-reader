import math
import unittest

import cv2
import numpy as np

from src.ethz_vision_reader import NumericLabel
from src.rapidocr_reader import (
    RobustScaleFit,
    pointer_from_rectified_mask,
    reading_from_fit,
    robust_scale_fit,
)


class RapidOCRScaleFitTests(unittest.TestCase):
    def label(
        self, value: float, phase: float, confidence: float = 0.95
    ) -> NumericLabel:
        radians = math.radians(phase)
        return NumericLabel(
            text=str(value),
            value=value,
            confidence=confidence,
            center=(320 + 165 * math.cos(radians), 320 + 165 * math.sin(radians)),
        )

    def test_merges_overlapping_hits_and_rejects_spurious_numbers(self):
        ellipse = ((320.0, 320.0), (600.0, 600.0), 0.0)
        labels = [
            self.label(0, 45),
            self.label(1, 60),
            self.label(1, 75),
            self.label(3, 120),
            self.label(3, 135),
            self.label(4, 150),
            self.label(7, 225),
            self.label(7, 240),
            self.label(8, 255),
            self.label(8, 270),
            self.label(9, 285),
            self.label(10, 315),
            self.label(-5, 180, 0.60),
            self.label(910, 300, 0.90),
        ]

        fit, inliers, rejected = robust_scale_fit(labels, ellipse)

        self.assertEqual([label.value for label in inliers], [0, 1, 3, 4, 7, 8, 9, 10])
        self.assertEqual({label.value for label in rejected}, {-5, 910})
        pointer_phase = math.radians(108.0)
        direction = np.asarray((math.cos(pointer_phase), math.sin(pointer_phase)))
        self.assertAlmostEqual(reading_from_fit(direction, fit), 2.4, delta=0.12)

    def test_reading_handles_a_wrapped_scale(self):
        fit = RobustScaleFit(
            slope=1 / 27,
            intercept=0,
            direction=1,
            origin=300,
            phase_min=0,
            phase_max=270,
            rmse=0,
        )
        pointer_phase = math.radians(8.0)
        direction = np.asarray((math.cos(pointer_phase), math.sin(pointer_phase)))
        self.assertAlmostEqual(reading_from_fit(direction, fit), 68 / 27, places=6)

    def test_wide_pointer_uses_its_tip_instead_of_triangle_area(self):
        mask = np.zeros((640, 640), dtype=np.uint8)
        triangle = np.asarray(((320, 320), (365, 270), (100, 430)), dtype=np.int32)
        cv2.fillConvexPoly(mask, triangle, 1)

        direction, _ = pointer_from_rectified_mask(mask)

        phase = math.degrees(math.atan2(direction[1], direction[0])) % 360
        expected = math.degrees(math.atan2(110, -220)) % 360
        self.assertAlmostEqual(phase, expected, delta=4.0)


if __name__ == "__main__":
    unittest.main()
