import math
import unittest

import cv2
import numpy as np

from src.color_scale_reader import read_color_segments


class ColorScaleReaderTests(unittest.TestCase):
    def setUp(self):
        self.dial = np.full((640, 640, 3), 230, dtype=np.uint8)
        center = (320, 320)
        axes = (270, 270)
        cv2.ellipse(self.dial, center, axes, 0, 120, 147, (0, 0, 210), 30)
        cv2.ellipse(self.dial, center, axes, 0, 155, 178, (40, 190, 50), 30)
        cv2.ellipse(self.dial, center, axes, 0, 186, 213, (40, 190, 50), 30)
        cv2.ellipse(self.dial, center, axes, 0, 221, 248, (40, 190, 50), 30)

    @staticmethod
    def direction(phase: float) -> np.ndarray:
        radians = math.radians(phase)
        return np.asarray((math.cos(radians), math.sin(radians)))

    def test_red_green_boundary_defines_relative_units(self):
        result = read_color_segments(self.dial, self.direction(181), 1.0)

        self.assertEqual(result.red_segments, 1)
        self.assertEqual(result.green_segments, 3)
        self.assertAlmostEqual(result.reading, 0.90, delta=0.08)

    def test_units_per_segment_scales_the_relative_reading(self):
        base = read_color_segments(self.dial, self.direction(181), 1.0)
        calibrated = read_color_segments(self.dial, self.direction(181), 2.0)

        self.assertAlmostEqual(calibrated.reading, base.reading * 2, places=6)

    def test_pointer_in_red_segment_is_negative(self):
        result = read_color_segments(self.dial, self.direction(135), 1.0)

        self.assertLess(result.reading, 0)


if __name__ == "__main__":
    unittest.main()
