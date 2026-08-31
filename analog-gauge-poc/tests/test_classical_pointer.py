import math
import unittest

import cv2
import numpy as np

from src.classical_pointer import detect_classical_pointer


class ClassicalPointerTests(unittest.TestCase):
    def test_thin_radial_pointer_beats_thick_center_mechanism(self):
        image = np.full((300, 300, 3), 225, dtype=np.uint8)
        cv2.circle(image, (150, 150), 130, (160, 160, 160), 3)
        cv2.rectangle(image, (135, 155), (165, 270), (25, 25, 25), -1)
        cv2.line(image, (150, 150), (62, 72), (20, 20, 20), 4)
        cv2.rectangle(image, (65, 0), (235, 45), (30, 30, 190), -1)
        cv2.circle(image, (150, 150), 22, (20, 20, 210), -1)

        result = detect_classical_pointer(image)

        self.assertAlmostEqual(result.center[0], 150, delta=8)
        self.assertAlmostEqual(result.center[1], 150, delta=8)
        expected = (math.degrees(math.atan2(-88, 78)) + 360.0) % 360.0
        self.assertAlmostEqual(result.angle_degrees, expected, delta=8)
        self.assertLess(result.tip[0], result.center[0])
        self.assertLess(result.tip[1], result.center[1])

    def test_requires_a_distinct_central_colored_hub(self):
        image = np.full((240, 240, 3), 220, dtype=np.uint8)
        cv2.line(image, (120, 120), (30, 40), (20, 20, 20), 3)

        with self.assertRaisesRegex(ValueError, "hub"):
            detect_classical_pointer(image)


if __name__ == "__main__":
    unittest.main()
