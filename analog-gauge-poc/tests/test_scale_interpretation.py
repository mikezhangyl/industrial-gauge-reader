import math
import unittest

import numpy as np

from src.ethz_vision_reader import (
    NumericLabel,
    fit_label_ellipse,
    fit_linear_scale,
    reading_from_pointer,
)


class ScaleInterpretationTests(unittest.TestCase):
    def setUp(self):
        self.ellipse = ((300.0, 200.0), (160.0, 360.0), 90.0)

    def point_at_phase(self, phase_degrees: float) -> tuple[float, float]:
        (center_x, center_y), (axis_a, axis_b), angle = self.ellipse
        phase = math.radians(phase_degrees)
        theta = math.radians(angle)
        local_x = axis_a / 2.0 * math.cos(phase)
        local_y = axis_b / 2.0 * math.sin(phase)
        return (
            center_x + local_x * math.cos(theta) - local_y * math.sin(theta),
            center_y + local_x * math.sin(theta) + local_y * math.cos(theta),
        )

    def test_rejects_external_number_and_recovers_reading(self):
        labels = [
            NumericLabel(
                str(value), float(value), 0.99, self.point_at_phase(300 + value * 27)
            )
            for value in (0, 1, 2, 4, 5, 7, 8, 10)
        ]
        labels.append(NumericLabel("2.6", 2.6, 0.99, (40.0, 40.0)))
        ellipse, inliers, rejected = fit_label_ellipse(labels, (0, 0, 600, 400))
        self.assertEqual([label.text for label in rejected], ["2.6"])
        scale = fit_linear_scale(ellipse, inliers)
        pointer_point = np.asarray(self.point_at_phase(300 + 2.6 * 27))
        direction = pointer_point - np.asarray(ellipse[0])
        reading = reading_from_pointer(ellipse, direction, scale)
        self.assertAlmostEqual(reading, 2.6, places=2)


if __name__ == "__main__":
    unittest.main()
