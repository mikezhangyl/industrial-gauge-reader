import unittest

import numpy as np

from src.gauge_reader import angle_from_points, format_reading_value, sweep_position


class GeometryTests(unittest.TestCase):
    def test_angle_convention(self):
        center = (50.0, 50.0)
        self.assertAlmostEqual(angle_from_points(center, (50.0, 10.0)), 0.0)
        self.assertAlmostEqual(angle_from_points(center, (90.0, 50.0)), 90.0)
        self.assertAlmostEqual(angle_from_points(center, (50.0, 90.0)), 180.0)
        self.assertAlmostEqual(angle_from_points(center, (10.0, 50.0)), 270.0)

    def test_normalized_clockwise_sweep(self):
        center = np.asarray((100.0, 100.0))
        start = np.asarray((50.0, 150.0))
        end = np.asarray((150.0, 150.0))
        tip = np.asarray((100.0, 50.0))
        angle, total, fraction = sweep_position(start, center, end, tip)
        self.assertAlmostEqual(angle, 0.0)
        self.assertAlmostEqual(total, 270.0)
        assert fraction is not None
        self.assertAlmostEqual(fraction, 0.5)

    def test_categorical_reading_label_is_not_formatted_as_a_float(self):
        self.assertEqual(format_reading_value("9a"), "9a")
        self.assertEqual(format_reading_value(4.0), "4.00")


if __name__ == "__main__":
    unittest.main()
