from __future__ import annotations

import unittest

import numpy as np

from camera_clock_poc.reusable.privacy import protect_background


class PrivacyTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(20260820)
        self.frame = generator.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

    def test_clock_center_stays_clear_while_background_is_obscured(self) -> None:
        protected = protect_background(self.frame, (110, 70, 210, 170))

        np.testing.assert_array_equal(
            protected[105:135, 145:175], self.frame[105:135, 145:175]
        )
        original_background_variance = float(self.frame[:, :70].var())
        protected_background_variance = float(protected[:, :70].var())
        self.assertLess(
            protected_background_variance, original_background_variance * 0.18
        )

    def test_frame_stays_clear_before_clock_is_detected(self) -> None:
        protected = protect_background(self.frame, None)

        np.testing.assert_array_equal(protected, self.frame)


if __name__ == "__main__":
    unittest.main()
