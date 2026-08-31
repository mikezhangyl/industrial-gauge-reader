from __future__ import annotations

import unittest

from camera_clock_poc.clock_demo.overlay import _panel_origin


class OverlayLayoutTests(unittest.TestCase):
    def test_panel_moves_away_from_a_clock_in_the_top_left(self) -> None:
        origin = _panel_origin((1280, 720), (500, 90), (20, 20, 430, 430))

        self.assertEqual(origin, (770, 620))

    def test_panel_stays_top_left_without_a_detected_clock(self) -> None:
        self.assertEqual(_panel_origin((1280, 720), (500, 90), None), (10, 10))


if __name__ == "__main__":
    unittest.main()
