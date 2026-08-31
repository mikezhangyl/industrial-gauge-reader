from __future__ import annotations

import unittest

from camera_clock_poc.reusable.capture import parse_source


class CaptureTests(unittest.TestCase):
    def test_numeric_source_becomes_camera_index(self) -> None:
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("12"), 12)

    def test_rtsp_source_remains_string(self) -> None:
        source = "rtsp://camera/live"
        self.assertEqual(parse_source(source), source)


if __name__ == "__main__":
    unittest.main()
