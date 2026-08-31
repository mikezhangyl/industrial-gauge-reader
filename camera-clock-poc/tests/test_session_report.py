from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import numpy as np

from camera_clock_poc.clock_demo.report import _tracking_continuity, write_clock_report
from camera_clock_poc.reusable.session import SessionRecord, SessionRecorder
from camera_clock_poc.reusable.types import Observation


class SessionReportTests(unittest.TestCase):
    def test_two_second_progress_detects_a_frozen_reading(self) -> None:
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)
        records = [
            cast(
                SessionRecord,
                {
                    "timestamp": (started + timedelta(seconds=index * 0.5)).isoformat(),
                    "seconds": 40.0,
                },
            )
            for index in range(7)
        ]

        continuity, failures, windows = _tracking_continuity(records)

        self.assertEqual(continuity, 0.0)
        self.assertGreater(failures, 0)
        self.assertGreater(windows, 0)

    def test_session_writes_jsonl_screenshot_and_chinese_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            recorder = SessionRecorder(
                session_dir,
                snapshot_interval_seconds=5.0,
                archive_all_raw_frames=True,
                privacy_applied=True,
            )
            frame = np.full((240, 320, 3), 220, dtype=np.uint8)
            observation = Observation(
                captured_at=datetime.now().astimezone(),
                detected=True,
                bbox=(60, 20, 260, 220),
                pointer_found=True,
                center=(160.0, 120.0),
                pointer_tip=(90.0, 50.0),
                angle_degrees=315.0,
                value=52.5,
                confidence=0.8,
                failure_reason=None,
                processing_ms=12.3,
                method="测试",
                tilt_degrees=24.0,
                perspective_rectified=True,
                scale_reference_labels=8,
                scale_reference_rotation_degrees=4.5,
            )
            annotated = frame.copy()
            record = recorder.record(
                frame,
                annotated,
                observation,
                "alarm",
                True,
                52.0,
                privacy_applied=True,
            )
            self.assertTrue(recorder.jsonl_path.is_file())
            self.assertIsNotNone(record["screenshot"])
            self.assertIsNotNone(record["raw_screenshot"])
            assert record["raw_screenshot"] is not None
            self.assertTrue((session_dir / record["raw_screenshot"]).is_file())
            self.assertIsNotNone(record["raw_frame"])
            assert record["raw_frame"] is not None
            self.assertTrue((session_dir / record["raw_frame"]).is_file())
            self.assertTrue(record["privacy_applied"])
            self.assertEqual(record["tilt_degrees"], 24.0)
            self.assertTrue(record["perspective_rectified"])
            self.assertEqual(record["scale_reference_labels"], 8)
            self.assertEqual(record["scale_reference_rotation_degrees"], 4.5)
            self.assertIn("privacy-", str(record["raw_frame"]))
            report = write_clock_report(
                session_dir, recorder.records, "0", 2.0, privacy_mode=True
            )
            self.assertTrue(report.is_file())
            html = report.read_text(encoding="utf-8")
            self.assertIn("闹钟摄像头", html)
            self.assertIn("两秒窗口进度一致性", html)
            self.assertIn("漏检帧保留清晰原图", html)
            self.assertIn("本帧背景虚化：是", html)
            self.assertIn("平均估计倾斜角", html)
            self.assertIn("数字刻度参照", html)
            self.assertIn("最多8个", html)
            self.assertIn("24.0°", html)
            self.assertIn("data:image/jpeg;base64,", html)


if __name__ == "__main__":
    unittest.main()
