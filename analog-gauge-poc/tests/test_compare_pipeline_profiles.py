import json
import tempfile
import time
import unittest
from pathlib import Path

from compare_pipeline_profiles import load_completed_report


class ComparePipelineProfilesTests(unittest.TestCase):
    def test_accepts_a_fresh_complete_report_with_expected_profile_and_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "instrument-report.json"
            html_path = root / "instrument-report.html"
            started_ns = time.time_ns()
            json_path.write_text(
                json.dumps(
                    {
                        "pipeline_profile": {"name": "640"},
                        "input_contract": {"image_order": ["meter.jpg"]},
                        "records": [{"image": "meter.jpg"}],
                    }
                ),
                encoding="utf-8",
            )
            html_path.write_text(
                "<!doctype html><html><title>report</title></html>", encoding="utf-8"
            )

            payload = load_completed_report(
                json_path,
                html_path,
                expected_profile="640",
                expected_images=["meter.jpg"],
                started_ns=started_ns,
            )

            self.assertEqual(payload["pipeline_profile"]["name"], "640")

    def test_accepts_fresh_report_with_small_filesystem_mtime_skew(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "instrument-report.json"
            html_path = root / "instrument-report.html"
            json_path.write_text(
                json.dumps(
                    {
                        "pipeline_profile": {"name": "640"},
                        "input_contract": {"image_order": ["meter.jpg"]},
                        "records": [{"image": "meter.jpg"}],
                    }
                ),
                encoding="utf-8",
            )
            html_path.write_text(
                "<!doctype html><html><title>report</title></html>", encoding="utf-8"
            )

            payload = load_completed_report(
                json_path,
                html_path,
                expected_profile="640",
                expected_images=["meter.jpg"],
                started_ns=max(
                    json_path.stat().st_mtime_ns,
                    html_path.stat().st_mtime_ns,
                )
                + 500_000,
                freshness_mtime_skew_ns=1_000_000,
            )

            self.assertEqual(payload["pipeline_profile"]["name"], "640")

    def test_rejects_a_stale_or_wrong_profile_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "instrument-report.json"
            html_path = root / "instrument-report.html"
            json_path.write_text(
                json.dumps(
                    {
                        "pipeline_profile": {"name": "448"},
                        "input_contract": {"image_order": ["meter.jpg"]},
                        "records": [{"image": "meter.jpg"}],
                    }
                ),
                encoding="utf-8",
            )
            html_path.write_text(
                "<!doctype html><html><title>report</title></html>", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "not freshly generated"):
                load_completed_report(
                    json_path,
                    html_path,
                    expected_profile="640",
                    expected_images=["meter.jpg"],
                    # Keep this well beyond the small filesystem-skew
                    # tolerance used for freshly written artifacts.
                    started_ns=time.time_ns() + 10_000_000,
                )


if __name__ == "__main__":
    unittest.main()
