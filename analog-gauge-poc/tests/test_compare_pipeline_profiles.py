import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from compare_pipeline_profiles import (
    create_timestamped_run_directory,
    load_completed_report,
)


class ComparePipelineProfilesTests(unittest.TestCase):
    def test_accepts_a_fresh_complete_report_with_expected_profile_and_images(self):
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
            )

            self.assertEqual(payload["pipeline_profile"]["name"], "640")

    def test_creates_an_isolated_timestamped_directory_per_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            instant = datetime(2026, 9, 4, 12, 34, 56, 123456, tzinfo=timezone.utc)

            run_directory = create_timestamped_run_directory(root, now=instant)

            self.assertEqual(
                run_directory.name,
                "20260904T123456123456+0000",
            )
            self.assertTrue(run_directory.is_dir())
            with self.assertRaises(FileExistsError):
                create_timestamped_run_directory(root, now=instant)

    def test_rejects_a_wrong_profile_report(self):
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

            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                load_completed_report(
                    json_path,
                    html_path,
                    expected_profile="640",
                    expected_images=["meter.jpg"],
                )

    def test_rejects_incomplete_report_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "instrument-report.json"
            json_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "artifacts are incomplete"):
                load_completed_report(
                    json_path,
                    root / "instrument-report.html",
                    expected_profile="640",
                    expected_images=["meter.jpg"],
                )


if __name__ == "__main__":
    unittest.main()
