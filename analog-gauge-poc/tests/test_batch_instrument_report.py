import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from batch_instrument_report import (
    compare_channel,
    render_html,
    summarize_automated,
    summarize_pointer_acceptance,
)
from src.instrument_image import ChannelAnalysis
from src.instrument_metadata import InstrumentMetadataCatalog
from src.instrument_observations import ConfirmedReadout


class BatchInstrumentReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = InstrumentMetadataCatalog.load()

    def test_dual_scale_candidates_match_confirmed_scale_position(self):
        automated = ChannelAnalysis(
            instance_id="instance_1",
            channel_id="synchronizing_voltage",
            value=None,
            unit="V",
            status="ambiguous",
            method="metadata:ambiguous_scale_candidates",
            confidence=0.7,
            candidates=(1.0, 2.0),
        )
        confirmed = ConfirmedReadout(
            instance_id="instance_1",
            channel_id="synchronizing_voltage",
            confirmed_value=None,
            confirmed_candidates=(1.0, 2.0),
            raw_display=None,
            unit="V",
            confirmation_status="user_confirmed_scale_position",
            note_zh=None,
        )
        channel = self.metadata.get("synchronous_voltmeter").channel(
            "synchronizing_voltage"
        )

        self.assertEqual(compare_channel(automated, confirmed, channel), "match")

    def test_counter_ocr_mismatch_is_not_hidden_by_human_confirmation(self):
        automated = ChannelAnalysis(
            instance_id="instance_1",
            channel_id="operation_count",
            value=3613,
            unit="count",
            status="recognized",
            method="ocr:mechanical_counter",
            confidence=0.95,
        )
        confirmed = ConfirmedReadout(
            instance_id="instance_1",
            channel_id="operation_count",
            confirmed_value=3617,
            confirmed_candidates=(),
            raw_display="003617",
            unit="count",
            confirmation_status="user_confirmed",
            note_zh=None,
        )
        channel = self.metadata.get("shm_d_motor_drive_unit").channel("operation_count")

        self.assertEqual(compare_channel(automated, confirmed, channel), "mismatch")

    def test_pointer_acceptance_ignores_non_pointer_channel_failures(self):
        records = [
            {
                "image": "shm-d.jpg",
                "instrument_type_id": "shm_d_motor_drive_unit",
                "expected_instrument_type_id": "shm_d_motor_drive_unit",
                "channels": [
                    {
                        "instance_id": "instance_1",
                        "channel_id": "tap_position",
                        "confirmed": {"confirmed_value": 4},
                        "comparison": "match",
                    },
                    {
                        "instance_id": "instance_1",
                        "channel_id": "operation_count",
                        "confirmed": {"confirmed_value": 3617},
                        "comparison": "mismatch",
                    },
                ],
            }
        ]

        summary = summarize_pointer_acceptance(records, self.metadata)

        self.assertEqual(summary["reviewed_pointer_channels"], 1)
        self.assertEqual(summary["automated_matches"], 1)
        self.assertEqual(summary["failures"], [])

    def test_pointer_acceptance_reports_missing_pointer_reading(self):
        records = [
            {
                "image": "jcq.jpg",
                "instrument_type_id": "surge_arrester_monitor",
                "expected_instrument_type_id": "surge_arrester_monitor",
                "channels": [
                    {
                        "instance_id": "instance_1",
                        "channel_id": "continuous_leakage_current",
                        "confirmed": {"confirmed_value": 0.0},
                        "comparison": "automated_not_available",
                    }
                ],
            }
        ]

        summary = summarize_pointer_acceptance(records, self.metadata)

        self.assertEqual(
            summary["failures"],
            [
                {
                    "image": "jcq.jpg",
                    "instance_id": "instance_1",
                    "channel_id": "continuous_leakage_current",
                    "comparison": "automated_not_available",
                }
            ],
        )

    def test_automated_summary_does_not_require_human_confirmations(self):
        records = [
            {
                "instrument_type_id": "shm_d_motor_drive_unit",
                "analysis_failure_reason": None,
                "channels": [
                    {"automated": {"status": "recognized"}},
                    {"automated": {"status": "ambiguous"}},
                    {"automated": {"status": "not_recognized"}},
                ],
            },
            {
                "instrument_type_id": None,
                "analysis_failure_reason": "No unique instrument type matched",
                "channels": [],
            },
        ]

        summary = summarize_automated(records)

        self.assertEqual(
            summary,
            {
                "images": 2,
                "instrument_types_recognized": 1,
                "channels": 3,
                "recognized": 1,
                "ambiguous": 1,
                "not_recognized": 1,
                "analysis_failures": 1,
            },
        )

    def test_html_report_embeds_image_and_shows_only_automated_results(self):
        payload = {
            "automated_summary": {
                "images": 1,
                "instrument_types_recognized": 1,
                "channels": 1,
                "recognized": 1,
                "ambiguous": 0,
                "not_recognized": 0,
                "analysis_failures": 0,
            },
            "records": [
                {
                    "image": "meter.jpg",
                    "instrument_type_id": "shm_d_motor_drive_unit",
                    "analysis_failure_reason": None,
                    "channels": [
                        {
                            "instance_id": "instance_1",
                            "channel_id": "tap_position",
                            "automated": {
                                "value": "4",
                                "candidates": [],
                                "unit": "tap_position",
                                "status": "recognized",
                                "method": "metadata:discrete_pointer_label",
                                "confidence": 0.87,
                                "raw_display": None,
                                "raw_ocr_text": None,
                                "note_zh": None,
                            },
                        }
                    ],
                }
            ],
        }
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "meter.jpg"
            image_path.write_bytes(b"\xff\xd8embedded-image\xff\xd9")

            report = render_html(payload, [image_path])

        self.assertIn("data:image/jpeg;base64,", report)
        self.assertIn("meter.jpg", report)
        self.assertIn("tap_position", report)
        self.assertIn("4 tap_position", report)
        self.assertIn("识别成功", report)
        self.assertIn("metadata:discrete_pointer_label", report)
        self.assertIn("程序识别通道 1", report)
        self.assertNotIn("置信度", report)
        self.assertNotIn("87.0%", report)
        self.assertNotIn("人工确认", report)
        self.assertNotIn("最终复核值", report)
        self.assertNotIn("<th>比对</th>", report)


if __name__ == "__main__":
    unittest.main()
