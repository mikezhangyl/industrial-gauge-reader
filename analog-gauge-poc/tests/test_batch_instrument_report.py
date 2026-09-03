import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image

from batch_instrument_report import (
    _serialize_detections,
    compare_channel,
    render_html,
    summarize_automated,
    summarize_pointer_acceptance,
)
from src.batch_io import NormalizedBatchImage
from src.gauge_reader import GaugeResult, StageTimings
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

    def test_outer_label_dial_report_box_includes_the_numeric_ring(self):
        result = GaugeResult(
            detected=True,
            bbox=(469, 557, 677, 761),
            detection_confidence=0.48,
            pointer_found=True,
            center=(580.3, 654.3),
            pointer_tip=(528.1, 605.8),
            angle_degrees=312.9,
            sweep_fraction=None,
            reading="4",
            unit="tap_position",
            confidence=0.48,
            center_method=("keypoints+pointer-aligned-outer-label-ocr"),
            timings=StageTimings(0.0, 0.0, 0.0),
        )
        analysis = SimpleNamespace(
            instances=("instance_1",),
            pointer_results=(result,),
        )

        detection = _serialize_detections(analysis, (1280, 1707))[0]

        self.assertEqual(detection["detector_bbox"], [469, 557, 677, 761])
        self.assertEqual(detection["bbox_method"], "concentric_outer_label")
        x1, y1, x2, y2 = detection["bbox"]
        self.assertLessEqual(x1, 410)
        self.assertLessEqual(y1, 440)
        self.assertGreaterEqual(x2, 760)
        self.assertGreaterEqual(y2, 780)

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
                    "source_dimensions": [1200, 800],
                    "analysis_dimensions": [1200, 800],
                    "detections": [
                        {
                            "instance_id": "instance_1",
                            "bbox": [200, 100, 700, 650],
                        }
                    ],
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
            Image.new("RGB", (1200, 800), "white").save(image_path)
            normalized = NormalizedBatchImage(
                source_path=image_path,
                analysis_path=image_path,
                source_sha256="a" * 64,
                source_size=(1200, 800),
                oriented_size=(1200, 800),
                normalized_size=(1200, 800),
            )

            report = render_html(payload, [normalized])

        self.assertIn("data:image/jpeg;base64,", report)
        self.assertIn("meter.jpg", report)
        self.assertIn("tap_position", report)
        self.assertIn("4 tap_position", report)
        self.assertIn("识别成功", report)
        self.assertIn("metadata:discrete_pointer_label", report)
        self.assertIn("程序识别通道 1", report)
        self.assertIn("检测框 1", report)
        self.assertIn("绿色框", report)
        self.assertNotIn("置信度", report)
        self.assertNotIn("87.0%", report)
        self.assertNotIn("人工确认", report)
        self.assertNotIn("最终复核值", report)
        self.assertNotIn("<th>比对</th>", report)

    def test_html_labels_metadata_free_result_as_generic_pointer_gauge(self):
        payload = {
            "automated_summary": {
                "images": 1,
                "instrument_types_recognized": 0,
                "channels": 1,
                "recognized": 1,
                "ambiguous": 0,
                "not_recognized": 0,
                "analysis_failures": 0,
            },
            "records": [
                {
                    "image": "unknown-gauge.jpg",
                    "instrument_type_id": None,
                    "analysis_failure_reason": None,
                    "detections": [],
                    "channels": [
                        {
                            "instance_id": "instance_1",
                            "channel_id": "pointer_reading",
                            "automated": {
                                "value": 2.62,
                                "candidates": [],
                                "unit": "unknown",
                                "status": "recognized",
                                "method": "generic:analog_pointer",
                                "raw_display": None,
                                "raw_ocr_text": None,
                                "note_zh": "未匹配类型 metadata；已保留通用指针读数。",
                            },
                        }
                    ],
                }
            ],
        }

        report = render_html(payload)

        self.assertIn("通用指针仪表（metadata 未匹配）", report)
        self.assertNotIn('<p class="failure-note">', report)

    def test_html_stage_gallery_embeds_thumbnail_and_links_original_png(self):
        payload = {
            "automated_summary": {
                "images": 1,
                "instrument_types_recognized": 0,
                "channels": 0,
                "recognized": 0,
                "ambiguous": 0,
                "not_recognized": 0,
                "analysis_failures": 1,
            },
            "records": [
                {
                    "image": "meter.jpg",
                    "instrument_type_id": None,
                    "analysis_failure_reason": "not read",
                    "detections": [],
                    "channels": [],
                    "processing_stages": [
                        {
                            "group": "dial-1",
                            "stage_id": "model-input",
                            "title_zh": "指针模型实际输入图",
                            "path": "processing-stages/run/image/dial-1/01-model-input.png",
                            "dimensions": [448, 448],
                            "aspect_ratio": 1.0,
                            "operation": "resize_600x400_to_448x448",
                            "source_stage": "segmentation-canvas",
                            "preserves_aspect_ratio": False,
                            "note_zh": "非正方形画布被缩放为正方形。",
                        }
                    ],
                }
            ],
        }
        with TemporaryDirectory() as temp_dir:
            report_directory = Path(temp_dir)
            stage_path = (
                report_directory
                / "processing-stages/run/image/dial-1/01-model-input.png"
            )
            stage_path.parent.mkdir(parents=True)
            Image.new("RGB", (448, 448), "white").save(stage_path)

            report = render_html(payload, report_directory=report_directory)

        self.assertIn("处理阶段审阅（1 张原尺寸 PNG）", report)
        self.assertIn("data:image/jpeg;base64,", report)
        self.assertIn("448×448 · 宽高比 1.000000", report)
        self.assertIn("改变宽高比", report)
        self.assertIn("打开原尺寸 PNG", report)
        self.assertIn("processing-stages/run/image/dial-1/01-model-input.png", report)


if __name__ == "__main__":
    unittest.main()
