import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.gauge_reader import GaugeResult, StageTimings
from src.instrument_image import (
    MetadataAwareImageAnalyzer,
    _generic_pointer_analyses,
    _generic_pointer_analysis,
    _generic_result_rank,
    _recover_generic_rectangular_pointer_results,
    _result_rank,
    _select_similar_dial_row,
    extract_counter_candidates,
    has_generated_visualization_overlay,
    normalize_counter_display,
)
from src.instrument_metadata import InstrumentMetadataCatalog
from src.instrument_reading import InstrumentReadingInterpreter
from src.rapidocr_reader import DialCandidate, deduplicate_dial_candidates


class InstrumentImageTests(unittest.TestCase):
    def test_rectified_result_ranks_above_higher_confidence_unrectified_result(self):
        base = GaugeResult(
            detected=True,
            bbox=(10, 10, 90, 90),
            detection_confidence=0.95,
            pointer_found=True,
            center=(50.0, 50.0),
            pointer_tip=(20.0, 80.0),
            angle_degrees=225.0,
            sweep_fraction=None,
            reading=3.3,
            unit=None,
            confidence=0.95,
            center_method="model+unrectified-pointer-fallback",
            timings=StageTimings(0.0, 0.0, 0.0),
        )
        rectified = replace(
            base,
            reading=3.4,
            confidence=0.67,
            center_method="model+edge-ellipse+affine-rectification",
        )

        self.assertGreater(_generic_result_rank(rectified), _generic_result_rank(base))
        self.assertGreater(_result_rank(base), _result_rank(rectified))

    def test_metadata_analyzer_prefers_stronger_rectangular_meter_geometry(self):
        generic_result = GaugeResult(
            detected=True,
            bbox=(10, 10, 90, 90),
            detection_confidence=0.7,
            pointer_found=True,
            center=(50.0, 70.0),
            pointer_tip=(20.0, 20.0),
            angle_degrees=323.0,
            sweep_fraction=None,
            reading=0.13,
            unit="mA",
            confidence=0.7,
            center_method="generic-model",
            timings=StageTimings(0.0, 0.0, 0.0),
            instrument_type_id="surge_arrester_monitor",
            readout_channel_id="continuous_leakage_current",
            interpretation_method="metadata:dial_arc_scale",
        )
        candidate = DialCandidate((20, 20, 100, 100), 0.9)
        metadata_catalog = InstrumentMetadataCatalog.load()
        metadata = metadata_catalog.get("surge_arrester_monitor")
        reader = SimpleNamespace(
            profile=SimpleNamespace(use_high_resolution_detail=False),
            reading_interpreter=InstrumentReadingInterpreter(metadata_catalog),
            ocr=lambda image: SimpleNamespace(
                boxes=None,
                txts=("JCQ-10/600Z", "mA", "005", "动作电流"),
                scores=(0.99, 0.99, 0.99, 0.99),
            ),
            detect_dial_candidates=lambda path: (candidate,),
            read=lambda path, visible_text_context=None: generic_result,
        )
        catalog = SimpleNamespace(find=lambda visible_text: (metadata,))
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "arrester-monitor.png"
            self.assertTrue(
                cv2.imwrite(
                    str(image_path),
                    np.full((120, 120, 3), 180, dtype=np.uint8),
                )
            )
            with patch(
                "src.instrument_image._detect_rectangular_meter_pointer",
                return_value=(
                    np.asarray((60.0, 85.0)),
                    np.asarray((25.0, 30.0)),
                    304.0,
                    0.0,
                ),
            ):
                analysis = MetadataAwareImageAnalyzer(reader, catalog).analyze(
                    image_path
                )

        self.assertEqual(analysis.pointer_results[0].reading, 0.0)
        self.assertEqual(
            analysis.pointer_results[0].center_method,
            "type-specific:ellipse-rectified-two-line-hidden-pivot+tick-scale",
        )

    def test_metadata_analyzer_preserves_dial_crop_pixels_before_model_reading(self):
        source = np.zeros((120, 160, 3), dtype=np.uint8)
        source[:, :, 0] = np.arange(160, dtype=np.uint8)
        source[:, :, 1] = np.arange(120, dtype=np.uint8)[:, None]
        source[25:95, 45:115, 2] = 173
        candidate = DialCandidate((40, 20, 120, 100), 0.9)
        expected_crop = source[4:116, 24:136]
        good_result = GaugeResult(
            detected=True,
            bbox=(0, 0, 112, 112),
            detection_confidence=0.9,
            pointer_found=True,
            center=(56.0, 78.0),
            pointer_tip=(20.0, 20.0),
            angle_degrees=330.0,
            sweep_fraction=None,
            reading=0.0,
            unit="mA",
            confidence=0.8,
            center_method="pixel-preserving-test",
            timings=StageTimings(0.0, 0.0, 0.0),
        )
        empty_result = GaugeResult(
            detected=False,
            bbox=None,
            detection_confidence=None,
            pointer_found=False,
            center=None,
            pointer_tip=None,
            angle_degrees=None,
            sweep_fraction=None,
            reading=None,
            unit=None,
            confidence=None,
            center_method=None,
            timings=StageTimings(0.0, 0.0, 0.0),
        )

        def read(path, visible_text_context=None):
            decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if decoded is not None and np.array_equal(decoded, expected_crop):
                return good_result
            return empty_result

        reader = SimpleNamespace(
            profile=SimpleNamespace(use_high_resolution_detail=False),
            reading_interpreter=None,
            ocr=lambda image: SimpleNamespace(
                boxes=None,
                txts=("JCQ-10/600Z", "mA", "005", "动作电流"),
                scores=(0.99, 0.99, 0.99, 0.99),
            ),
            detect_dial_candidates=lambda path: (candidate,),
            read=read,
        )
        metadata = InstrumentMetadataCatalog.load().get("surge_arrester_monitor")
        catalog = SimpleNamespace(find=lambda visible_text: (metadata,))
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "arrester-monitor.png"
            self.assertTrue(cv2.imwrite(str(image_path), source))

            analysis = MetadataAwareImageAnalyzer(reader, catalog).analyze(image_path)

        self.assertEqual(analysis.pointer_results[0].reading, 0.0)

    def test_missing_metadata_falls_back_to_generic_pointer_reading(self):
        result = GaugeResult(
            detected=True,
            bbox=(10, 10, 90, 90),
            detection_confidence=0.9,
            pointer_found=True,
            center=(50.0, 50.0),
            pointer_tip=(50.0, 10.0),
            angle_degrees=0.0,
            sweep_fraction=0.25,
            reading=2.62,
            unit=None,
            confidence=0.85,
            center_method="model-segmentation",
            timings=StageTimings(1.0, 2.0, 3.0),
        )
        reader = SimpleNamespace(
            ocr=lambda image: SimpleNamespace(txts=("unknown gauge",), scores=(0.9,)),
            detect_dial_candidates=lambda path: (),
            read=lambda path, visible_text_context=None: result,
        )
        catalog = SimpleNamespace(find=lambda visible_text: ())
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "unknown-gauge.png"
            self.assertTrue(
                cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
            )

            analysis = MetadataAwareImageAnalyzer(reader, catalog).analyze(image_path)

        self.assertIsNone(analysis.instrument_type_id)
        self.assertIsNone(analysis.failure_reason)
        self.assertEqual(analysis.instances, ("instance_1",))
        self.assertEqual(len(analysis.channels), 1)
        self.assertEqual(analysis.channels[0].channel_id, "pointer_reading")
        self.assertEqual(analysis.channels[0].value, 2.62)
        self.assertEqual(analysis.channels[0].status, "recognized")

    def test_generic_detected_gauge_without_scale_is_a_channel_failure(self):
        result = GaugeResult(
            detected=True,
            bbox=(10, 10, 90, 90),
            detection_confidence=0.8,
            pointer_found=True,
            center=(50.0, 50.0),
            pointer_tip=(20.0, 20.0),
            angle_degrees=315.0,
            sweep_fraction=None,
            reading=None,
            unit=None,
            confidence=0.7,
            center_method="model-segmentation",
            timings=StageTimings(1.0, 2.0, 3.0),
            failure_reason="OCR found only 0 numeric scale candidates",
        )

        analysis = _generic_pointer_analysis(
            "a" * 64,
            "unknown gauge",
            result,
            metadata_match_count=0,
        )

        self.assertIsNone(analysis.failure_reason)
        self.assertEqual(analysis.channels[0].status, "not_recognized")
        self.assertIsNone(analysis.channels[0].value)

    def test_generic_multi_dial_image_keeps_one_channel_per_detected_dial(self):
        results = tuple(
            GaugeResult(
                detected=True,
                bbox=(10 + index * 100, 10, 90 + index * 100, 90),
                detection_confidence=0.8,
                pointer_found=True,
                center=(50.0 + index * 100, 50.0),
                pointer_tip=(20.0 + index * 100, 20.0),
                angle_degrees=315.0,
                sweep_fraction=None,
                reading=float(index + 1),
                unit="V",
                confidence=0.7,
                center_method="model-segmentation",
                timings=StageTimings(1.0, 2.0, 3.0),
            )
            for index in range(3)
        )

        analysis = _generic_pointer_analyses(
            "a" * 64,
            "three voltmeters",
            results,
            metadata_match_count=0,
        )

        self.assertEqual(len(analysis.instances), 3)
        self.assertEqual([item.value for item in analysis.channels], [1.0, 2.0, 3.0])

    def test_generic_wide_meter_recovers_pointer_from_full_candidate(self):
        candidate = DialCandidate((20, 20, 220, 220), 0.8)
        missed = GaugeResult(
            detected=True,
            bbox=(90, 150, 120, 180),
            detection_confidence=0.9,
            pointer_found=False,
            center=None,
            pointer_tip=None,
            angle_degrees=None,
            sweep_fraction=None,
            reading=None,
            unit=None,
            confidence=0.9,
            center_method=None,
            timings=StageTimings(0.0, 0.0, 0.0),
        )

        with patch(
            "src.instrument_image._detect_rectangular_meter_pointer",
            return_value=(
                np.asarray((120.0, 180.0)),
                np.asarray((80.0, 40.0)),
                344.0,
                None,
            ),
        ):
            recovered = _recover_generic_rectangular_pointer_results(
                np.zeros((240, 240, 3), dtype=np.uint8),
                (candidate,),
                (missed,),
                None,
            )

        self.assertTrue(recovered[0].pointer_found)
        self.assertEqual(recovered[0].angle_degrees, 344.0)
        self.assertEqual(recovered[0].bbox, candidate.bbox)

    def test_dial_ocr_metadata_match_is_preserved_in_generic_fallback(self):
        result = GaugeResult(
            detected=True,
            bbox=(10, 10, 90, 90),
            detection_confidence=0.8,
            pointer_found=True,
            center=(50.0, 50.0),
            pointer_tip=(20.0, 80.0),
            angle_degrees=225.0,
            sweep_fraction=None,
            reading=3.4,
            unit="relative_scale",
            confidence=0.7,
            center_method="model-segmentation",
            timings=StageTimings(1.0, 2.0, 3.0),
            instrument_type_id="transformer_pointer_oil_level_indicator",
            readout_channel_id="relative_oil_level",
            interpretation_method="metadata:dial_arc_scale",
        )

        analysis = _generic_pointer_analysis(
            "a" * 64,
            "MAX MIN",
            result,
            metadata_match_count=0,
        )

        self.assertEqual(
            analysis.instrument_type_id,
            "transformer_pointer_oil_level_indicator",
        )
        self.assertEqual(analysis.channels[0].channel_id, "relative_oil_level")
        self.assertIn("表盘 OCR", analysis.channels[0].note_zh)

    def test_generated_visualization_overlay_is_rejected_despite_ocr_noise(self):
        for visible_text in (
            "Pointer angle: 34.48 deg Sweep position: N/A Reading: N/A",
            "Peinter ongle: 292.58 deg Sweep postion: 33.9% Rending N/A",
            "Poister angle: N/A Sweop pesiliom: N/A Reeding Ny/A",
        ):
            with self.subTest(visible_text=visible_text):
                self.assertTrue(has_generated_visualization_overlay(visible_text))

    def test_normal_instrument_text_is_not_an_overlay(self):
        visible_text = (
            "山东泰开电力电子有限公司 Shandong Taikai Power Electronic "
            "KEQI MAX 油位计 OIL-LEVEL MIN"
        )

        self.assertFalse(has_generated_visualization_overlay(visible_text))

    def test_counter_normalization_preserves_raw_ocr_separately(self):
        self.assertEqual(normalize_counter_display("00E"), "005")
        self.assertEqual(normalize_counter_display("003617"), "003617")
        self.assertIsNone(normalize_counter_display("2024.2"))

    def test_counter_extraction_rejects_digits_on_a_light_nameplate(self):
        image = np.full((80, 240, 3), 220, dtype=np.uint8)
        image[10:35, 10:70] = 35
        image[10:35, 90:150] = 35
        result = SimpleNamespace(
            boxes=np.asarray(
                [
                    [[12, 12], [68, 12], [68, 32], [12, 32]],
                    [[92, 12], [148, 12], [148, 32], [92, 32]],
                    [[170, 45], [225, 45], [225, 65], [170, 65]],
                ],
                dtype=np.float32,
            ),
            txts=("00E", "005", "0151"),
            scores=(0.8, 0.99, 0.95),
        )

        candidates = extract_counter_candidates(result, image)

        self.assertEqual(
            [candidate.normalized_display for candidate in candidates],
            ["005", "005"],
        )
        self.assertEqual([candidate.value for candidate in candidates], [5, 5])

    def test_dial_candidate_deduplication_keeps_three_physical_dials(self):
        candidates = [
            DialCandidate((0, 0, 100, 100), 0.9),
            DialCandidate((5, 5, 95, 95), 0.7),
            DialCandidate((120, 0, 220, 100), 0.8),
            DialCandidate((240, 0, 340, 100), 0.85),
        ]

        selected = deduplicate_dial_candidates(candidates)

        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0].bbox, (0, 0, 100, 100))

    def test_similar_dial_row_rejects_small_and_full_frame_false_positives(self):
        candidates = (
            DialCandidate((1274, 435, 1693, 858), 0.90),
            DialCandidate((684, 415, 1131, 834), 0.85),
            DialCandidate((74, 383, 542, 812), 0.77),
            DialCandidate((180, 837, 357, 1008), 0.59),
            DialCandidate((1054, 0, 1707, 868), 0.35),
        )

        selected = _select_similar_dial_row(candidates)

        self.assertEqual(
            [candidate.bbox for candidate in selected],
            [
                (74, 383, 542, 812),
                (684, 415, 1131, 834),
                (1274, 435, 1693, 858),
            ],
        )


if __name__ == "__main__":
    unittest.main()
