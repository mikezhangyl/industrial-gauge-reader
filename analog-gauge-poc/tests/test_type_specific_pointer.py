import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.gauge_reader import GaugeResult, StageTimings
from src.instrument_image import (
    HiddenPivotEstimate,
    _recover_type_specific_pointer_results,
    _shm_mechanism_status_analysis,
    detect_colored_component_pointer,
    detect_outer_meter_ellipse,
    infer_hidden_meter_pivot,
    infer_hidden_meter_sweep_fraction,
    infer_three_line_hidden_meter_pivot,
    infer_two_line_hidden_meter_pivot,
    red_chroma_mask,
    refine_hidden_meter_pivot_from_three_lines,
    resolve_hidden_meter_projection,
    select_consensus_discrete_label,
    select_meter_adjustment_reference,
    select_meter_face_geometry,
    select_meter_hub,
    select_meter_pointer_line,
    select_rectified_middle_scale_mark,
)
from src.instrument_metadata import InstrumentMetadataCatalog
from src.instrument_reading import InstrumentReadingInterpreter
from src.rapidocr_reader import RAPIDOCR_PARAMS, DialCandidate


class TypeSpecificPointerTests(unittest.TestCase):
    def test_detects_outer_meter_ellipse_for_pose_normalization(self):
        image = np.full((450, 500, 3), 240, dtype=np.uint8)
        cv2.ellipse(image, (250, 225), (230, 190), 8, 0, 360, (20, 20, 20), 5)

        ellipse = detect_outer_meter_ellipse(image)

        np.testing.assert_allclose(ellipse[0], [250.0, 225.0], atol=4.0)
        np.testing.assert_allclose(sorted(ellipse[1]), [380.0, 460.0], atol=8.0)

    def test_outer_meter_ellipse_rejects_a_larger_weak_partial_arc(self):
        image = np.full((455, 498, 3), 240, dtype=np.uint8)
        cv2.ellipse(
            image,
            (240, 208),
            (220, 210),
            48,
            0,
            360,
            (20, 20, 20),
            3,
        )
        cv2.ellipse(
            image,
            (231, 233),
            (240, 249),
            11,
            220,
            420,
            (60, 60, 60),
            3,
        )

        ellipse = detect_outer_meter_ellipse(image)

        np.testing.assert_allclose(ellipse[0], [240.0, 208.0], atol=5.0)
        np.testing.assert_allclose(sorted(ellipse[1]), [420.0, 440.0], atol=12.0)

    def test_middle_scale_mark_is_selected_after_pose_normalization(self):
        tick_lines = np.asarray(
            [
                [200.0, 174.0, 200.0, 186.0],
                [220.0, 174.0, 220.0, 186.0],
                [280.0, 174.0, 280.0, 186.0],
            ]
        )
        transform = np.asarray(
            [[1.0, 0.2, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        index, _, midpoint = select_rectified_middle_scale_mark(
            tick_lines,
            visible_adjustment_center=np.asarray([200.0, 300.0]),
            crop_shape=(400, 400),
            rectification_transform=transform,
        )

        self.assertEqual(index, 1)
        np.testing.assert_allclose(midpoint, [220.0, 180.0])

    def test_two_lines_locate_hidden_pivot_without_right_end_tick(self):
        pointer_line = np.asarray([90.0, 245.0, 145.0, 300.0])
        middle_scale_point = np.asarray([220.0, 200.0])
        adjustment_center = np.asarray([208.0, 293.0])

        estimate = infer_two_line_hidden_meter_pivot(
            pointer_line,
            middle_scale_point,
            adjustment_center,
            crop_shape=(430, 460),
            inlier_indices=(1, 2, 3, 4),
            scale_residual=2.0,
        )

        np.testing.assert_allclose(estimate.center, [200.0, 355.0], atol=1.0)
        self.assertIsNone(estimate.right_scale_line)
        self.assertEqual(estimate.projection_mode, "two_line_hidden_pivot")

    def test_hidden_meter_projection_keeps_precise_three_line_solution(self):
        rough = HiddenPivotEstimate(
            center=np.asarray([220.0, 370.0]),
            inlier_indices=(0, 1, 2, 3),
            median_residual=2.0,
            angular_spread_degrees=80.0,
        )
        precise = HiddenPivotEstimate(
            center=np.asarray([222.0, 369.0]),
            inlier_indices=(0, 1, 2),
            median_residual=1.0,
            angular_spread_degrees=80.0,
        )

        selected = resolve_hidden_meter_projection(
            rough,
            precise,
            crop_shape=(500, 480),
        )

        np.testing.assert_allclose(selected.center, precise.center)
        self.assertEqual(selected.projection_mode, "three_line_coplanar")

    def test_hidden_meter_projection_uses_scale_plane_when_parallax_is_large(self):
        rough = HiddenPivotEstimate(
            center=np.asarray([220.0, 370.0]),
            inlier_indices=(0, 1, 2, 3, 4),
            median_residual=2.0,
            angular_spread_degrees=80.0,
        )
        parallax_affected = HiddenPivotEstimate(
            center=np.asarray([245.0, 355.0]),
            inlier_indices=(0, 1, 2),
            median_residual=14.0,
            angular_spread_degrees=70.0,
            pointer_line=np.asarray([80.0, 220.0, 170.0, 315.0]),
            right_scale_line=np.asarray([330.0, 260.0, 350.0, 240.0]),
            middle_scale_line=np.asarray([225.0, 235.0, 226.0, 325.0]),
            visible_adjustment_center=np.asarray([226.0, 325.0]),
            axis_offset_degrees=8.0,
        )

        selected = resolve_hidden_meter_projection(
            rough,
            parallax_affected,
            crop_shape=(500, 480),
        )

        np.testing.assert_allclose(selected.center, rough.center)
        self.assertEqual(selected.inlier_indices, rough.inlier_indices)
        self.assertEqual(selected.projection_mode, "scale_plane_parallax_fallback")
        self.assertIsNone(selected.axis_offset_degrees)
        self.assertAlmostEqual(selected.cross_plane_residual, 14.0)

    def test_selects_zero_adjustment_screw_in_calibrated_vertical_band(self):
        circles = np.asarray(
            [
                [230.0, 295.0, 14.0],
                [215.0, 340.0, 30.0],
                [115.0, 300.0, 22.0],
            ]
        )

        selected = select_meter_adjustment_reference(
            circles,
            crop_shape=(430, 460),
            face_center_x=220.0,
        )

        np.testing.assert_allclose(selected, [230.0, 295.0])

    def test_adjustment_screw_uses_meter_body_center_not_noisy_face_line(self):
        circles = np.asarray(
            [
                [208.5, 332.5, 16.6],  # false circle favoured by noisy frame centre
                [250.5, 298.5, 14.7],  # actual zero-adjustment screw
                [263.5, 331.5, 21.0],
            ]
        )

        selected = select_meter_adjustment_reference(
            circles,
            crop_shape=(471, 489),
            face_center_x=221.5,
        )

        np.testing.assert_allclose(selected, [250.5, 298.5])

    def test_three_line_refinement_ignores_noisy_scale_votes(self):
        pointer_line = np.asarray([90.0, 245.0, 145.0, 300.0])
        tick_lines = np.asarray(
            [
                [300.0, 255.0, 320.0, 235.0],
                [220.0, 200.0, 218.0, 216.0],
                [145.0, 215.0, 152.0, 225.0],
                [260.0, 205.0, 255.0, 220.0],
                [370.0, 220.0, 378.0, 214.0],
                [335.0, 285.0, 345.0, 294.0],
            ],
            dtype=np.float64,
        )
        rough = HiddenPivotEstimate(
            center=np.asarray([230.0, 325.0]),
            inlier_indices=(0, 1, 2, 3, 4, 5),
            median_residual=4.0,
            angular_spread_degrees=80.0,
        )

        estimate = refine_hidden_meter_pivot_from_three_lines(
            rough,
            tick_lines,
            pointer_line=pointer_line,
            visible_adjustment_center=np.asarray([208.0, 293.0]),
            crop_shape=(430, 460),
        )

        np.testing.assert_allclose(estimate.center, [200.0, 355.0], atol=2.0)
        np.testing.assert_allclose(estimate.right_scale_line, tick_lines[0])
        self.assertAlmostEqual(estimate.axis_offset_degrees, -7.35, delta=0.75)
        self.assertLessEqual(len(estimate.inlier_indices), len(tick_lines))

    def test_three_line_refinement_uses_the_rightmost_valid_scale_tick(self):
        pointer_line = np.asarray([90.0, 245.0, 145.0, 300.0])
        rightmost_tick = np.asarray([340.0, 248.0, 356.0, 236.0])
        nearer_tick_with_better_radius_match = np.asarray(
            [294.0, 238.0, 306.0, 223.0]
        )
        tick_lines = np.asarray(
            [
                rightmost_tick,
                nearer_tick_with_better_radius_match,
                [218.0, 210.0, 220.0, 195.0],
                [150.0, 225.0, 158.0, 235.0],
            ]
        )
        rough = HiddenPivotEstimate(
            center=np.asarray([200.0, 355.0]),
            inlier_indices=(0, 1, 2, 3),
            median_residual=1.0,
            angular_spread_degrees=80.0,
        )

        estimate = refine_hidden_meter_pivot_from_three_lines(
            rough,
            tick_lines,
            pointer_line=pointer_line,
            visible_adjustment_center=np.asarray([208.0, 293.0]),
            crop_shape=(430, 460),
        )

        np.testing.assert_allclose(estimate.right_scale_line, rightmost_tick)

    def test_three_calibration_lines_locate_hidden_pivot_and_camera_offset(self):
        pointer_line = np.asarray([90.0, 245.0, 145.0, 300.0])
        right_scale_line = np.asarray([320.0, 235.0, 300.0, 255.0])
        middle_scale_line = np.asarray([220.0, 200.0, 208.0, 293.0])

        estimate = infer_three_line_hidden_meter_pivot(
            pointer_line,
            right_scale_line,
            middle_scale_line,
            crop_shape=(430, 460),
        )

        np.testing.assert_allclose(
            estimate.center,
            np.asarray([200.0, 355.0]),
            atol=1.0,
        )
        self.assertAlmostEqual(estimate.axis_offset_degrees, -7.35, delta=0.5)
        self.assertLess(estimate.median_residual, 1.0)

    def test_infers_hidden_pivot_from_radial_ticks_despite_false_visible_hub(self):
        tick_lines = np.asarray(
            [
                [84, 258, 100, 270],
                [126, 218, 138, 235],
                [186, 193, 190, 213],
                [254, 193, 250, 213],
                [314, 218, 302, 235],
                [356, 258, 340, 270],
                [70, 300, 370, 300],
                [45, 120, 180, 120],
            ],
            dtype=np.float64,
        )

        estimate = infer_hidden_meter_pivot(
            tick_lines,
            crop_shape=(430, 460),
            face_center_x=220.0,
            face_bottom_y=320.0,
            visible_hub=np.asarray([220.0, 295.0]),
        )

        np.testing.assert_allclose(estimate.center, np.asarray([220.0, 360.0]), atol=4)
        self.assertGreaterEqual(len(estimate.inlier_indices), 5)
        self.assertGreater(float(np.linalg.norm(estimate.center - [220.0, 295.0])), 40)

    def test_hidden_pivot_scale_uses_observed_zero_tick_direction(self):
        tick_lines = np.asarray(
            [
                [84, 258, 100, 270],
                [126, 218, 138, 235],
                [186, 193, 190, 213],
                [254, 193, 250, 213],
                [314, 218, 302, 235],
                [356, 258, 340, 270],
            ],
            dtype=np.float64,
        )
        estimate = infer_hidden_meter_pivot(
            tick_lines,
            crop_shape=(430, 460),
            face_center_x=220.0,
            face_bottom_y=320.0,
        )

        fraction = infer_hidden_meter_sweep_fraction(
            estimate,
            tick_lines,
            pointer_tip=np.asarray([96.0, 267.0]),
        )

        self.assertEqual(fraction, 0.0)

    def test_rectangular_meter_exports_accepted_pointer_when_recovery_is_unneeded(self):
        catalog = InstrumentMetadataCatalog.load()
        metadata = catalog.get("rectangular_panel_voltmeter")
        existing = GaugeResult(
            detected=True,
            bbox=(20, 20, 180, 180),
            detection_confidence=0.8,
            pointer_found=True,
            center=(100.0, 150.0),
            pointer_tip=(60.0, 80.0),
            angle_degrees=330.0,
            sweep_fraction=None,
            reading=50.0,
            unit="V",
            confidence=0.7,
            center_method="accepted-model-pointer",
            timings=StageTimings(0.0, 0.0, 0.0),
        )
        reader = SimpleNamespace(
            reading_interpreter=InstrumentReadingInterpreter(catalog)
        )
        writes = []
        writer = SimpleNamespace(
            write=lambda group, stage_id, _image, **_kwargs: writes.append(
                (group, stage_id)
            )
        )

        with patch(
            "src.instrument_image._detect_rectangular_meter_pointer",
            side_effect=ValueError("no recovery needed"),
        ):
            recovered = _recover_type_specific_pointer_results(
                np.zeros((200, 200, 3), dtype=np.uint8),
                metadata,
                (DialCandidate((20, 20, 180, 180), 0.8),),
                (existing,),
                "50 100 150",
                reader,
                writer,
            )

        self.assertEqual(recovered[0], existing)
        self.assertIn(
            ("dial-1-selected-rectangular-meter", "selected-pointer-geometry"),
            writes,
        )

    def test_hidden_meter_uses_nested_reader_bbox_for_three_line_geometry(self):
        catalog = InstrumentMetadataCatalog.load()
        metadata = catalog.get("surge_arrester_monitor")
        existing = GaugeResult(
            detected=True,
            bbox=(10, 15, 110, 115),
            detection_confidence=0.8,
            pointer_found=True,
            center=(60.0, 90.0),
            pointer_tip=(25.0, 30.0),
            angle_degrees=330.0,
            sweep_fraction=None,
            reading=0.1,
            unit="mA",
            confidence=0.7,
            center_method="generic-model",
            timings=StageTimings(0.0, 0.0, 0.0),
        )
        reader = SimpleNamespace(
            reading_interpreter=InstrumentReadingInterpreter(catalog)
        )

        with patch(
            "src.instrument_image._detect_rectangular_meter_pointer",
            return_value=(
                np.asarray((60.0, 100.0)),
                np.asarray((20.0, 40.0)),
                326.0,
                0.0,
            ),
        ) as detector:
            _recover_type_specific_pointer_results(
                np.zeros((140, 140, 3), dtype=np.uint8),
                metadata,
                (DialCandidate((20, 20, 100, 100), 0.8),),
                (existing,),
                "JCQ-10/600Z mA 005",
                reader,
            )

        geometry_candidate = detector.call_args.args[1]
        self.assertEqual(geometry_candidate.bbox, (13, 18, 107, 112))

    def test_shm_short_red_arm_reports_at_position_independently(self):
        image = np.full((240, 240, 3), 210, dtype=np.uint8)
        red = (35, 35, 165)
        cv2.line(image, (120, 120), (45, 45), red, 7)
        cv2.line(image, (120, 120), (120, 172), red, 14)
        cv2.circle(image, (120, 120), 12, red, -1)
        outer_pointer = GaugeResult(
            detected=True,
            bbox=(20, 20, 220, 220),
            detection_confidence=0.8,
            pointer_found=True,
            center=(120.0, 120.0),
            pointer_tip=(45.0, 45.0),
            angle_degrees=315.0,
            sweep_fraction=None,
            reading="4",
            unit="tap_position",
            confidence=0.8,
            center_method="test",
            timings=StageTimings(0.0, 0.0, 0.0),
        )

        result = _shm_mechanism_status_analysis(
            image,
            DialCandidate((20, 20, 220, 220), 0.8),
            outer_pointer,
            None,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.value, "at_position")
        self.assertEqual(result.status, "recognized")

    def test_discharge_counter_red_pointer_overrides_generic_wrong_direction(self):
        catalog = InstrumentMetadataCatalog.load()
        metadata = catalog.get("arrester_discharge_counter")
        existing = GaugeResult(
            detected=True,
            bbox=(20, 20, 180, 180),
            detection_confidence=0.8,
            pointer_found=True,
            center=(100.0, 100.0),
            pointer_tip=(100.0, 160.0),
            angle_degrees=180.0,
            sweep_fraction=None,
            reading=4.0,
            unit=None,
            confidence=0.7,
            center_method="generic-mask",
            timings=StageTimings(0.0, 0.0, 0.0),
        )
        reader = SimpleNamespace(
            reading_interpreter=InstrumentReadingInterpreter(catalog)
        )

        with patch(
            "src.instrument_image.detect_colored_component_pointer",
            return_value=(
                np.asarray((80.0, 80.0)),
                np.asarray((80.0, 10.0)),
                0.0,
                0.75,
            ),
        ):
            recovered = _recover_type_specific_pointer_results(
                np.zeros((200, 200, 3), dtype=np.uint8),
                metadata,
                (DialCandidate((20, 20, 180, 180), 0.8),),
                (existing,),
                "JS-9 放电计数器",
                reader,
            )

        self.assertEqual(recovered[0].reading, 0.0)
        self.assertEqual(recovered[0].angle_degrees, 0.0)
        self.assertIn("colored-component-pointer", recovered[0].center_method)

    def test_shm_uses_colored_outer_pointer_before_label_ocr(self):
        catalog = InstrumentMetadataCatalog.load()
        metadata = catalog.get("shm_d_motor_drive_unit")
        existing = GaugeResult(
            detected=True,
            bbox=(20, 20, 180, 180),
            detection_confidence=0.8,
            pointer_found=True,
            center=(100.0, 100.0),
            pointer_tip=(150.0, 150.0),
            angle_degrees=135.0,
            sweep_fraction=None,
            reading=None,
            unit=None,
            confidence=0.7,
            center_method="generic-mask",
            timings=StageTimings(0.0, 0.0, 0.0),
        )
        reader = SimpleNamespace(
            reading_interpreter=InstrumentReadingInterpreter(catalog)
        )

        with (
            patch(
                "src.instrument_image.detect_colored_component_pointer",
                return_value=(
                    np.asarray((80.0, 80.0)),
                    np.asarray((10.0, 10.0)),
                    315.0,
                    0.75,
                ),
            ),
            patch(
                "src.instrument_image._recognize_pointer_aligned_discrete_label",
                return_value=("4", 0.8),
            ),
        ):
            recovered = _recover_type_specific_pointer_results(
                np.zeros((200, 200, 3), dtype=np.uint8),
                metadata,
                (DialCandidate((20, 20, 180, 180), 0.8),),
                (existing,),
                "SHM-D Motor drive unit",
                reader,
            )

        self.assertEqual(recovered[0].reading, "4")
        self.assertEqual(recovered[0].angle_degrees, 315.0)
        self.assertIn("colored-component-pointer", recovered[0].center_method)

    def test_lower_meter_boundary_defines_face_center_despite_longer_upper_line(self):
        lines = np.asarray(
            [
                [47, 195, 198, 198],
                [183, 350, 322, 353],
                [134, 249, 262, 258],
            ],
            dtype=np.int32,
        )

        center_x, bottom_y = select_meter_face_geometry(
            lines,
            crop_shape=(471, 466),
        )

        self.assertAlmostEqual(center_x, 252.5, places=1)
        self.assertAlmostEqual(bottom_y, 351.5, places=1)

    def test_colored_component_pointer_prefers_the_long_end(self):
        image = np.full((300, 300, 3), 210, dtype=np.uint8)
        cv2.line(image, (150, 160), (150, 45), (35, 35, 145), 7)
        cv2.line(image, (150, 160), (185, 195), (35, 35, 180), 5)
        cv2.circle(image, (150, 160), 14, (35, 35, 160), -1)

        center, tip, angle, confidence = detect_colored_component_pointer(image)

        self.assertAlmostEqual(center[0], 150, delta=12)
        self.assertLess(tip[1], center[1])
        self.assertTrue(angle >= 345 or angle <= 15)
        self.assertGreaterEqual(confidence, 0.45)

    def test_red_chroma_mask_rejects_low_chroma_reddish_shadow(self):
        hsv = np.full((100, 100, 3), (0, 35, 100), dtype=np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.line(image, (50, 80), (50, 20), (30, 30, 170), 5)

        mask = red_chroma_mask(image)

        self.assertEqual(int(mask[5, 5]), 0)
        self.assertEqual(int(mask[50, 50]), 1)
        self.assertLess(float(mask.mean()), 0.08)

    def test_colored_pointer_line_is_not_skewed_by_short_status_arm(self):
        image = np.full((300, 300, 3), 210, dtype=np.uint8)
        color = (35, 35, 160)
        cv2.line(image, (150, 150), (5, 5), color, 8)
        cv2.line(image, (150, 150), (150, 270), color, 30)
        cv2.circle(image, (150, 150), 18, color, -1)

        _, _, angle, _ = detect_colored_component_pointer(image)

        self.assertAlmostEqual(angle, 315.0, delta=3.0)

    def test_specialized_ocr_uses_bounded_onnx_thread_pools(self):
        self.assertEqual(
            RAPIDOCR_PARAMS["EngineConfig.onnxruntime.inter_op_num_threads"], 1
        )
        self.assertEqual(
            RAPIDOCR_PARAMS["EngineConfig.onnxruntime.intra_op_num_threads"], 4
        )

    def test_selects_hub_near_meter_face_center_and_expected_vertical_band(self):
        circles = np.asarray(
            [
                [112.5, 346.5, 18.8],
                [209.5, 287.5, 13.0],
                [275.5, 354.5, 14.4],
            ],
            dtype=np.float32,
        )

        hub = select_meter_hub(circles, face_center_x=208.0, crop_shape=(423, 440))

        np.testing.assert_allclose(hub, np.asarray([209.5, 287.5]), atol=0.01)

    def test_rejects_false_circle_on_meter_bottom_boundary(self):
        circles = np.asarray(
            [
                [242.5, 292.5, 14.7],
                [255.5, 325.5, 22.3],
            ],
            dtype=np.float32,
        )

        hub = select_meter_hub(
            circles,
            face_center_x=258.0,
            crop_shape=(429, 463),
            face_bottom_y=331.5,
        )

        np.testing.assert_allclose(hub, np.asarray([242.5, 292.5]), atol=0.01)

    def test_selects_long_pointer_line_that_passes_near_hub(self):
        lines = np.asarray(
            [
                [75, 191, 163, 289],
                [75, 199, 143, 265],
                [200, 243, 224, 309],
                [149, 219, 246, 214],
            ],
            dtype=np.int32,
        )

        center, tip, angle = select_meter_pointer_line(
            lines,
            hub=np.asarray([210.0, 288.0]),
            crop_shape=(423, 440),
        )

        np.testing.assert_allclose(center, np.asarray([210.0, 288.0]))
        np.testing.assert_allclose(tip, np.asarray([75.0, 191.0]))
        self.assertAlmostEqual(angle, 305.7, places=1)

    def test_hidden_pointer_prefers_segment_terminating_near_adjustment_screw(self):
        lines = np.asarray(
            [
                [30, 278, 177, 417],  # longer outer-ring edge
                [158, 220, 241, 326],  # actual pointer
                [300, 210, 330, 235],
            ],
            dtype=np.int32,
        )

        _, tip, _ = select_meter_pointer_line(
            lines,
            hub=np.asarray([212.8, 385.6]),
            adjustment_center=np.asarray([208.5, 332.5]),
            crop_shape=(471, 489),
        )

        np.testing.assert_allclose(tip, np.asarray([158.0, 220.0]))

    def test_discrete_label_consensus_rejects_adjacent_label_pair(self):
        recognized = [
            ("45", 0.90),
            ("45", 0.95),
            ("45", 0.93),
            ("4", 0.72),
            ("4", 0.78),
            ("4", 0.87),
            ("4", 0.61),
            ("4", 0.58),
            ("A", 0.86),
        ]

        label, confidence = select_consensus_discrete_label(recognized)

        self.assertEqual(label, "4")
        self.assertGreater(confidence, 0.7)


if __name__ == "__main__":
    unittest.main()
