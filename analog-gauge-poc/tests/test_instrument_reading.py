import unittest
from dataclasses import replace

from src.gauge_reader import GaugeResult, StageTimings
from src.instrument_metadata import InstrumentMetadataCatalog
from src.instrument_reading import InstrumentReadingInterpreter


def raw_result(reading: float) -> GaugeResult:
    return GaugeResult(
        detected=True,
        bbox=(10, 10, 100, 100),
        detection_confidence=0.9,
        pointer_found=True,
        center=(50.0, 50.0),
        pointer_tip=(50.0, 10.0),
        angle_degrees=0.0,
        sweep_fraction=None,
        reading=reading,
        unit=None,
        confidence=0.8,
        center_method="test",
        timings=StageTimings(1.0, 2.0, 3.0),
    )


class InstrumentReadingInterpreterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interpreter = InstrumentReadingInterpreter(InstrumentMetadataCatalog.load())

    def test_js9_near_zero_raw_reading_becomes_discrete_zero(self):
        result = self.interpreter.interpret(
            raw_result(-0.05), "JS -9 放电 计数器 上海电瓷厂有限公司"
        )

        self.assertEqual(result.reading, 0.0)
        self.assertEqual(result.raw_reading, -0.05)
        self.assertEqual(result.unit, "count")
        self.assertEqual(result.instrument_type_id, "arrester_discharge_counter")
        self.assertEqual(result.readout_channel_id, "discharge_count_display")
        self.assertEqual(
            result.interpretation_method, "metadata:nearest_allowed_integer"
        )
        self.assertTrue(result.level3)

    def test_js9_rejects_raw_value_far_outside_an_allowed_integer(self):
        result = self.interpreter.interpret(raw_result(-0.8), "JS-9 放电计数器")

        self.assertIsNone(result.reading)
        self.assertEqual(result.raw_reading, -0.8)
        self.assertEqual(
            result.interpretation_method, "metadata:normalization_rejected"
        )
        self.assertFalse(result.level3)

    def test_decimal_pointer_reading_is_preserved_with_metadata_unit(self):
        result = self.interpreter.interpret(
            replace(raw_result(3.4), angle_degrees=None),
            "KEQI YZF3 OIL LEVEL",
        )

        self.assertEqual(result.reading, 3.4)
        self.assertEqual(result.raw_reading, 3.4)
        self.assertEqual(result.unit, "relative_scale")
        self.assertEqual(
            result.instrument_type_id,
            "transformer_pointer_oil_level_indicator",
        )
        self.assertEqual(result.interpretation_method, "metadata:preserve_raw_reading")

    def test_oil_level_uses_confirmed_zero_and_ten_arc_endpoints(self):
        visual = replace(raw_result(3.6), angle_degrees=216.0)

        result = self.interpreter.interpret(visual, "KEOI OIL LEVEL MAX MIN")

        self.assertAlmostEqual(result.reading, 3.37, delta=0.03)
        self.assertEqual(result.interpretation_method, "metadata:dial_arc_scale")

    def test_round_oil_level_correct_ellipse_maps_pointer_to_about_2_6(self):
        visual = replace(raw_result(2.7), angle_degrees=201.52776105436746)

        result = self.interpreter.interpret(
            visual,
            "油位 8 9 10 5 4 3 2 1 0",
        )

        self.assertAlmostEqual(result.reading, 2.60, delta=0.02)
        self.assertEqual(result.interpretation_method, "metadata:dial_arc_scale")

    def test_known_single_scale_arc_recovers_zero_without_numeric_ocr(self):
        visual = replace(raw_result(0.2), reading=None, angle_degrees=311.0)

        result = self.interpreter.interpret(visual, "mA 005 机械动作电流 10kA")

        self.assertEqual(result.reading, 0.0)
        self.assertEqual(result.unit, "mA")
        self.assertEqual(result.instrument_type_id, "surge_arrester_monitor")
        self.assertEqual(result.interpretation_method, "metadata:dial_arc_scale")

    def test_arrester_zero_endpoint_accepts_perspective_affected_pointer_angle(self):
        visual = replace(raw_result(0.2), reading=None, angle_degrees=301.5)

        result = self.interpreter.interpret(
            visual,
            "JCQ-10/600Z mA 005 动作电流",
        )

        self.assertEqual(result.reading, 0.0)
        self.assertEqual(result.interpretation_method, "metadata:dial_arc_scale")

    def test_hidden_pivot_tick_fraction_overrides_perspective_affected_angle(self):
        visual = replace(
            raw_result(0.2),
            reading=None,
            angle_degrees=322.0,
            sweep_fraction=0.0,
            center_method="type-specific:hidden-pivot+tick-scale+extended-line",
        )

        result = self.interpreter.interpret(
            visual,
            "JCQ-10/600Z mA 005 动作电流",
        )

        self.assertEqual(result.reading, 0.0)
        self.assertEqual(result.interpretation_method, "metadata:visual_sweep_scale")

    def test_hidden_pivot_visual_fraction_uses_metadata_endpoint_snap(self):
        visual = replace(
            raw_result(0.2),
            reading=None,
            angle_degrees=309.2,
            sweep_fraction=0.05042564285794058,
            center_method=(
                "type-specific:perspective-aware-hidden-pivot+tick-scale+pointer-tip"
            ),
        )

        result = self.interpreter.interpret(
            visual,
            "JCQ-10/600Z mA 005 动作电流",
        )

        self.assertEqual(result.reading, 0.0)
        self.assertEqual(result.interpretation_method, "metadata:visual_sweep_scale")

    def test_dual_scale_arc_returns_candidates_instead_of_guessing_scale(self):
        visual = replace(raw_result(1.0), reading=None, angle_degrees=274.0)

        result = self.interpreter.interpret(visual, "D96-V 同期电压表")

        self.assertIsNone(result.reading)
        self.assertEqual(result.reading_candidates, (1.0, 2.0))
        self.assertEqual(result.unit, "V")
        self.assertEqual(
            result.interpretation_method,
            "metadata:ambiguous_scale_candidates",
        )

    def test_dual_scale_first_tick_is_stable_under_original_image_angle(self):
        visual = replace(raw_result(1.0), reading=None, angle_degrees=278.2)

        result = self.interpreter.interpret(visual, "D96-V 同期电压表")

        self.assertEqual(result.reading_candidates, (1.0, 2.0))

    def test_unmatched_visible_text_leaves_raw_result_unchanged(self):
        original = raw_result(2.5)

        result = self.interpreter.interpret(original, "unknown instrument")

        self.assertEqual(result, original)


if __name__ == "__main__":
    unittest.main()
