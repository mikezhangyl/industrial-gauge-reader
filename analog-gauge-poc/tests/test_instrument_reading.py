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
        result = self.interpreter.interpret(raw_result(3.4), "KEQI YZF3 OIL LEVEL")

        self.assertEqual(result.reading, 3.4)
        self.assertEqual(result.raw_reading, 3.4)
        self.assertEqual(result.unit, "relative_scale")
        self.assertEqual(
            result.instrument_type_id,
            "transformer_pointer_oil_level_indicator",
        )
        self.assertEqual(result.interpretation_method, "metadata:preserve_raw_reading")

    def test_known_single_scale_arc_recovers_zero_without_numeric_ocr(self):
        visual = replace(raw_result(0.2), reading=None, angle_degrees=311.0)

        result = self.interpreter.interpret(visual, "mA 005 机械动作电流 10kA")

        self.assertEqual(result.reading, 0.0)
        self.assertEqual(result.unit, "mA")
        self.assertEqual(result.instrument_type_id, "surge_arrester_monitor")
        self.assertEqual(result.interpretation_method, "metadata:dial_arc_scale")

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

    def test_unmatched_visible_text_leaves_raw_result_unchanged(self):
        original = raw_result(2.5)

        result = self.interpreter.interpret(original, "unknown instrument")

        self.assertEqual(result, original)


if __name__ == "__main__":
    unittest.main()
