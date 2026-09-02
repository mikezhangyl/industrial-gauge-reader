import json
import tempfile
import unittest
from pathlib import Path

from src.instrument_metadata import (
    InstrumentMetadataCatalog,
    MetadataValidationError,
)


class InstrumentMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = InstrumentMetadataCatalog.load()

    def test_resolves_arrester_monitor_from_visible_model_text(self):
        matches = self.catalog.find("JCQ-10/600Z 避雷器监测器")

        self.assertEqual([item.type_id for item in matches], ["surge_arrester_monitor"])

    def test_resolves_arrester_monitor_from_multi_term_signature(self):
        matches = self.catalog.find("mA 005 机械动作电流 10kA")

        self.assertEqual([item.type_id for item in matches], ["surge_arrester_monitor"])

    def test_loads_one_metadata_bundle_per_instrument_type(self):
        self.assertEqual(
            {item.type_id for item in self.catalog.instrument_types},
            {
                "synchronous_voltmeter",
                "surge_arrester_monitor",
                "transformer_pointer_oil_level_indicator",
                "shm_d_motor_drive_unit",
                "arrester_discharge_counter",
            },
        )

    def test_arrester_counter_is_separate_from_current_channel(self):
        metadata = self.catalog.get("surge_arrester_monitor")
        current = metadata.channel("continuous_leakage_current")
        counter = metadata.channel("arrester_operation_count")

        self.assertEqual(current.display_type, "analog_pointer")
        self.assertEqual(current.unit, "mA")
        self.assertEqual(current.scales[0].minor_division, 0.05)
        self.assertEqual(counter.display_type, "mechanical_counter")
        self.assertEqual(counter.quantity_kind, "event_count")
        self.assertEqual(counter.unit, "count")
        self.assertIn(
            "例如 005 解释为累计动作 5 次",
            "".join(metadata.interpretation_rules_zh),
        )

    def test_resolves_dual_scale_synchronous_voltmeter(self):
        metadata = self.catalog.find("D96-V 同期电压表")[0]
        scales = metadata.channel("synchronizing_voltage").scales

        self.assertEqual(metadata.type_id, "synchronous_voltmeter")
        self.assertEqual(
            [(scale.maximum, scale.minor_division) for scale in scales],
            [(30.0, 1.0), (60.0, 2.0)],
        )
        self.assertEqual(
            metadata.channel("synchronizing_voltage").dial_arc.start_angle_degrees,
            268.0,
        )

    def test_oil_level_indicator_distinguishes_pointer_from_counterweight(self):
        metadata = self.catalog.find("KEQI YZF3 OIL LEVEL")[0]
        channel = metadata.channel("relative_oil_level")
        rules = "".join(metadata.interpretation_rules_zh)

        self.assertEqual(metadata.type_id, "transformer_pointer_oil_level_indicator")
        self.assertEqual(channel.display_type, "analog_pointer_with_counterweight")
        self.assertEqual(channel.unit, "relative_scale")
        self.assertEqual(channel.scales[0].maximum, 10.0)
        self.assertIn("短粗块是指针配重", rules)
        self.assertIn("不得把 3.4/10 直接解释", rules)

    def test_oil_level_indicator_matches_common_ocr_confusion(self):
        matches = self.catalog.find(
            "Shandong Taikai Power Electronic KEOI MAX 油位计 OIL-LEY MIN"
        )

        self.assertEqual(
            [item.type_id for item in matches],
            ["transformer_pointer_oil_level_indicator"],
        )

    def test_shm_d_separates_position_status_and_operation_count(self):
        metadata = self.catalog.find("SHM-D Motor drive unit")[0]
        position = metadata.channel("tap_position")
        status = metadata.channel("mechanism_status")
        counter = metadata.channel("operation_count")
        rules = "".join(metadata.interpretation_rules_zh)

        self.assertEqual(metadata.type_id, "shm_d_motor_drive_unit")
        self.assertEqual(position.display_type, "discrete_pointer_dial")
        self.assertEqual(position.scales, ())
        self.assertEqual(status.allowed_values, ("at_position", "in_transition"))
        self.assertEqual(counter.display_type, "mechanical_counter")
        self.assertIn("9a、9b、9c 不能合并成 9", rules)
        self.assertIn("003617 表示累计操作 3617 次", rules)

    def test_shm_d_manual_is_referenced_by_local_name_and_digest(self):
        metadata = self.catalog.get("shm_d_motor_drive_unit")
        manual = next(
            source
            for source in metadata.sources
            if source.source_type == "manufacturer_operation_manual"
        )

        self.assertEqual(
            manual.local_path.name,
            "SHM-D_SHM-DL_operation_instruction_HM0-460-1381.pdf",
        )
        self.assertEqual(
            manual.sha256,
            "d001a2701cbf1c90bf51b81ce24d8bdff789fd2d5562965aaf57a116310a9513",
        )
        self.assertEqual(manual.document_code, "HM 0.460.1381-01.10/2014")

    def test_js9_is_a_discrete_single_digit_counter(self):
        metadata = self.catalog.find("上海电瓷厂 JS-9 放电计数器")[0]
        channel = metadata.channel("discharge_count_display")
        rules = "".join(metadata.interpretation_rules_zh)

        self.assertEqual(metadata.type_id, "arrester_discharge_counter")
        self.assertEqual(channel.display_type, "single_pointer_circular_counter")
        self.assertEqual(channel.scales, ())
        self.assertEqual(
            channel.allowed_values, tuple(str(value) for value in range(10))
        )
        self.assertEqual(
            channel.reading_normalization.strategy, "nearest_allowed_integer"
        )
        self.assertEqual(channel.reading_normalization.maximum_distance, 0.5)
        self.assertIn("不输出小数", rules)
        self.assertIn("不允许负数", rules)
        self.assertIn("按 0 记录", rules)
        self.assertIn("只报告当前可见数字", rules)

    def test_js9_uses_the_current_monitoring_device_standard(self):
        metadata = self.catalog.get("arrester_discharge_counter")
        standard = next(
            source
            for source in metadata.sources
            if source.source_type == "current_industry_standard"
        )

        self.assertIn("JB/T 10492-2025", standard.title)
        self.assertIn("代替 JB/T 10492-2011", standard.reference)

    def test_rejects_duplicate_instrument_type_ids(self):
        valid_path = (
            Path(__file__).parents[1]
            / "metadata"
            / "instrument-types"
            / "synchronous_voltmeter"
            / "metadata.json"
        )
        valid_payload = json.loads(valid_path.read_text(encoding="utf-8"))
        instrument_type = valid_payload["instrument_type"]
        payload = {
            "schema_version": 1,
            "instrument_types": [instrument_type, instrument_type],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_path = Path(temporary_directory) / "invalid.json"
            invalid_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                MetadataValidationError, "Instrument type IDs must be unique"
            ):
                InstrumentMetadataCatalog.load(invalid_path)


if __name__ == "__main__":
    unittest.main()
