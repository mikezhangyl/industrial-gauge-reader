import unittest

import numpy as np

from src.instrument_image import (
    select_consensus_discrete_label,
    select_meter_hub,
    select_meter_pointer_line,
)
from src.rapidocr_reader import RAPIDOCR_PARAMS


class TypeSpecificPointerTests(unittest.TestCase):
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
