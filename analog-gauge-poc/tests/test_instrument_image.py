import unittest
from types import SimpleNamespace

import numpy as np

from src.instrument_image import (
    extract_counter_candidates,
    normalize_counter_display,
)
from src.rapidocr_reader import DialCandidate, deduplicate_dial_candidates


class InstrumentImageTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
