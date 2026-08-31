import unittest

import cv2
import numpy as np

from src.rapidocr_reader import (
    adaptive_tile_levels,
    is_plausible_dial_bbox,
    is_unclipped_tile_candidate,
    pointer_center_and_tip,
)


class AdaptiveDetectionTests(unittest.TestCase):
    def test_full_panorama_is_not_a_plausible_dial_box(self):
        self.assertFalse(
            is_plausible_dial_bbox((0, 0, 2400, 600), (600, 2400))
        )
        self.assertTrue(
            is_plausible_dial_bbox((800, 100, 1400, 500), (600, 2400))
        )

    def test_tiles_cover_landscape_and_portrait_without_ratio_assumption(self):
        for image_shape in ((600, 2400), (2400, 600), (1200, 1200)):
            levels = adaptive_tile_levels(image_shape)
            self.assertGreaterEqual(len(levels), 1)
            for level in levels:
                self.assertGreater(len(level), 0)
                for x1, y1, x2, y2 in level:
                    self.assertGreaterEqual(x1, 0)
                    self.assertGreaterEqual(y1, 0)
                    self.assertLessEqual(x2, image_shape[1])
                    self.assertLessEqual(y2, image_shape[0])
                    self.assertGreater(x2, x1)
                    self.assertGreater(y2, y1)

    def test_pointer_hub_is_thickest_region_and_tip_is_longer_end(self):
        mask = np.zeros((240, 500), dtype=np.uint8)
        cv2.line(mask, (70, 120), (420, 120), 1, 8)
        cv2.circle(mask, (310, 120), 18, 1, -1)

        center, tip = pointer_center_and_tip(mask)

        self.assertAlmostEqual(center[0], 310, delta=12)
        self.assertAlmostEqual(center[1], 120, delta=5)
        self.assertLess(tip[0], 90)
        self.assertAlmostEqual(tip[1], 120, delta=5)

    def test_candidate_cut_by_internal_tile_edge_is_rejected(self):
        image_shape = (600, 2400)
        tile = (0, 0, 1200, 600)

        self.assertFalse(
            is_unclipped_tile_candidate((800, 100, 1200, 500), tile, image_shape)
        )
        self.assertTrue(
            is_unclipped_tile_candidate(
                (800, 100, 1400, 500), (600, 0, 1800, 600), image_shape
            )
        )


if __name__ == "__main__":
    unittest.main()
