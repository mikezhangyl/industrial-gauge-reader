import unittest

import cv2
import numpy as np

from src.gauge_reader import angle_from_points
from src.rapidocr_reader import (
    DialCandidate,
    PointerSegmentationTrace,
    adaptive_tile_levels,
    detect_dial_ellipse,
    detect_rectangular_gauge_candidates,
    ellipse_fits_crop,
    expand_candidate_and_pointer_mask,
    is_plausible_dial_bbox,
    is_unclipped_tile_candidate,
    map_bbox_between_images,
    pad_dial_crop_to_square,
    pointer_center_and_tip,
    pointer_from_rectified_mask,
    select_consensus_dial_ellipse,
    select_hub_consistent_ellipse_center,
    select_pointer_segmentation_trace,
)


class AdaptiveDetectionTests(unittest.TestCase):
    @staticmethod
    def _trace(
        bbox: tuple[int, int, int, int],
        global_center: tuple[int, int],
        pointer_confidence: float,
        detection_confidence: float,
    ) -> PointerSegmentationTrace:
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        local_center = (global_center[0] - x1, global_center[1] - y1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(
            mask,
            local_center,
            (max(0, local_center[0] - 100), max(0, local_center[1] - 100)),
            1,
            8,
        )
        cv2.circle(mask, local_center, 18, 1, -1)
        crop = np.zeros((height, width, 3), dtype=np.uint8)
        return PointerSegmentationTrace(
            candidate=DialCandidate(bbox, detection_confidence),
            mask=mask,
            confidence=pointer_confidence,
            method="test",
            crop=crop,
            canvas=crop,
            model_input=None,
            model_output_mask=None,
            content_bbox=(0, 0, width, height),
        )

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

    def test_detection_bbox_maps_back_to_high_resolution_source(self):
        mapped = map_bbox_between_images(
            (100, 50, 500, 450),
            source_shape=(800, 1000),
            target_shape=(2400, 3000),
        )

        self.assertEqual(mapped, (300, 150, 1500, 1350))

    def test_non_square_dial_crop_is_padded_without_stretching(self):
        crop = np.zeros((200, 400, 3), dtype=np.uint8)

        padded, content_bbox = pad_dial_crop_to_square(crop)

        self.assertEqual(padded.shape, (400, 400, 3))
        self.assertEqual(content_bbox, (0, 100, 400, 300))

    def test_geometry_crop_expansion_keeps_pointer_at_original_coordinates(self):
        mask = np.ones((100, 200), dtype=np.uint8)
        candidate = DialCandidate((100, 50, 300, 150), 0.9)

        expanded, expanded_mask = expand_candidate_and_pointer_mask(
            candidate,
            mask,
            image_shape=(400, 500),
            margin_fraction=0.10,
        )

        self.assertEqual(expanded.bbox, (80, 40, 320, 160))
        self.assertEqual(expanded_mask.shape, (120, 240))
        self.assertEqual(int(expanded_mask[10:110, 20:220].sum()), 20_000)
        self.assertEqual(int(expanded_mask.sum()), 20_000)

    def test_ellipse_outside_crop_is_not_accepted_for_rectification(self):
        self.assertFalse(
            ellipse_fits_crop(
                ((20.0, 100.0), (180.0, 220.0), 0.0),
                (200, 200),
                tolerance_fraction=0.03,
            )
        )

    def test_large_dark_rectangular_meter_frames_are_detected(self):
        image = np.full((500, 900, 3), 220, dtype=np.uint8)
        for x1 in (50, 325, 600):
            cv2.rectangle(image, (x1, 120), (x1 + 220, 360), (20, 20, 20), 24)
            cv2.rectangle(image, (x1 + 24, 144), (x1 + 196, 336), (245, 245, 245), -1)

        candidates = detect_rectangular_gauge_candidates(image)

        self.assertEqual(len(candidates), 3)

    def test_scene_sized_frame_does_not_beat_local_dial_candidate(self):
        scene_frame = self._trace(
            (7, 0, 1440, 1487), (846, 848), 0.95, 0.66
        )
        complete_dial = self._trace(
            (552, 547, 1110, 1184), (846, 848), 0.75, 0.60
        )
        incomplete_face = self._trace(
            (660, 625, 1067, 1100), (826, 950), 0.67, 0.75
        )

        selected = select_pointer_segmentation_trace(
            [scene_frame, complete_dial, incomplete_face],
            image_shape=(1920, 1440),
        )

        self.assertEqual(selected.candidate.bbox, complete_dial.candidate.bbox)

    def test_thick_counterweight_is_not_mistaken_for_pointer_tip(self):
        mask = np.zeros((640, 640), dtype=np.uint8)
        cv2.line(mask, (170, 470), (320, 320), 1, 8)
        cv2.line(mask, (320, 320), (500, 130), 1, 24)

        direction, _ = pointer_from_rectified_mask(mask)

        self.assertAlmostEqual(
            angle_from_points((0.0, 0.0), direction),
            224.0,
            delta=5.0,
        )

    def test_outer_rim_is_preferred_over_centered_inner_decoration(self):
        image = np.full((400, 400, 3), 240, dtype=np.uint8)
        cv2.circle(image, (190, 190), 170, (20, 20, 20), 2)
        cv2.circle(image, (200, 200), 90, (20, 20, 20), 4)
        pointer_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.line(pointer_mask, (200, 200), (200, 50), 1, 8)
        cv2.circle(pointer_mask, (200, 200), 18, 1, -1)

        ellipse = detect_dial_ellipse(image, pointer_mask)

        self.assertGreater(min(ellipse[1]), 300.0)

    def test_multiple_rims_use_consensus_pointer_rectification(self):
        pointer_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.line(pointer_mask, (200, 200), (100, 300), 1, 8)
        candidates = [
            (0.010, 0.010, ((200.0, 200.0), (300.0, 390.0), 5.0)),
            (0.020, 0.020, ((180.0, 205.0), (330.0, 390.0), 5.0)),
            (0.030, 0.030, ((182.0, 203.0), (350.0, 414.0), 5.0)),
        ]

        selected = select_consensus_dial_ellipse(candidates, pointer_mask)

        self.assertIn(selected, candidates)
        self.assertNotEqual(selected, candidates[0])

    def test_shifted_large_contour_does_not_beat_concentric_dial_rims(self):
        pointer_mask = np.zeros((520, 1091), dtype=np.uint8)
        cv2.line(pointer_mask, (596, 212), (528, 458), 1, 8)
        candidates = [
            (
                0.089,
                0.053,
                ((559.0, 253.0), (288.0, 668.0), 96.2),
            ),
            (
                0.108,
                0.065,
                ((552.0, 254.0), (239.0, 564.0), 96.6),
            ),
            (
                0.131,
                0.082,
                ((534.0, 265.0), (202.0, 482.0), 97.5),
            ),
            (
                0.123,
                0.091,
                ((688.0, 240.0), (309.0, 836.0), 91.0),
            ),
            (
                0.125,
                0.106,
                ((706.0, 238.0), (402.0, 832.0), 81.9),
            ),
            (
                0.146,
                0.113,
                ((591.0, 304.0), (304.0, 627.0), 92.1),
            ),
            (
                0.146,
                0.113,
                ((593.0, 303.0), (306.0, 631.0), 91.8),
            ),
        ]

        selected = select_consensus_dial_ellipse(candidates, pointer_mask)

        self.assertEqual(selected, candidates[0])

    def test_hub_circle_repairs_shifted_partial_rim_center(self):
        ellipse = ((622.0, 774.0), (994.0, 1066.0), 160.0)

        refined = select_hub_consistent_ellipse_center(
            ellipse,
            np.asarray((732.0, 771.0)),
            np.asarray(((722.0, 763.0, 32.0), (680.0, 800.0, 55.0))),
            (1607, 1406),
        )

        self.assertEqual(refined[0], (722.0, 763.0))

    def test_existing_hub_circle_keeps_concentric_rim_center(self):
        ellipse = ((559.0, 253.0), (288.0, 668.0), 96.0)

        refined = select_hub_consistent_ellipse_center(
            ellipse,
            np.asarray((596.0, 212.0)),
            np.asarray(((563.0, 235.0, 21.0), (596.0, 209.0, 14.0))),
            (520, 1091),
        )

        self.assertEqual(refined, ellipse)


if __name__ == "__main__":
    unittest.main()
