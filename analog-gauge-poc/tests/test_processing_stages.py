import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from src.processing_stages import (
    ProcessingStageWriter,
    draw_dial_candidates,
    draw_pointer_geometry,
)


class ProcessingStageWriterTests(unittest.TestCase):
    def test_stage_png_keeps_exact_dimensions_and_aspect_ratio(self):
        with TemporaryDirectory() as temp_dir:
            report_directory = Path(temp_dir)
            writer = ProcessingStageWriter(
                report_directory,
                Path("processing-stages/run/image-1"),
            )
            source = np.zeros((20, 60, 3), dtype=np.uint8)

            artifact = writer.write(
                "dial-1",
                "crop",
                source,
                title_zh="裁剪",
                operation="crop",
                source_stage="analysis-image",
                preserves_aspect_ratio=True,
            )

            output_path = report_directory / artifact.path
            saved = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(saved)
            self.assertEqual(saved.shape[:2], (20, 60))
            self.assertEqual(artifact.dimensions, (60, 20))
            self.assertEqual(artifact.aspect_ratio, 3.0)
            self.assertEqual(writer.records()[0]["dimensions"], [60, 20])

    def test_oriented_source_applies_exif_without_resizing(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.png"
            Image.new("RGB", (31, 17), "white").save(source_path)
            writer = ProcessingStageWriter(root, Path("stages/image"))

            artifact = writer.write_oriented_source(source_path)

            self.assertEqual(artifact.dimensions, (31, 17))
            self.assertEqual(artifact.operation, "exif_transpose+rgb")

    def test_detection_overlay_does_not_change_pixel_geometry(self):
        source = np.zeros((80, 120, 3), dtype=np.uint8)

        overlay = draw_dial_candidates(
            source,
            [((20, 10, 90, 70), 0.8)],
            selected_bbox=(20, 10, 90, 70),
        )

        self.assertEqual(overlay.shape, source.shape)
        self.assertFalse(np.array_equal(overlay, source))

    def test_pointer_overlay_keeps_crop_size_and_marks_center_and_tip(self):
        source = np.zeros((80, 120, 3), dtype=np.uint8)

        overlay = draw_pointer_geometry(source, (60.0, 60.0), (20.0, 20.0))

        self.assertEqual(overlay.shape, source.shape)
        self.assertGreater(int(overlay.sum()), 0)


if __name__ == "__main__":
    unittest.main()
