import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from src.batch_io import (
    annotated_preview_data_uri,
    default_report_path,
    discover_input_images,
    normalize_batch_images,
    preview_crop_bbox,
)


class BatchIOTests(unittest.TestCase):
    def test_directory_input_is_naturally_sorted_and_ignores_non_images(self):
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "iter2"
            directory.mkdir()
            for name in ("meter-10.jpg", "meter-2.jpg", "meter-1.png"):
                Image.new("RGB", (20, 10), "white").save(directory / name)
            (directory / ".DS_Store").write_bytes(b"ignored")
            (directory / "notes.txt").write_text("ignored", encoding="utf-8")

            images, input_directory = discover_input_images([directory])

        self.assertEqual(
            [path.name for path in images],
            ["meter-1.png", "meter-2.jpg", "meter-10.jpg"],
        )
        self.assertEqual(input_directory, directory)

    def test_directory_input_gets_a_stable_batch_output_path(self):
        path = default_report_path(Path("/project"), Path("input/iter 2"))

        self.assertEqual(
            path,
            Path("/project/output/iter-2/instrument-report.json"),
        )

    def test_normalization_caps_the_long_edge_and_writes_lossless_png(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "large.jpg"
            Image.new("RGB", (3000, 1000), "white").save(source)

            normalized = normalize_batch_images(
                [source],
                root / "normalized",
                max_edge=1920,
                preserve_full_resolution_detail=True,
            )[0]

            self.assertEqual(normalized.source_size, (3000, 1000))
            self.assertEqual(normalized.normalized_size, (1920, 640))
            self.assertEqual(normalized.detail_size, (3000, 1000))
            self.assertEqual(normalized.analysis_path.suffix, ".png")
            self.assertNotEqual(normalized.detail_path, normalized.analysis_path)
            with Image.open(normalized.analysis_path) as image:
                self.assertEqual(image.size, (1920, 640))
            with Image.open(normalized.detail_path) as image:
                self.assertEqual(image.size, (3000, 1000))

    def test_small_image_reuses_the_lossless_analysis_file_as_detail_source(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "small.jpg"
            Image.new("RGB", (640, 480), "white").save(source)

            normalized = normalize_batch_images(
                [source], root / "normalized", max_edge=1920
            )[0]

            self.assertEqual(normalized.detail_path, normalized.analysis_path)
            self.assertEqual(normalized.detail_size, (640, 480))

    def test_default_normalization_does_not_duplicate_unused_full_resolution(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "large.jpg"
            Image.new("RGB", (3000, 1000), "white").save(source)

            normalized = normalize_batch_images(
                [source], root / "normalized", max_edge=1920
            )[0]

            self.assertEqual(normalized.detail_path, normalized.analysis_path)
            self.assertEqual(normalized.detail_size, (1920, 640))

    def test_preview_uses_a_padded_detection_crop_and_fixed_canvas(self):
        detections = [{"instance_id": "instance_1", "bbox": [200, 100, 600, 500]}]
        self.assertEqual(
            preview_crop_bbox(detections, (1000, 800)),
            (152, 52, 648, 548),
        )
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "meter.png"
            Image.new("RGB", (1000, 800), "white").save(image_path)

            data_uri = annotated_preview_data_uri(image_path, detections)

        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
