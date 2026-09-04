import unittest

from src.pipeline_profile import (
    DEFAULT_GAUGE_PIPELINE_PROFILE_NAME,
    get_gauge_pipeline_profile,
)


class GaugePipelineProfileTests(unittest.TestCase):
    def test_high_resolution_padded_profile_is_the_audited_default(self):
        self.assertEqual(DEFAULT_GAUGE_PIPELINE_PROFILE_NAME, "448-highres-pad")

    def test_baseline_profile_preserves_existing_448_pipeline(self):
        profile = get_gauge_pipeline_profile("448")

        self.assertEqual(profile.name, "448")
        self.assertEqual(profile.detection_size, 640)
        self.assertEqual(profile.dial_canvas_size, 448)
        self.assertEqual(profile.segmentation_inference_size, 448)
        self.assertEqual(profile.segmentation_confidence, 0.12)
        self.assertFalse(profile.use_high_resolution_detail)
        self.assertFalse(profile.segment_on_high_resolution_detail)
        self.assertFalse(profile.preserve_canvas_aspect_ratio)
        self.assertEqual(profile.geometry_crop_margin_fraction, 0.0)

    def test_official_style_profile_separates_canvas_from_model_input(self):
        profile = get_gauge_pipeline_profile("448-model640")

        self.assertEqual(profile.dial_canvas_size, 448)
        self.assertEqual(profile.segmentation_inference_size, 640)

    def test_candidate_profile_uses_640_canvas_and_model_input(self):
        profile = get_gauge_pipeline_profile("640")

        self.assertEqual(profile.name, "640")
        self.assertEqual(profile.detection_size, 640)
        self.assertEqual(profile.dial_canvas_size, 640)
        self.assertEqual(profile.segmentation_inference_size, 640)

    def test_high_resolution_profiles_isolate_detail_and_padding_changes(self):
        highres = get_gauge_pipeline_profile("448-highres")
        padded = get_gauge_pipeline_profile("448-highres-pad")

        self.assertTrue(highres.use_high_resolution_detail)
        self.assertFalse(highres.segment_on_high_resolution_detail)
        self.assertFalse(highres.preserve_canvas_aspect_ratio)
        self.assertTrue(padded.use_high_resolution_detail)
        self.assertTrue(padded.preserve_canvas_aspect_ratio)
        self.assertEqual(padded.geometry_crop_margin_fraction, 0.10)

    def test_high_resolution_segmentation_is_a_separate_diagnostic_profile(self):
        profile = get_gauge_pipeline_profile("448-highres-seg")

        self.assertTrue(profile.use_high_resolution_detail)
        self.assertTrue(profile.segment_on_high_resolution_detail)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown gauge pipeline profile"):
            get_gauge_pipeline_profile("1024")


if __name__ == "__main__":
    unittest.main()
