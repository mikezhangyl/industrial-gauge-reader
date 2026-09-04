import unittest

from src.profile_comparison import compare_profile_payloads


def _channel(instance_id, channel_id, *, status, value=None, candidates=()):
    return {
        "instance_id": instance_id,
        "channel_id": channel_id,
        "automated": {
            "status": status,
            "value": value,
            "candidates": list(candidates),
        },
    }


class ProfileComparisonTests(unittest.TestCase):
    def test_compares_channels_by_image_instance_and_channel(self):
        baseline = {
            "pipeline_profile": {"name": "448"},
            "records": [
                {
                    "image": "meter.jpg",
                    "image_sha256": "abc",
                    "channels": [
                        _channel(
                            "instance_1", "tap_position", status="recognized", value=4
                        ),
                        _channel(
                            "instance_1", "mechanism_status", status="not_recognized"
                        ),
                    ],
                    "detections": [{"instance_id": "instance_1", "angle_degrees": 313.0}],
                }
            ],
        }
        candidate = {
            "pipeline_profile": {"name": "640"},
            "records": [
                {
                    "image": "meter.jpg",
                    "image_sha256": "abc",
                    "channels": [
                        _channel(
                            "instance_1", "tap_position", status="recognized", value=1
                        ),
                        _channel(
                            "instance_1", "mechanism_status", status="recognized", value="at_position"
                        ),
                    ],
                    "detections": [{"instance_id": "instance_1", "angle_degrees": 295.0}],
                }
            ],
        }

        comparison = compare_profile_payloads(baseline, candidate)

        self.assertEqual(comparison["summary"]["channels"], 2)
        self.assertEqual(comparison["summary"]["value_changed"], 1)
        self.assertEqual(comparison["summary"]["coverage_gain"], 1)
        channels = comparison["records"][0]["channels"]
        self.assertEqual(channels[0]["change"], "value_changed")
        self.assertEqual(channels[0]["angle_delta_degrees"], 18.0)
        self.assertEqual(channels[1]["change"], "coverage_gain")

    def test_rejects_reports_for_different_source_images(self):
        baseline = {
            "pipeline_profile": {"name": "448"},
            "records": [{"image": "meter.jpg", "image_sha256": "abc", "channels": []}],
        }
        candidate = {
            "pipeline_profile": {"name": "640"},
            "records": [{"image": "meter.jpg", "image_sha256": "different", "channels": []}],
        }

        with self.assertRaisesRegex(ValueError, "image SHA-256 differs"):
            compare_profile_payloads(baseline, candidate)

    def test_method_only_change_is_not_mislabeled_as_status_change(self):
        baseline_channel = _channel(
            "instance_1", "reading", status="recognized", value=3.4
        )
        candidate_channel = _channel(
            "instance_1", "reading", status="recognized", value=3.4
        )
        baseline_channel["automated"]["method"] = "model-segmentation"
        candidate_channel["automated"]["method"] = "geometry-fallback"
        baseline = {
            "records": [
                {
                    "image": "meter.jpg",
                    "image_sha256": "abc",
                    "channels": [baseline_channel],
                }
            ]
        }
        candidate = {
            "records": [
                {
                    "image": "meter.jpg",
                    "image_sha256": "abc",
                    "channels": [candidate_channel],
                }
            ]
        }

        comparison = compare_profile_payloads(baseline, candidate)

        self.assertEqual(comparison["summary"]["details_changed"], 1)
        self.assertEqual(comparison["summary"]["status_changed"], 0)


if __name__ == "__main__":
    unittest.main()
