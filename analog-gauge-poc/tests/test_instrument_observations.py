import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.instrument_metadata import InstrumentMetadataCatalog
from src.instrument_observations import InstrumentObservationCatalog


class InstrumentObservationTests(unittest.TestCase):
    def test_local_confirmation_is_resolved_by_digest_not_filename(self):
        metadata = InstrumentMetadataCatalog.load()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "renamed-image.jpg"
            image_path.write_bytes(b"synthetic image bytes")
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            observation = {
                "schema_version": 1,
                "observation_id": "synthetic-d96",
                "image_sha256": digest,
                "instrument_type_id": "synchronous_voltmeter",
                "readouts": [
                    {
                        "channel_id": "synchronizing_voltage",
                        "confirmed_value": None,
                        "confirmed_candidates": [1, 2],
                        "unit": "V",
                        "confirmation_status": "user_confirmed_scale_position",
                    }
                ],
                "privacy": {"publish_to_public_repository": False},
            }
            (root / "observation.json").write_text(
                json.dumps(observation), encoding="utf-8"
            )

            catalog = InstrumentObservationCatalog.load(metadata, root)
            matched = catalog.for_image(image_path)

        self.assertIsNotNone(matched)
        self.assertEqual(matched.observation_id, "synthetic-d96")
        self.assertEqual(matched.readouts[0].confirmed_candidates, (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
