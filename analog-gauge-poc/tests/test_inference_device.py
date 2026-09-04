from __future__ import annotations

import unittest
from unittest.mock import patch

from src.inference_device import (
    accelerator_fingerprint,
    available_devices,
    ensure_device_available,
    synchronize,
)


class InferenceDeviceTests(unittest.TestCase):
    def test_cpu_is_always_available(self):
        ensure_device_available("cpu")
        self.assertEqual(available_devices("cpu"), ["cpu"])

    def test_cuda_request_fails_before_model_loading_when_unavailable(self):
        with (
            patch("src.inference_device.torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "PyTorch cannot access"),
        ):
            ensure_device_available("cuda")

    def test_all_includes_cuda_when_available(self):
        with (
            patch(
                "src.inference_device.torch.backends.mps.is_built",
                return_value=False,
            ),
            patch(
                "src.inference_device.torch.backends.mps.is_available",
                return_value=False,
            ),
            patch("src.inference_device.torch.cuda.is_available", return_value=True),
        ):
            self.assertEqual(available_devices("all"), ["cpu", "cuda"])

    def test_cuda_synchronization_is_used_for_timing(self):
        with (
            patch("src.inference_device.torch.cuda.is_available", return_value=True),
            patch("src.inference_device.torch.cuda.synchronize") as cuda_sync,
        ):
            synchronize("cuda")
        cuda_sync.assert_called_once_with()

    def test_mps_synchronization_remains_supported(self):
        with (
            patch(
                "src.inference_device.torch.backends.mps.is_available",
                return_value=True,
            ),
            patch("src.inference_device.torch.mps.synchronize") as mps_sync,
        ):
            synchronize("mps")
        mps_sync.assert_called_once_with()

    def test_accelerator_fingerprint_records_cuda_device(self):
        with (
            patch("src.inference_device.torch.cuda.is_available", return_value=True),
            patch("src.inference_device.torch.cuda.device_count", return_value=1),
            patch(
                "src.inference_device.torch.cuda.get_device_name",
                return_value="NVIDIA A10",
            ),
            patch("src.inference_device.torch.cuda.current_device", return_value=0),
            patch(
                "src.inference_device.torch.cuda.memory_allocated",
                return_value=123,
            ),
            patch(
                "src.inference_device.torch.cuda.memory_reserved",
                return_value=456,
            ),
            patch(
                "src.inference_device.torch.backends.mps.is_built",
                return_value=False,
            ),
            patch(
                "src.inference_device.torch.backends.mps.is_available",
                return_value=False,
            ),
        ):
            result = accelerator_fingerprint("cuda")
        self.assertEqual(result["selected_device"], "cuda")
        self.assertEqual(result["cuda_device_count"], 1)
        self.assertEqual(result["cuda_devices"], ["NVIDIA A10"])
        self.assertEqual(result["cuda_current_device"], 0)
        self.assertEqual(result["cuda_memory_allocated_bytes"], 123)
        self.assertEqual(result["cuda_memory_reserved_bytes"], 456)
        self.assertEqual(result["ocr_execution_policy"], "cpu")

    def test_unknown_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported inference device"):
            ensure_device_available("gpu")


if __name__ == "__main__":
    unittest.main()
