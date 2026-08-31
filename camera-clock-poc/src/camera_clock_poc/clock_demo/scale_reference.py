"""Clock-number reference used after geometric perspective rectification."""

from __future__ import annotations

import math
import re

import numpy as np

from .reader import ClockReferenceCorrection


def _signed_angle_delta(current: float, expected: float) -> float:
    return (current - expected + 180.0) % 360.0 - 180.0


def _clock_number(text: str) -> int | None:
    match = re.search(r"(?<!\d)(1[0-2]|[1-9])(?!\d)", text.replace(" ", ""))
    return int(match.group(1)) if match else None


class RapidOcrClockOrientation:
    """Use PP-OCRv6 clock numbers to identify the true 12 o'clock direction."""

    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self._ocr = RapidOCR(
            params={
                "Global.use_cls": False,
                "Global.log_level": "error",
                "Rec.rec_batch_num": 32,
                "EngineConfig.onnxruntime.use_coreml": False,
            }
        )

    def estimate(
        self,
        image: np.ndarray,
        center: tuple[float, float],
        radius: float,
    ) -> ClockReferenceCorrection | None:
        import cv2

        scale = 2.0
        enlarged = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        result = self._ocr(enlarged)
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is None or texts is None or scores is None:
            return None

        dial_center = np.asarray(center, dtype=np.float64)
        observations: list[tuple[float, float, int, np.ndarray]] = []
        for box, text_raw, confidence_raw in zip(boxes, texts, scores, strict=True):
            confidence = float(confidence_raw)
            number = _clock_number(str(text_raw))
            if confidence < 0.60 or number is None:
                continue
            label_center = np.mean(np.asarray(box, dtype=np.float64), axis=0) / scale
            offset = label_center - dial_center
            radial_ratio = float(np.linalg.norm(offset) / max(radius, 1e-6))
            if not 0.43 <= radial_ratio <= 0.90:
                continue
            observed_angle = (
                math.degrees(math.atan2(float(offset[0]), float(-offset[1]))) + 360.0
            ) % 360.0
            expected_angle = (number % 12) * 30.0
            observations.append(
                (
                    _signed_angle_delta(observed_angle, expected_angle),
                    confidence,
                    number,
                    label_center,
                )
            )
        if len({item[2] for item in observations}) < 3:
            return None

        best: (
            tuple[int, float, float, list[tuple[float, float, int, np.ndarray]]] | None
        ) = None
        for candidate, _, _, _ in observations:
            inliers = [
                item
                for item in observations
                if abs(_signed_angle_delta(item[0], candidate)) <= 12.0
            ]
            score = (
                len({item[2] for item in inliers}),
                sum(item[1] for item in inliers),
                -sum(abs(_signed_angle_delta(item[0], candidate)) for item in inliers),
            )
            if best is None or score > best[:3]:
                best = (*score, inliers)
        assert best is not None
        inliers = best[3]
        if len({item[2] for item in inliers}) < 3:
            return None
        anchor = inliers[0][0]
        weighted_delta = sum(
            _signed_angle_delta(item[0], anchor) * item[1] for item in inliers
        ) / sum(item[1] for item in inliers)
        rotation = _signed_angle_delta(anchor + weighted_delta, 0.0)

        observed_points = np.asarray([item[3] for item in inliers], dtype=np.float32)
        label_radius = float(
            np.median(np.linalg.norm(observed_points - dial_center, axis=1))
        )
        expected_points = np.asarray(
            [
                (
                    center[0]
                    + math.sin(math.radians((item[2] % 12) * 30.0)) * label_radius,
                    center[1]
                    - math.cos(math.radians((item[2] % 12) * 30.0)) * label_radius,
                )
                for item in inliers
            ],
            dtype=np.float32,
        )
        transform = cv2.getRotationMatrix2D(center, rotation, 1.0)
        reference_transform = np.eye(3, dtype=np.float32)
        reference_transform[:2, :] = transform
        if len(inliers) >= 4:
            source = np.vstack((observed_points, dial_center.astype(np.float32)))
            destination = np.vstack((expected_points, dial_center.astype(np.float32)))
            homography, mask = cv2.findHomography(
                source,
                destination,
                cv2.RANSAC,
                6.0,
            )
            if homography is not None and mask is not None and int(mask.sum()) >= 4:
                transformed_center = cv2.perspectiveTransform(
                    dial_center.astype(np.float32).reshape(1, 1, 2),
                    homography.astype(np.float32),
                ).reshape(2)
                if float(np.linalg.norm(transformed_center - dial_center)) <= 12.0:
                    reference_transform = homography.astype(np.float32)
        return ClockReferenceCorrection(
            transform=reference_transform,
            rotation_degrees=rotation,
            label_count=len({item[2] for item in inliers}),
        )
