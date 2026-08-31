"""Pretrained COCO clock detector used only by the clock demonstration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .reader import ClockDetection


class YoloClockDetector:
    """Find COCO clock candidates, including clocks mislabeled as frisbees."""

    CLOCK_CLASS_ID = 74
    FRISBEE_CLASS_ID = 29

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        confidence: float = 0.12,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._device = device
        self._confidence = confidence

    def detect(self, image: np.ndarray) -> list[ClockDetection]:
        results = self._model.predict(
            source=image,
            classes=[self.CLOCK_CLASS_ID, self.FRISBEE_CLASS_ID],
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        detections: list[ClockDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
                confidence = float(box.conf[0].detach().cpu())
                class_id = int(box.cls[0].detach().cpu())
                detections.append(
                    ClockDetection(
                        bbox=(round(x1), round(y1), round(x2), round(y2)),
                        confidence=confidence,
                        candidate_class=(
                            "clock" if class_id == self.CLOCK_CLASS_ID else "frisbee"
                        ),
                    )
                )
        return detections
