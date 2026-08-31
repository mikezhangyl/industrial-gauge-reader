"""Latest-frame camera adapter that deliberately drops stale buffered frames."""

from __future__ import annotations

import platform
import threading
import time
from datetime import datetime
from typing import Self

import cv2

from .types import CapturedFrame


def parse_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


class LatestFrameCamera:
    def __init__(
        self,
        source: int | str,
        width: int | None = 1280,
        height: int | None = 720,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self._capture: cv2.VideoCapture | None = None
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._sequence = 0
        self._closed = False
        self._thread: threading.Thread | None = None

    def start(self) -> Self:
        if isinstance(self.source, int) and platform.system() == "Darwin":
            capture = cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)
        else:
            capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开摄像头源：{self.source}")
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture
        self._closed = False
        self._thread = threading.Thread(
            target=self._read_loop, name="latest-frame-camera", daemon=True
        )
        self._thread.start()
        return self

    def _read_loop(self) -> None:
        assert self._capture is not None
        while not self._closed:
            ok, image = self._capture.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self._condition:
                self._sequence += 1
                self._latest = CapturedFrame(
                    self._sequence, datetime.now().astimezone(), image
                )
                self._condition.notify_all()

    def read_latest(
        self, after_sequence: int = 0, timeout: float = 2.0
    ) -> CapturedFrame:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                self._latest is None or self._latest.sequence <= after_sequence
            ) and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("等待摄像头最新帧超时")
                self._condition.wait(remaining)
            if self._latest is None:
                raise RuntimeError("摄像头已关闭且没有可用画面")
            latest = self._latest
            return CapturedFrame(
                latest.sequence, latest.captured_at, latest.image.copy()
            )

    def close(self) -> None:
        self._closed = True
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()
