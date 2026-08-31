"""Replay archived clock frames through the dial and second-hand pipeline.

The archived bounding boxes are deliberately reused here so detector misses do not
hide regressions in dial normalization, image preprocessing, or hand tracking.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_clock_poc.clock_demo.reader import (
    ClockDetection,
    ClockSecondHandReader,
)


@dataclass
class _RecordedDetector:
    detection: ClockDetection | None = None

    def detect(self, image: object) -> list[ClockDetection]:
        del image
        return [] if self.detection is None else [self.detection]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def replay(session_dir: Path) -> dict[str, float | int | str]:
    records = [
        json.loads(line)
        for line in (session_dir / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    detector = _RecordedDetector()
    reader = ClockSecondHandReader(detector)
    eligible = 0
    pointer_found = 0
    processing_ms: list[float] = []
    readings: list[tuple[datetime, float]] = []
    started = time.perf_counter()

    for record in records:
        raw_frame = record.get("raw_frame")
        bbox = record.get("bbox")
        if not record.get("detected") or raw_frame is None or bbox is None:
            detector.detection = None
            continue
        frame = cv2.imread(str(session_dir / raw_frame))
        if frame is None:
            raise RuntimeError(f"无法读取归档帧：{session_dir / raw_frame}")
        detector.detection = ClockDetection(tuple(bbox), 0.95)
        eligible += 1
        observation = reader.read(frame, datetime.fromisoformat(record["timestamp"]))
        processing_ms.append(observation.processing_ms)
        pointer_found += int(observation.pointer_found)
        if observation.value is not None:
            readings.append((observation.captured_at, observation.value))

    continuity_errors: list[float] = []
    for (previous_at, previous), (current_at, current) in pairwise(readings):
        elapsed_seconds = (current_at - previous_at).total_seconds()
        if not 0.05 <= elapsed_seconds <= 2.0:
            continue
        observed_seconds = (current - previous + 30.0) % 60.0 - 30.0
        continuity_errors.append(abs(observed_seconds - elapsed_seconds))

    elapsed = time.perf_counter() - started
    valid_rate = pointer_found / eligible if eligible else 0.0
    return {
        "session": session_dir.name,
        "frames": len(records),
        "eligible": eligible,
        "pointer_found": pointer_found,
        "valid_rate": valid_rate,
        "p50_ms": statistics.median(processing_ms) if processing_ms else 0.0,
        "p95_ms": _percentile(processing_ms, 0.95),
        "continuity_rate": (
            0.0
            if not continuity_errors
            else sum(error <= 1.0 for error in continuity_errors)
            / len(continuity_errors)
        ),
        "large_jumps": sum(error > 3.0 for error in continuity_errors),
        "continuity_pairs": len(continuity_errors),
        "wall_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="回放已定位表盘后的秒针识别链路")
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument(
        "--min-valid-rate",
        type=float,
        default=0.0,
        help="任一会话低于该有效率时返回失败，便于作为回归门禁",
    )
    args = parser.parse_args()

    passed = True
    for session_dir in args.sessions:
        result = replay(session_dir)
        valid_rate = float(result["valid_rate"])
        passed = passed and valid_rate >= args.min_valid_rate
        print(
            f"{result['session']}: 秒针有效={result['pointer_found']}/"
            f"{result['eligible']} ({valid_rate:.1%}), "
            f"p50={result['p50_ms']:.1f}ms, p95={result['p95_ms']:.1f}ms, "
            f"连续={result['continuity_rate']:.1%}, "
            f"大跳变={result['large_jumps']}/{result['continuity_pairs']}, "
            f"回放={result['wall_seconds']:.1f}s"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
