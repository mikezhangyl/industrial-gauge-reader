"""Run repeatable full-image gauge benchmarks with two pretrained solutions."""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from src.gauge_reader import AnalogGaugeReader, GaugeResult, annotate
from src.model_store import ensure_models
from src.rapidocr_reader import EthzPaddleGaugeReader

PROJECT_ROOT = Path(__file__).resolve().parent
Metrics = dict[str, dict[str, float]]


class Reader(Protocol):
    def read(self, image_path: Path) -> GaugeResult: ...


ReaderFactory = Callable[[str], Reader]


@dataclass(frozen=True)
class BenchmarkRecord:
    model_name: str
    device: str
    result: GaugeResult
    metrics: Metrics
    model_load_ms: float
    reader: Reader


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def available_devices(selection: str) -> list[str]:
    mps_available = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    if selection == "cpu":
        return ["cpu"]
    if selection == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is not available")
        return ["mps"]
    devices = ["cpu"]
    if mps_available:
        devices.append("mps")
    return devices


def print_result(result: GaugeResult) -> None:
    print(f"Gauge detected: {'yes' if result.detected else 'no'}")
    print(f"Bounding box: {list(result.bbox) if result.bbox else 'N/A'}")
    print(
        f"Reading: {result.reading:.2f}"
        if result.reading is not None
        else "Reading: N/A"
    )
    print(
        f"Confidence: {result.confidence:.2f}"
        if result.confidence is not None
        else "Confidence: N/A"
    )
    print(
        "Estimated pointer angle: "
        + (
            f"{result.angle_degrees:.2f} deg"
            if result.angle_degrees is not None
            else "N/A"
        )
    )
    print(f"Center method: {result.center_method or 'N/A'}")
    if result.unit:
        print(f"Reading unit: {result.unit}")
    if result.ocr_labels:
        print(f"OCR scale labels used: {list(result.ocr_labels)}")
    if result.rejected_numeric_labels:
        print(f"Rejected numeric OCR labels: {list(result.rejected_numeric_labels)}")
    if result.scale_rmse is not None:
        print(f"Scale fit RMSE: {result.scale_rmse:.3f}")
    if result.failure_reason:
        print(f"Failure reason: {result.failure_reason}")
    print(f"Level 1 / detection: {'PASS' if result.level1 else 'FAIL'}")
    print(f"Level 2 / pointer angle: {'PASS' if result.level2 else 'FAIL'}")
    print(f"Level 3 / scale reading: {'PASS' if result.level3 else 'FAIL'}")


def benchmark_reader(
    image_path: Path,
    factory: ReaderFactory,
    device: str,
    warmups: int,
    runs: int,
) -> tuple[GaugeResult, Metrics, float, Reader]:
    model_load_start = time.perf_counter_ns()
    reader = factory(device)
    model_load_ms = (time.perf_counter_ns() - model_load_start) / 1e6
    for _ in range(warmups):
        reader.read(image_path)
    results = [reader.read(image_path) for _ in range(runs)]
    metrics = {
        "preprocess": summary([item.timings.preprocess_ms for item in results]),
        "inference": summary([item.timings.inference_ms for item in results]),
        "postprocess": summary([item.timings.postprocess_ms for item in results]),
        "total": summary([item.timings.total_ms for item in results]),
    }
    return results[-1], metrics, model_load_ms, reader


def print_benchmark(record: BenchmarkRecord) -> None:
    print(f"Model load: {record.model_load_ms:.2f} ms (excluded from steady state)")
    print_result(record.result)
    print(f"Preprocess: {record.metrics['preprocess']['p50']:.2f} ms (p50)")
    print(
        f"Detection/model inference: {record.metrics['inference']['p50']:.2f} ms (p50)"
    )
    print(f"Postprocess/reading: {record.metrics['postprocess']['p50']:.2f} ms (p50)")
    print(f"Total steady-state latency: {record.metrics['total']['p50']:.2f} ms (p50)")
    print("Total latency statistics:")
    for key in ("mean", "p50", "p95", "min", "max"):
        print(f"  {key}: {record.metrics['total'][key]:.2f} ms")
    verdict = "PASS" if record.metrics["total"]["p95"] < 500.0 else "FAIL"
    print(f"Latency PoC: {verdict} (p95 < 500 ms)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Full input image; no ROI is accepted")
    parser.add_argument("--device", choices=("all", "cpu", "mps"), default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--units-per-major-segment",
        type=float,
        default=1.0,
        help="Calibrated value of one colored major segment (default: 1.0)",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output/result.jpg"
    )
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input image does not exist: {args.image}")
    if args.units_per_major_segment <= 0:
        parser.error("--units-per-major-segment must be positive")
    warmups = max(5, args.warmup)
    runs = max(20, args.runs)

    print(f"Machine: {platform.machine()} / {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Warm-up runs: {warmups}")
    print(f"Measured runs: {runs}")
    print(f"Units per colored major segment: {args.units_per_major_segment:g}")

    setup_start = time.perf_counter_ns()
    models = ensure_models(PROJECT_ROOT / "models")
    setup_ms = (time.perf_counter_ns() - setup_start) / 1e6
    print(f"Artifact/helper setup: {setup_ms:.2f} ms (excluded from steady state)")

    candidates: list[tuple[str, ReaderFactory]] = [
        (
            "Synanthropic two-stage YOLO Pose",
            lambda device: AnalogGaugeReader(
                models["corners-best.pt"], models["keypoints-best.pt"], device
            ),
        ),
        (
            "ETHZ detector/segmenter + PP-OCRv6 (RapidOCR/ONNX)",
            lambda device: EthzPaddleGaugeReader(
                models["ethz-gauge-detection.pt"],
                models["ethz-segmentation.pt"],
                device,
                args.units_per_major_segment,
            ),
        ),
    ]

    records: list[BenchmarkRecord] = []
    for model_name, factory in candidates:
        for device in available_devices(args.device):
            print(f"\n=== {model_name} / {device.upper()} ===")
            result, metrics, model_load_ms, reader = benchmark_reader(
                args.image, factory, device, warmups, runs
            )
            record = BenchmarkRecord(
                model_name, device, result, metrics, model_load_ms, reader
            )
            records.append(record)
            print_benchmark(record)

    best = min(
        records,
        key=lambda record: (
            not record.result.level3,
            not record.result.level2,
            not record.result.level1,
            record.metrics["total"]["p95"],
        ),
    )
    annotate(args.image, best.result, args.output)
    print(f"\nVisualization: {args.output}")
    save_artifacts = getattr(best.reader, "save_artifacts", None)
    if save_artifacts is not None:
        artifacts = save_artifacts(args.output)
        if artifacts is not None:
            print(f"Rectified dial: {artifacts[0]}")
            print(f"Unwrapped scale ring: {artifacts[1]}")
    print("\n| Model / Project | Device | Detection | Pointer | Reading | p50 | p95 |")
    print("| --- | --- | --- | --- | --- | ---: | ---: |")
    for record in records:
        print(
            f"| {record.model_name} | {record.device.upper()} | "
            f"{'PASS' if record.result.level1 else 'FAIL'} | "
            f"{'PASS' if record.result.level2 else 'FAIL'} | "
            f"{'PASS' if record.result.level3 else 'FAIL'} | "
            f"{record.metrics['total']['p50']:.2f} ms | "
            f"{record.metrics['total']['p95']:.2f} ms |"
        )
    print(
        f"\nBest usable result: {best.model_name} / {best.device.upper()} / "
        f"reading {best.result.reading:.2f} / "
        f"p50 {best.metrics['total']['p50']:.2f} ms / "
        f"p95 {best.metrics['total']['p95']:.2f} ms"
        if best.result.reading is not None
        else f"\nNo candidate produced a physical reading; best stage result: {best.model_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
