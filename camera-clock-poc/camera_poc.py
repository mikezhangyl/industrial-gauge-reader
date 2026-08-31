"""Run the isolated MacBook-camera analog-clock demonstration."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_clock_poc.clock_demo.alarm import ClockDemoAlarm
from camera_clock_poc.clock_demo.object_detector import YoloClockDetector
from camera_clock_poc.clock_demo.overlay import render_overlay
from camera_clock_poc.clock_demo.reader import ClockSecondHandReader
from camera_clock_poc.clock_demo.report import write_clock_report
from camera_clock_poc.clock_demo.scale_reference import RapidOcrClockOrientation
from camera_clock_poc.reusable.capture import (
    LatestFrameCamera,
    parse_source,
)
from camera_clock_poc.reusable.privacy import protect_background
from camera_clock_poc.reusable.session import SessionRecorder


def expected_system_seconds(captured_at: datetime, offset: float) -> float:
    return (captured_at.second + captured_at.microsecond / 1_000_000.0 + offset) % 60.0


def main() -> int:
    parser = argparse.ArgumentParser(description="MacBook摄像头闹钟秒针PoC")
    parser.add_argument("--source", default="0", help="摄像头编号或RTSP地址")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration", type=float, default=0.0, help="0表示手动结束")
    parser.add_argument("--max-samples", type=int, default=0, help="0表示不限次数")
    parser.add_argument("--headless", action="store_true", help="不显示实时窗口")
    parser.add_argument("--alarm-start", type=float, default=50.0)
    parser.add_argument("--alarm-confirmations", type=int, default=2)
    parser.add_argument("--snapshot-interval", type=float, default=5.0)
    parser.add_argument(
        "--archive-all-raw-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存每个采样未标注帧，便于离线重放（默认开启）",
    )
    parser.add_argument(
        "--privacy-blur",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅检测成功后保留仪表清晰并虚化背景；漏检帧保持清晰（默认开启）",
    )
    parser.add_argument("--compare-system-time", action="store_true")
    parser.add_argument("--clock-offset-seconds", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "sessions")
    parser.add_argument(
        "--clock-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "yolo11n.pt",
        help="预训练COCO闹钟检测权重",
    )
    args = parser.parse_args()
    if args.sample_fps <= 0:
        parser.error("--sample-fps 必须大于0")
    if args.duration < 0 or args.max_samples < 0:
        parser.error("--duration 和 --max-samples 不能为负数")

    session_name = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    session_dir = args.output_root / session_name
    recorder = SessionRecorder(
        session_dir,
        args.snapshot_interval,
        archive_all_raw_frames=args.archive_all_raw_frames,
        privacy_applied=args.privacy_blur,
    )
    if not args.clock_model.is_file():
        parser.error(f"找不到闹钟检测权重：{args.clock_model}")
    model_started = time.perf_counter()
    detector = YoloClockDetector(args.clock_model)
    orientation_estimator = RapidOcrClockOrientation()
    reader = ClockSecondHandReader(detector, orientation_estimator)
    model_load_ms = (time.perf_counter() - model_started) * 1000.0
    alarm = ClockDemoAlarm(args.alarm_start, args.alarm_confirmations)
    source = parse_source(args.source)
    started = time.monotonic()
    next_sample = started
    last_sequence = 0
    samples = 0
    print(f"实验目录：{session_dir}")
    print(f"模型加载：{model_load_ms:.1f}ms（不计入逐帧耗时）")
    print(
        "隐私虚化：条件开启（仅检测成功后虚化背景；漏检帧保留清晰原图）"
        if args.privacy_blur
        else "隐私虚化：关闭"
    )
    print("按 q 结束，按 s 保存当前截图。")

    try:
        with LatestFrameCamera(source, args.width, args.height) as camera:
            while True:
                now = time.monotonic()
                if args.duration > 0 and now - started >= args.duration:
                    break
                if args.max_samples > 0 and samples >= args.max_samples:
                    break
                if now < next_sample:
                    time.sleep(min(0.02, next_sample - now))
                    continue
                captured = camera.read_latest(last_sequence, timeout=3.0)
                last_sequence = captured.sequence
                observation = reader.read(captured.image, captured.captured_at)
                decision = alarm.update(observation.value)
                expected = (
                    expected_system_seconds(
                        captured.captured_at, args.clock_offset_seconds
                    )
                    if args.compare_system_time
                    else None
                )
                frame_privacy_applied = bool(
                    args.privacy_blur and observation.detected and observation.bbox
                )
                privacy_started = time.perf_counter()
                presentation_frame = (
                    protect_background(captured.image, observation.bbox)
                    if frame_privacy_applied
                    else captured.image
                )
                if frame_privacy_applied:
                    privacy_ms = (time.perf_counter() - privacy_started) * 1000.0
                    observation = replace(
                        observation,
                        processing_ms=observation.processing_ms + privacy_ms,
                        method=f"{observation.method}+隐私背景虚化",
                    )
                annotated = render_overlay(
                    presentation_frame, observation, decision, expected
                )
                manual_snapshot = False
                quit_requested = False
                if not args.headless:
                    cv2.imshow("闹钟摄像头 PoC", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    manual_snapshot = key == ord("s")
                    quit_requested = key == ord("q")
                recorder.record(
                    presentation_frame,
                    annotated,
                    observation,
                    decision.state.value,
                    decision.changed,
                    expected,
                    manual_snapshot,
                    privacy_applied=frame_privacy_applied,
                )
                samples += 1
                seconds = (
                    "无" if observation.value is None else f"{observation.value:.2f}"
                )
                print(
                    f"{observation.captured_at.isoformat(timespec='seconds')} "
                    f"检测={observation.detected} 秒数={seconds} "
                    f"报警={decision.state.value} "
                    f"耗时={observation.processing_ms:.1f}ms",
                    flush=True,
                )
                if quit_requested:
                    break
                next_sample += 1.0 / args.sample_fps
    except (RuntimeError, TimeoutError) as error:
        print(f"摄像头运行失败：{error}", file=sys.stderr)
        if not recorder.records:
            return 2
    except KeyboardInterrupt:
        print("收到中断，正在生成报告。")
    finally:
        cv2.destroyAllWindows()

    report_path = write_clock_report(
        session_dir,
        recorder.records,
        str(source),
        args.sample_fps,
        privacy_mode=args.privacy_blur,
    )
    print(f"JSONL：{recorder.jsonl_path}")
    print(f"中文报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
