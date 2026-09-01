"""Benchmark every input image and write an auditable self-contained HTML report."""

from __future__ import annotations

import argparse
import base64
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import cast

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

from benchmark import Metrics, available_devices, summary
from src.gauge_reader import GaugeResult, StageTimings
from src.instrument_metadata import InstrumentMetadataCatalog
from src.instrument_reading import InstrumentReadingInterpreter
from src.model_store import ensure_models
from src.rapidocr_reader import EthzPaddleGaugeReader

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME = "ETHZ 表盘检测/指针分割 + PP-OCRv6 + 通用几何回退"


@dataclass(frozen=True)
class RegressionRecord:
    image_path: Path
    device: str
    result: GaugeResult
    metrics: Metrics
    annotated_path: Path


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def benchmark_image(
    reader: EthzPaddleGaugeReader,
    image_path: Path,
    warmups: int,
    runs: int,
) -> tuple[GaugeResult, Metrics]:
    for _ in range(warmups):
        reader.read(image_path)
    results = [reader.read(image_path) for _ in range(runs)]
    metrics = {
        "preprocess": summary([item.timings.preprocess_ms for item in results]),
        "inference": summary([item.timings.inference_ms for item in results]),
        "postprocess": summary([item.timings.postprocess_ms for item in results]),
        "total": summary([item.timings.total_ms for item in results]),
    }
    return results[-1], metrics


def chinese_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("未找到可用于报告中文标注的字体")


def annotate_chinese(image_path: Path, result: GaugeResult, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取报告图片：{image_path}")
    height, width = image.shape[:2]
    thickness = max(2, round(min(height, width) / 300))
    if result.bbox is not None:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), thickness)
    if result.center is not None:
        center = tuple(round(value) for value in result.center)
        cv2.circle(image, center, thickness * 3, (255, 80, 0), -1)
        if result.pointer_tip is not None:
            tip = tuple(round(value) for value in result.pointer_tip)
            cv2.arrowedLine(
                image,
                center,
                tip,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
                tipLength=0.18,
            )

    angle = format_value(result.angle_degrees, "°")
    reading = format_reading(result)
    lines = [
        f"表盘检测：{'成功' if result.level1 else '失败'}",
        f"指针角度：{angle}",
        f"实际读数：{reading}",
        f"三级读取：{'通过' if result.level3 else '失败'}",
    ]
    if result.raw_reading is not None and result.raw_reading != result.reading:
        lines.insert(3, f"视觉原始值：{format_value(result.raw_reading)}")
    if result.instrument_type_id:
        lines.insert(0, f"仪表类型：{result.instrument_type_id}")
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = chinese_font(max(18, min(42, round(width / 55))))
    text = "\n".join(lines)
    text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=5)
    padding = max(10, font.size // 2)
    panel = (
        12,
        12,
        12 + text_box[2] + 2 * padding,
        12 + text_box[3] + 2 * padding,
    )
    draw.rounded_rectangle(panel, radius=8, fill=(0, 0, 0, 165))
    draw.multiline_text(
        (12 + padding, 12 + padding),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(output_path, quality=90)


def image_data_uri(path: Path, max_width: int = 720) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot decode report image: {path}")
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(
            image,
            (max_width, max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not encoded:
        raise RuntimeError(f"Cannot encode report image: {path}")
    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def result_payload(record: RegressionRecord) -> dict[str, object]:
    result = record.result
    return {
        "image": record.image_path.name,
        "device": record.device,
        "detected": result.detected,
        "pointer_found": result.pointer_found,
        "level1": result.level1,
        "level2": result.level2,
        "level3": result.level3,
        "bbox": list(result.bbox) if result.bbox else None,
        "reading": result.reading,
        "reading_candidates": list(result.reading_candidates),
        "raw_reading": result.raw_reading,
        "unit": result.unit,
        "instrument_type_id": result.instrument_type_id,
        "readout_channel_id": result.readout_channel_id,
        "interpretation_method": result.interpretation_method,
        "angle_degrees": result.angle_degrees,
        "confidence": result.confidence,
        "center_method": result.center_method,
        "center": list(result.center) if result.center else None,
        "pointer_tip": list(result.pointer_tip) if result.pointer_tip else None,
        "ocr_labels": list(result.ocr_labels),
        "failure_reason": result.failure_reason,
        "metrics_ms": record.metrics,
        "annotated_image": str(record.annotated_path),
    }


def badge(passed: bool, label: str) -> str:
    state = "pass" if passed else "fail"
    return f'<span class="badge {state}">{escape(label)} {"通过" if passed else "失败"}</span>'


def format_value(value: float | str | None, suffix: str = "") -> str:
    if value is None:
        return "无"
    if isinstance(value, str):
        return f"{value}{suffix}"
    return f"{value:.2f}{suffix}"


def format_reading(result: GaugeResult) -> str:
    if result.reading is not None:
        return format_value(result.reading)
    if result.reading_candidates:
        candidates = "/".join(f"{value:.2f}" for value in result.reading_candidates)
        return f"{candidates}（量程待确认）"
    return "无"


def method_label(method: str | None) -> str:
    if method is None:
        return "无"
    translations = {
        "model-segmentation": "预训练指针分割",
        "colored-hub": "彩色轴心",
        "line-segment": "连续直线",
        "edge-ellipse": "边缘椭圆拟合",
        "affine-rectification": "仿射校正",
        "color-segment-scale": "彩色分段刻度",
    }
    return " + ".join(translations.get(part, part) for part in method.split("+"))


def failure_label(reason: str | None) -> str:
    if reason is None:
        return "无"
    translations = {
        "OCR found only 0 numeric scale candidates": "OCR 未找到可靠的数字刻度候选",
        "Pointer segmenter returned no valid mask": "指针分割模型未返回有效掩膜",
        "No geometrically valid gauge candidate": "未找到几何上有效的表盘候选",
    }
    return translations.get(reason, reason)


def build_html(
    records: list[RegressionRecord],
    devices: list[str],
    model_load_ms: dict[str, float],
    warmups: int,
    runs: int,
    generated_at: str,
) -> str:
    by_image: dict[str, dict[str, RegressionRecord]] = {}
    for record in records:
        by_image.setdefault(record.image_path.name, {})[record.device] = record

    total_cases = len(records)
    level3_passes = sum(record.result.level3 for record in records)
    latency_passes = sum(record.metrics["total"]["p95"] < 500.0 for record in records)
    cards: list[str] = []
    for image_name in sorted(by_image, key=lambda name: natural_key(Path(name))):
        device_records = by_image[image_name]
        reference = device_records[devices[0]]
        visual = image_data_uri(reference.annotated_path)
        original = image_data_uri(reference.image_path)
        rows: list[str] = []
        for device in devices:
            record = device_records[device]
            result = record.result
            metrics = record.metrics
            reading_text = format_reading(result)
            if result.raw_reading is not None and result.raw_reading != result.reading:
                reading_text += (
                    f"<br><small>原始 {format_value(result.raw_reading)}</small>"
                )
            rows.append(
                "<tr>"
                f"<td><strong>{escape(device.upper())}</strong></td>"
                f"<td>{badge(result.level1, '一级')} {badge(result.level2, '二级')} {badge(result.level3, '三级')}</td>"
                f"<td>{reading_text}</td>"
                f"<td>{format_value(result.angle_degrees, '°')}</td>"
                f"<td>{metrics['total']['mean']:.2f}</td>"
                f"<td>{metrics['total']['p50']:.2f}</td>"
                f"<td>{metrics['total']['p95']:.2f}</td>"
                f"<td>{metrics['total']['min']:.2f}</td>"
                f"<td>{metrics['total']['max']:.2f}</td>"
                f"<td>{badge(metrics['total']['p95'] < 500.0, '延迟')}</td>"
                "</tr>"
            )
        method = escape(method_label(reference.result.center_method))
        failure = escape(failure_label(reference.result.failure_reason))
        cards.append(
            f"""
            <section class="case">
              <div class="case-head">
                <div><span class="eyebrow">完整图片输入</span><h2>{escape(image_name)}</h2></div>
                <div class="method">识别方法：{method}<br>失败原因：{failure}</div>
              </div>
              <div class="images">
                <figure><img src="{original}" alt="Original {escape(image_name)}"><figcaption>原始整图缩略图</figcaption></figure>
                <figure><img src="{visual}" alt="Annotated {escape(image_name)}"><figcaption>CPU 识别可视化</figcaption></figure>
              </div>
              <div class="table-wrap"><table>
                <thead><tr><th>设备</th><th>识别层级</th><th>读数</th><th>指针角度</th><th>平均耗时</th><th>p50</th><th>p95</th><th>最小耗时</th><th>最大耗时</th><th>&lt;500毫秒</th></tr></thead>
                <tbody>{"".join(rows)}</tbody>
              </table></div>
            </section>
            """
        )

    load_text = " · ".join(
        f"{device.upper()} 模型加载 {model_load_ms[device]:.2f} 毫秒"
        for device in devices
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>指针式仪表回归测试报告</title>
<style>
:root {{--ink:#18211d;--muted:#63716a;--paper:#f3f1e9;--panel:#fffef8;--line:#d8d6cc;--green:#137a4b;--red:#b44035;--blue:#2459a6}}
* {{box-sizing:border-box}} body {{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main {{max-width:1480px;margin:auto;padding:40px 28px 80px}} h1 {{font-size:clamp(32px,5vw,64px);line-height:1;margin:8px 0 18px;letter-spacing:-.04em}}
h2 {{margin:4px 0 0;font-size:26px}} .eyebrow {{font:700 12px/1.2 ui-monospace,monospace;color:var(--blue);letter-spacing:.12em}}
.lead {{max-width:920px;color:var(--muted);font-size:17px;line-height:1.65}} .summary {{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:28px 0}}
.stat,.case {{background:var(--panel);border:1px solid var(--line);box-shadow:0 5px 18px rgba(22,31,27,.05)}} .stat {{padding:18px 20px}} .stat strong {{display:block;font-size:30px}} .stat span {{color:var(--muted)}}
.case {{margin:18px 0;padding:22px}} .case-head {{display:flex;gap:20px;justify-content:space-between;align-items:flex-start;margin-bottom:16px}} .method {{max-width:620px;color:var(--muted);font:12px/1.55 ui-monospace,monospace;text-align:right}}
.images {{display:grid;grid-template-columns:1fr 1fr;gap:14px}} figure {{margin:0;background:#171b19;border-radius:6px;overflow:hidden}} figure img {{display:block;width:100%;height:270px;object-fit:contain}} figcaption {{padding:9px 12px;background:#252c28;color:#dfe7e2;font-size:12px}}
.table-wrap {{overflow:auto;margin-top:16px}} table {{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}} th,td {{padding:10px 9px;border-top:1px solid var(--line);text-align:right}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{text-align:left}}
.badge {{display:inline-block;padding:3px 6px;border-radius:3px;font:700 10px/1.2 ui-monospace,monospace}} .pass {{background:#d9f0e3;color:var(--green)}} .fail {{background:#f5ded9;color:var(--red)}} footer {{margin-top:28px;color:var(--muted);font-size:12px;line-height:1.7}}
@media(max-width:760px) {{main {{padding:24px 12px 50px}} .summary,.images {{grid-template-columns:1fr}} .case-head {{display:block}} .method {{margin-top:10px;text-align:left}} figure img {{height:210px}}}}
</style>
</head>
<body><main>
<span class="eyebrow">回归测试 · {escape(generated_at)}</span>
<h1>指针式仪表<br>回归测试报告</h1>
<p class="lead">模型：{escape(MODEL_NAME)}。每张图片均以完整图片作为输入；每个设备预热 {warmups} 次，正式执行 {runs} 次。模型加载时间单独统计，不计入稳态延迟。识别层级与延迟判定分开报告。</p>
<div class="summary">
  <div class="stat"><strong>{len(by_image)}</strong><span>扫描图片</span></div>
  <div class="stat"><strong>{level3_passes}/{total_cases}</strong><span>三级实际读数通过（设备测试项）</span></div>
  <div class="stat"><strong>{latency_passes}/{total_cases}</strong><span>p95 &lt; 500 毫秒</span></div>
</div>
{"".join(cards)}
<footer>运行环境：{escape(platform.machine())} · {escape(platform.platform())} · Python {escape(sys.version.split()[0])} · PyTorch {escape(torch.__version__)}<br>{escape(load_text)}<br>注意：三级通过表示程序产生了实际读数，不等同于已经获得人工标注的准确性证明。</footer>
</main></body></html>"""


def load_existing_results(
    json_path: Path,
) -> tuple[list[RegressionRecord], list[str], dict[str, float], int, int, str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    records: list[RegressionRecord] = []
    for item in payload["records"]:
        raw_bbox = item["bbox"]
        bbox = cast(
            tuple[int, int, int, int] | None,
            tuple(raw_bbox) if raw_bbox is not None else None,
        )
        raw_center = item.get("center")
        center = cast(
            tuple[float, float] | None,
            tuple(raw_center) if raw_center is not None else None,
        )
        raw_tip = item.get("pointer_tip")
        pointer_tip = cast(
            tuple[float, float] | None,
            tuple(raw_tip) if raw_tip is not None else None,
        )
        result = GaugeResult(
            detected=bool(item["detected"]),
            bbox=bbox,
            detection_confidence=None,
            pointer_found=bool(item["pointer_found"]),
            center=center,
            pointer_tip=pointer_tip,
            angle_degrees=item["angle_degrees"],
            sweep_fraction=None,
            reading=item["reading"],
            unit=item.get("unit"),
            confidence=item["confidence"],
            center_method=item["center_method"],
            timings=StageTimings(0.0, 0.0, 0.0),
            ocr_labels=tuple(item["ocr_labels"]),
            failure_reason=item["failure_reason"],
            raw_reading=item.get("raw_reading"),
            instrument_type_id=item.get("instrument_type_id"),
            readout_channel_id=item.get("readout_channel_id"),
            interpretation_method=item.get("interpretation_method"),
            reading_candidates=tuple(item.get("reading_candidates", [])),
        )
        annotated_path = Path(item["annotated_image"])
        if not annotated_path.is_absolute():
            annotated_path = PROJECT_ROOT / annotated_path
        records.append(
            RegressionRecord(
                PROJECT_ROOT / "input" / item["image"],
                item["device"],
                result,
                cast(Metrics, item["metrics_ms"]),
                annotated_path,
            )
        )
    model_load_ms = {
        str(device): float(value) for device, value in payload["model_load_ms"].items()
    }
    return (
        records,
        list(model_load_ms),
        model_load_ms,
        int(payload["warmups_per_image"]),
        int(payload["measured_runs_per_image"]),
        str(payload["generated_at"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--device", choices=("all", "cpu", "mps"), default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--reuse-json",
        type=Path,
        help="只用已有 JSON 测量数据重新渲染报告，不重新运行推理",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output/regression-report.html"
    )
    args = parser.parse_args()
    if args.reuse_json is not None:
        records, devices, model_load_ms, warmups, runs, generated_at = (
            load_existing_results(args.reuse_json)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            build_html(records, devices, model_load_ms, warmups, runs, generated_at),
            encoding="utf-8",
        )
        print(f"HTML: {args.output}（复用 {args.reuse_json} 的测量数据）")
        return 0

    warmups = max(5, args.warmup)
    runs = max(20, args.runs)
    image_paths = args.images or sorted(
        (PROJECT_ROOT / "input").glob("gauge*"), key=natural_key
    )
    if not image_paths or any(not path.is_file() for path in image_paths):
        parser.error("Every input image must exist")

    models = ensure_models(PROJECT_ROOT / "models")
    reading_interpreter = InstrumentReadingInterpreter(InstrumentMetadataCatalog.load())
    devices = available_devices(args.device)
    asset_dir = args.output.with_name(f"{args.output.stem}-assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    records: list[RegressionRecord] = []
    model_load_ms: dict[str, float] = {}
    for device in devices:
        load_start = time.perf_counter_ns()
        reader = EthzPaddleGaugeReader(
            models["ethz-gauge-detection.pt"],
            models["ethz-segmentation.pt"],
            device,
            reading_interpreter=reading_interpreter,
        )
        model_load_ms[device] = (time.perf_counter_ns() - load_start) / 1e6
        for image_path in image_paths:
            print(f"[{device.upper()}] {image_path.name}", flush=True)
            result, metrics = benchmark_image(reader, image_path, warmups, runs)
            annotated_path = asset_dir / f"{image_path.stem}-{device}.jpg"
            annotate_chinese(image_path, result, annotated_path)
            records.append(
                RegressionRecord(image_path, device, result, metrics, annotated_path)
            )
            print(
                f"  L1={result.level1} L2={result.level2} L3={result.level3} "
                f"reading={result.reading} p50={metrics['total']['p50']:.2f} "
                f"p95={metrics['total']['p95']:.2f} ms",
                flush=True,
            )

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_html(records, devices, model_load_ms, warmups, runs, generated_at),
        encoding="utf-8",
    )
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "model": MODEL_NAME,
                "warmups_per_image": warmups,
                "measured_runs_per_image": runs,
                "model_load_ms": model_load_ms,
                "records": [result_payload(record) for record in records],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"HTML: {args.output}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
