"""Analyze instrument images and emit automated HTML plus audit JSON."""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.batch_io import (
    NORMALIZED_MAX_EDGE,
    NormalizedBatchImage,
    annotated_preview_data_uri,
    default_report_path,
    discover_input_images,
    normalize_batch_images,
    preview_crop_bbox,
)
from src.instrument_image import (
    GENERATED_VISUALIZATION_FAILURE,
    POINTER_DISPLAY_TYPES,
    MetadataAwareImageAnalyzer,
)
from src.instrument_metadata import InstrumentMetadataCatalog, ReadoutChannel
from src.instrument_observations import (
    ConfirmedReadout,
    InstrumentObservationCatalog,
)
from src.instrument_reading import InstrumentReadingInterpreter
from src.model_store import ensure_models
from src.rapidocr_reader import EthzPaddleGaugeReader

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run metadata-aware automated analysis over instrument images"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One image directory, or an ordered list of image files",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSON output path; HTML is written beside it with the same stem. "
            "A directory input defaults to output/<directory>/instrument-report.*"
        ),
    )
    parser.add_argument(
        "--require-pointer-match",
        action="store_true",
        help=(
            "Optional labeled-regression gate: return non-zero unless every "
            "locally confirmed pointer channel matches the automated result"
        ),
    )
    args = parser.parse_args()
    try:
        image_paths, input_directory = discover_input_images(args.inputs)
    except ValueError as error:
        parser.error(str(error))
    output_path = args.output or default_report_path(PROJECT_ROOT, input_directory)

    metadata_catalog = InstrumentMetadataCatalog.load()
    observation_catalog = InstrumentObservationCatalog.load(metadata_catalog)
    models = ensure_models(PROJECT_ROOT / "models")
    reader = EthzPaddleGaugeReader(
        models["ethz-gauge-detection.pt"],
        models["ethz-segmentation.pt"],
        args.device,
        reading_interpreter=InstrumentReadingInterpreter(metadata_catalog),
    )
    analyzer = MetadataAwareImageAnalyzer(reader, metadata_catalog)
    with TemporaryDirectory(prefix="instrument-batch-normalized-") as temp_dir:
        normalized_images = normalize_batch_images(
            image_paths,
            Path(temp_dir),
            max_edge=NORMALIZED_MAX_EDGE,
        )
        records = [
            evaluate_image(
                batch_image,
                analyzer,
                metadata_catalog,
                observation_catalog,
            )
            for batch_image in normalized_images
        ]
        pointer_acceptance = summarize_pointer_acceptance(records, metadata_catalog)
        automated_summary = summarize_automated(records)
        payload = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "device": args.device,
            "input_contract": {
                "mode": "directory" if input_directory is not None else "files",
                "batch_name": input_directory.name
                if input_directory is not None
                else None,
                "image_order": [path.name for path in image_paths],
                "normalization": {
                    "exif_orientation": True,
                    "color_mode": "RGB",
                    "maximum_edge_pixels": NORMALIZED_MAX_EDGE,
                    "analysis_format": "PNG",
                },
            },
            "runtime": runtime_fingerprint(models),
            "separation_rule": (
                "Automated results remain independent; user-confirmed values own the "
                "final reviewed value and never become model input."
            ),
            "summary": summarize(records),
            "automated_summary": automated_summary,
            "pointer_acceptance": pointer_acceptance,
            "records": records,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path = output_path.with_suffix(".html")
        html_path.write_text(render_html(payload, normalized_images), encoding="utf-8")
    reader.close()
    del analyzer
    del reader
    gc.collect()
    print(json.dumps(automated_summary, ensure_ascii=False, indent=2))
    if input_directory is not None:
        print(f"Input directory: {input_directory.resolve()}")
    print(f"JSON: {output_path.resolve()}")
    print(f"HTML: {html_path.resolve()}")
    rejected_visualizations = [
        record["image"]
        for record in records
        if record["analysis_failure_reason"] == GENERATED_VISUALIZATION_FAILURE
    ]
    if rejected_visualizations:
        print("Rejected generated visualization images; use original photos:")
        for image_name in rejected_visualizations:
            print(f"- {image_name}")
        return 2
    if (
        args.require_pointer_match
        and not pointer_acceptance["reviewed_pointer_channels"]
    ):
        print("Pointer acceptance unavailable: no confirmed pointer channels")
        return 1
    if args.require_pointer_match and pointer_acceptance["failures"]:
        print("Pointer acceptance failures:")
        for failure in pointer_acceptance["failures"]:
            print(
                f"- {failure['image']} {failure['instance_id']} "
                f"{failure['channel_id']}: {failure['comparison']}"
            )
        return 1
    return 0


def evaluate_image(
    batch_image: NormalizedBatchImage,
    analyzer: MetadataAwareImageAnalyzer,
    metadata_catalog: InstrumentMetadataCatalog,
    observation_catalog: InstrumentObservationCatalog,
) -> dict[str, Any]:
    analysis = analyzer.analyze(batch_image.analysis_path)
    observation = observation_catalog.for_image(batch_image.source_path)
    automated_by_key = {
        (channel.instance_id, channel.channel_id): channel
        for channel in analysis.channels
    }
    confirmed_by_key = (
        {
            (readout.instance_id, readout.channel_id): readout
            for readout in observation.readouts
        }
        if observation is not None
        else {}
    )
    keys = sorted(set(automated_by_key) | set(confirmed_by_key))
    channel_records: list[dict[str, Any]] = []
    type_id = analysis.instrument_type_id or (
        observation.instrument_type_id if observation is not None else None
    )
    for key in keys:
        automated = automated_by_key.get(key)
        confirmed = confirmed_by_key.get(key)
        try:
            channel = (
                metadata_catalog.get(type_id).channel(key[1])
                if type_id is not None
                else None
            )
        except KeyError:
            channel = None
        comparison = compare_channel(automated, confirmed, channel)
        channel_records.append(
            {
                "instance_id": key[0],
                "channel_id": key[1],
                "automated": asdict(automated) if automated is not None else None,
                "confirmed": asdict(confirmed) if confirmed is not None else None,
                "comparison": comparison,
                "final": final_value(automated, confirmed),
            }
        )
    detections = _serialize_detections(analysis, batch_image.normalized_size)
    return {
        "image": batch_image.source_path.name,
        "image_sha256": batch_image.source_sha256,
        "normalized_image_sha256": analysis.image_sha256,
        "source_dimensions": list(batch_image.source_size),
        "oriented_dimensions": list(batch_image.oriented_size),
        "analysis_dimensions": list(batch_image.normalized_size),
        "instrument_type_id": analysis.instrument_type_id,
        "expected_instrument_type_id": (
            observation.instrument_type_id if observation is not None else None
        ),
        "type_match": (
            analysis.instrument_type_id == observation.instrument_type_id
            if observation is not None
            else None
        ),
        "observation_id": observation.observation_id if observation else None,
        "analysis_failure_reason": analysis.failure_reason,
        "detections": detections,
        "preview_crop_bbox": list(
            preview_crop_bbox(detections, batch_image.normalized_size)
        ),
        "channels": channel_records,
    }


def _serialize_detections(
    analysis: Any,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for index, result in enumerate(analysis.pointer_results):
        if not result.detected or result.bbox is None:
            continue
        detector_bbox = tuple(int(value) for value in result.bbox)
        bbox = detector_bbox
        bbox_method = "detector"
        if (
            result.center_method is not None
            and "pointer-aligned-outer-label-ocr" in result.center_method
        ):
            bbox = _concentric_outer_label_bbox(result, image_size)
            bbox_method = "concentric_outer_label"
        instance_id = (
            analysis.instances[index]
            if index < len(analysis.instances)
            else f"instance_{index + 1}"
        )
        detections.append(
            {
                "instance_id": instance_id,
                "bbox": list(bbox),
                "detector_bbox": list(detector_bbox),
                "bbox_method": bbox_method,
                "pointer_found": result.pointer_found,
                "center": list(result.center) if result.center is not None else None,
                "pointer_tip": (
                    list(result.pointer_tip) if result.pointer_tip is not None else None
                ),
                "angle_degrees": result.angle_degrees,
                "detection_confidence": result.detection_confidence,
            }
        )
    return detections


def _concentric_outer_label_bbox(
    result: Any,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Expand an inner-dial detection to include its concentric OCR label ring."""
    if result.bbox is None:
        raise ValueError("Outer-label display region requires a detector bbox")
    x1, y1, x2, y2 = (int(value) for value in result.bbox)
    center_x, center_y = (
        result.center if result.center is not None else ((x1 + x2) / 2, (y1 + y2) / 2)
    )
    inner_dial_size = max(x2 - x1, y2 - y1)
    outer_extent = inner_dial_size * 1.20
    image_width, image_height = image_size
    return (
        max(0, round(center_x - outer_extent)),
        max(0, round(center_y - outer_extent)),
        min(image_width, round(center_x + outer_extent)),
        min(image_height, round(center_y + outer_extent)),
    )


def runtime_fingerprint(models: dict[str, Path]) -> dict[str, Any]:
    """Record enough runtime identity to compare two machines."""
    packages = {}
    for package in (
        "numpy",
        "opencv-python",
        "onnxruntime",
        "rapidocr",
        "torch",
        "torchvision",
        "ultralytics",
        "pillow",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        git_worktree_dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(PROJECT_ROOT.parent),
                    "status",
                    "--porcelain",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        git_commit = None
        git_worktree_dirty = None
    return {
        "git_commit": git_commit,
        "git_worktree_dirty": git_worktree_dirty,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "models": {
            name: {
                "filename": path.name,
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sorted(models.items())
        },
        "executable_name": Path(sys.executable).name,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_channel(
    automated: Any,
    confirmed: ConfirmedReadout | None,
    channel: ReadoutChannel | None,
) -> str:
    if confirmed is None:
        return "not_reviewed"
    if automated is None or automated.status not in {"recognized", "ambiguous"}:
        return "automated_not_available"
    if confirmed.confirmed_candidates:
        observed = tuple(float(value) for value in automated.candidates)
        expected = tuple(float(value) for value in confirmed.confirmed_candidates)
        return "match" if _same_candidates(observed, expected) else "mismatch"
    expected_value = confirmed.confirmed_value
    observed_value = automated.value
    if observed_value is None:
        return "automated_not_available"
    if isinstance(expected_value, int | float) and isinstance(
        observed_value, int | float
    ):
        tolerance = _comparison_tolerance(channel)
        return (
            "match"
            if abs(float(observed_value) - float(expected_value)) <= tolerance
            else "mismatch"
        )
    return "match" if observed_value == expected_value else "mismatch"


def final_value(automated: Any, confirmed: ConfirmedReadout | None) -> dict[str, Any]:
    if confirmed is not None:
        return {
            "source": "human_confirmation",
            "value": confirmed.confirmed_value,
            "candidates": list(confirmed.confirmed_candidates),
            "unit": confirmed.unit,
        }
    if automated is None:
        return {"source": "unavailable", "value": None, "candidates": [], "unit": None}
    return {
        "source": "automated",
        "value": automated.value,
        "candidates": list(automated.candidates),
        "unit": automated.unit,
    }


def _same_candidates(first: tuple[float, ...], second: tuple[float, ...]) -> bool:
    if len(first) != len(second):
        return False
    return all(
        abs(left - right) <= 1e-6
        for left, right in zip(sorted(first), sorted(second), strict=True)
    )


def _comparison_tolerance(channel: ReadoutChannel | None) -> float:
    if channel is None or not channel.scales:
        return 1e-6
    return max(scale.minor_division for scale in channel.scales) / 2.0


def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    comparisons = [
        channel["comparison"]
        for record in records
        for channel in record["channels"]
        if channel["confirmed"] is not None
    ]
    return {
        "images": len(records),
        "type_matches": sum(record["type_match"] is True for record in records),
        "confirmed_channels": len(comparisons),
        "automated_matches": comparisons.count("match"),
        "automated_mismatches": comparisons.count("mismatch"),
        "automated_not_available": comparisons.count("automated_not_available"),
    }


def summarize_automated(records: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize only results produced by the current automated run."""
    statuses = [
        channel["automated"]["status"]
        for record in records
        for channel in record["channels"]
        if channel.get("automated") is not None
    ]
    return {
        "images": len(records),
        "instrument_types_recognized": sum(
            record.get("instrument_type_id") is not None for record in records
        ),
        "channels": len(statuses),
        "recognized": statuses.count("recognized"),
        "ambiguous": statuses.count("ambiguous"),
        "not_recognized": statuses.count("not_recognized"),
        "analysis_failures": sum(
            record.get("analysis_failure_reason") is not None for record in records
        ),
    }


def summarize_pointer_acceptance(
    records: list[dict[str, Any]],
    metadata_catalog: InstrumentMetadataCatalog,
) -> dict[str, Any]:
    """Summarize only confirmed pointer channels for a hard regression gate."""
    reviewed = 0
    matches = 0
    failures: list[dict[str, str]] = []
    for record in records:
        type_id = record["expected_instrument_type_id"] or record["instrument_type_id"]
        if type_id is None:
            continue
        pointer_channel_ids = {
            channel.channel_id
            for channel in metadata_catalog.get(type_id).readout_channels
            if channel.display_type in POINTER_DISPLAY_TYPES
        }
        for channel in record["channels"]:
            if (
                channel["channel_id"] not in pointer_channel_ids
                or channel["confirmed"] is None
            ):
                continue
            reviewed += 1
            if channel["comparison"] == "match":
                matches += 1
                continue
            failures.append(
                {
                    "image": record["image"],
                    "instance_id": channel["instance_id"],
                    "channel_id": channel["channel_id"],
                    "comparison": channel["comparison"],
                }
            )
    return {
        "reviewed_pointer_channels": reviewed,
        "automated_matches": matches,
        "failures": failures,
    }


def render_html(
    payload: dict[str, Any],
    normalized_images: list[NormalizedBatchImage] | None = None,
) -> str:
    summary = payload.get("automated_summary") or summarize_automated(
        payload["records"]
    )
    cards: list[str] = []
    for index, record in enumerate(payload["records"]):
        rows: list[str] = []
        for channel in record["channels"]:
            automated = channel.get("automated")
            if automated is None:
                continue
            automated_text = _format_value(
                automated.get("value"),
                automated.get("candidates", []),
                automated.get("unit"),
            )
            status = str(automated.get("status") or "not_recognized")
            method = str(automated.get("method") or "—")
            raw_display = _format_raw_display(automated)
            note = str(automated.get("note_zh") or "—")
            rows.append(
                f'<tr class="status-{html.escape(status)}">'
                f"<td>{html.escape(channel['instance_id'])}</td>"
                f"<td>{html.escape(channel['channel_id'])}</td>"
                f"<td>{html.escape(automated_text)}</td>"
                f'<td><span class="status">{html.escape(_automated_status_label(status))}</span></td>'
                "</tr>"
                '<tr class="channel-detail"><td colspan="4">'
                f"<span><b>识别方法：</b>{html.escape(method)}</span>"
                f"<span><b>原始显示：</b>{html.escape(raw_display)}</span>"
                f"<span><b>说明：</b>{html.escape(note)}</span>"
                "</td></tr>"
            )
        if not rows:
            rows.append(
                '<tr class="status-not_recognized">'
                '<td colspan="4">程序未识别到可报告通道</td>'
                "</tr>"
            )
        batch_image = (
            normalized_images[index]
            if normalized_images is not None and index < len(normalized_images)
            else None
        )
        image_markup = (
            f'<img src="{annotated_preview_data_uri(batch_image.analysis_path, record.get("detections", []))}" '
            f'alt="{html.escape(record["image"])}">'
            if batch_image is not None and batch_image.analysis_path.is_file()
            else '<div class="image-missing">未提供可嵌入的原图</div>'
        )
        source_dimensions = record.get("source_dimensions") or []
        analysis_dimensions = record.get("analysis_dimensions") or []
        dimension_markup = (
            f'<p class="dimensions">原图 {source_dimensions[0]}×{source_dimensions[1]} · '
            f"分析图 {analysis_dimensions[0]}×{analysis_dimensions[1]} · "
            f"检测框 {len(record.get('detections', []))}</p>"
            if len(source_dimensions) == 2 and len(analysis_dimensions) == 2
            else ""
        )
        failure_reason = record.get("analysis_failure_reason")
        instrument_type = record.get("instrument_type_id") or (
            "未识别" if failure_reason else "通用指针仪表（metadata 未匹配）"
        )
        failure_markup = (
            f'<p class="failure-note">图片级失败：{html.escape(str(failure_reason))}</p>'
            if failure_reason
            else ""
        )
        cards.append(
            '<section class="report-card">'
            '<div class="image-panel">'
            f"{image_markup}"
            f'<p class="filename">{html.escape(record["image"])}</p>'
            f"{dimension_markup}"
            '<p class="preview-note">绿色框为本次程序检测到的仪表区域；预览按检测区域裁剪，输入文件未修改。</p>'
            "</div>"
            '<div class="data-panel">'
            f"<h2>图片 {index + 1}</h2>"
            f'<p class="instrument-type">仪表类型：<code>{html.escape(str(instrument_type))}</code></p>'
            f"{failure_markup}"
            '<div class="table-wrap"><table><thead><tr>'
            "<th>实例</th><th>通道</th><th>程序识别结果</th><th>状态</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            "</div>"
            "</section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>仪表自动识别报告</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;margin:0;background:#f4f6f4;color:#17221d}}
main{{max-width:1500px;margin:0 auto;padding:32px}}
h1{{margin:0 0 8px}} h2{{margin:0 0 8px;font-size:20px}}
.note{{color:#5f6f67;margin:6px 0}}
.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0 30px}}
.summary div{{padding:12px 16px;background:#fff;border:1px solid #d8ded9;border-radius:10px;box-shadow:0 1px 3px #17221d12}}
.summary .primary{{background:#e8f5ec;border-color:#90c69f;font-weight:650}}
.summary .failure{{background:#fff4e6;border-color:#e9bb77}}
.report-card{{display:grid;grid-template-columns:minmax(300px,.9fr) minmax(0,1.35fr);gap:24px;background:#fff;border:1px solid #d8ded9;border-radius:14px;padding:20px;margin:0 0 24px;box-shadow:0 2px 8px #17221d12}}
.image-panel img{{display:block;width:100%;aspect-ratio:3/2;object-fit:contain;background:#eef1ee;border-radius:8px}}
.filename{{font-size:12px;color:#65736b;overflow-wrap:anywhere;margin:10px 0 0}}
.dimensions,.preview-note{{font-size:12px;color:#65736b;margin:5px 0 0;line-height:1.45}}
.image-missing{{display:grid;place-items:center;min-height:220px;background:#eef1ee;color:#65736b;border-radius:8px}}
.instrument-type{{margin:0 0 16px;color:#405047}} code{{font-size:13px}}
.failure-note{{padding:9px 12px;background:#fff0d8;color:#80520c;border-radius:7px}}
.table-wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #d8ded9;padding:9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}
th{{background:#eef3ef}} .status{{display:inline-block;padding:2px 7px;border-radius:999px;font-size:12px}}
.channel-detail td{{padding:7px 9px 12px;background:#fafbfa;color:#506057;font-size:12px}}
.channel-detail span{{display:inline-block;margin-right:18px}}
.status-recognized .status{{background:#dff2e5;color:#17612d}}
.status-ambiguous .status{{background:#fff0d8;color:#80520c}}
.status-not_recognized .status{{background:#fde2df;color:#902d25}}
@media (max-width:900px){{main{{padding:18px}}.report-card{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>仪表自动识别报告</h1>
<p class="note">每张图片均经过 EXIF 方向校正和最长边 {NORMALIZED_MAX_EDGE}px 归一化；绿色框与读数来自本次程序运行。</p>
<div class="summary"><div>图片 {summary["images"]}</div>
<div>匹配到类型 metadata {summary["instrument_types_recognized"]}/{summary["images"]}</div>
<div>程序识别通道 {summary["channels"]}</div>
<div class="primary">识别成功 {summary["recognized"]}</div>
<div>候选结果 {summary["ambiguous"]}</div>
<div class="{"failure" if summary["not_recognized"] else ""}">未识别 {summary["not_recognized"]}</div>
<div class="{"failure" if summary["analysis_failures"] else ""}">图片级失败 {summary["analysis_failures"]}</div></div>
{"".join(cards)}</main></body></html>"""


def _automated_status_label(status: str) -> str:
    return {
        "recognized": "识别成功",
        "ambiguous": "存在候选值",
        "not_recognized": "未识别",
    }.get(status, status)


def _format_raw_display(automated: dict[str, Any]) -> str:
    raw_display = automated.get("raw_display")
    raw_ocr_text = automated.get("raw_ocr_text")
    if raw_display is None and raw_ocr_text is None:
        return "—"
    if raw_ocr_text is None or raw_ocr_text == raw_display:
        return str(raw_display)
    if raw_display is None:
        return f"OCR: {raw_ocr_text}"
    return f"显示: {raw_display} / OCR: {raw_ocr_text}"


def _format_value(value: Any, candidates: list[float], unit: str | None) -> str:
    if candidates:
        body = "/".join(f"{candidate:g}" for candidate in candidates)
    elif value is None:
        body = "未获得"
    else:
        body = str(value)
    return f"{body} {unit or ''}".strip()


if __name__ == "__main__":
    raise SystemExit(main())
