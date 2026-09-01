"""Analyze instrument images and emit automated HTML plus audit JSON."""

from __future__ import annotations

import argparse
import base64
import gc
import html
import json
import mimetypes
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.instrument_image import POINTER_DISPLAY_TYPES, MetadataAwareImageAnalyzer
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
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/user-instrument-batch.json",
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
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        parser.error(f"Input images do not exist: {missing}")

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
    records = [
        evaluate_image(
            image_path,
            analyzer,
            metadata_catalog,
            observation_catalog,
        )
        for image_path in args.images
    ]
    del analyzer
    del reader
    gc.collect()
    pointer_acceptance = summarize_pointer_acceptance(records, metadata_catalog)
    automated_summary = summarize_automated(records)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": args.device,
        "separation_rule": (
            "Automated results remain independent; user-confirmed values own the "
            "final reviewed value and never become model input."
        ),
        "summary": summarize(records),
        "automated_summary": automated_summary,
        "pointer_acceptance": pointer_acceptance,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path = args.output.with_suffix(".html")
    html_path.write_text(render_html(payload, args.images), encoding="utf-8")
    print(json.dumps(automated_summary, ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}")
    print(f"HTML: {html_path}")
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
    image_path: Path,
    analyzer: MetadataAwareImageAnalyzer,
    metadata_catalog: InstrumentMetadataCatalog,
    observation_catalog: InstrumentObservationCatalog,
) -> dict[str, Any]:
    analysis = analyzer.analyze(image_path)
    observation = observation_catalog.for_image(image_path)
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
        channel = (
            metadata_catalog.get(type_id).channel(key[1])
            if type_id is not None
            else None
        )
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
    return {
        "image": image_path.name,
        "image_sha256": analysis.image_sha256,
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
        "channels": channel_records,
    }


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


def render_html(payload: dict[str, Any], image_paths: list[Path] | None = None) -> str:
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
        image_path = (
            image_paths[index]
            if image_paths is not None and index < len(image_paths)
            else None
        )
        image_markup = (
            f'<img src="{_image_data_uri(image_path)}" '
            f'alt="{html.escape(record["image"])}">'
            if image_path is not None and image_path.is_file()
            else '<div class="image-missing">未提供可嵌入的原图</div>'
        )
        instrument_type = record.get("instrument_type_id") or "未识别"
        failure_reason = record.get("analysis_failure_reason")
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
.image-panel img{{display:block;width:100%;max-height:620px;object-fit:contain;background:#eef1ee;border-radius:8px}}
.filename{{font-size:12px;color:#65736b;overflow-wrap:anywhere;margin:10px 0 0}}
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
<p class="note">每张图片的结果均来自本次程序运行。</p>
<div class="summary"><div>图片 {summary["images"]}</div>
<div>识别到仪表类型 {summary["instrument_types_recognized"]}/{summary["images"]}</div>
<div>程序识别通道 {summary["channels"]}</div>
<div class="primary">识别成功 {summary["recognized"]}</div>
<div>候选结果 {summary["ambiguous"]}</div>
<div class="{"failure" if summary["not_recognized"] else ""}">未识别 {summary["not_recognized"]}</div>
<div class="{"failure" if summary["analysis_failures"] else ""}">图片级失败 {summary["analysis_failures"]}</div></div>
{"".join(cards)}</main></body></html>"""


def _image_data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
