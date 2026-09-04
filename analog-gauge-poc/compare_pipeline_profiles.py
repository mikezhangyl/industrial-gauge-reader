"""Run and compare the versioned 448 and 640 gauge-processing profiles."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import signal
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from src.batch_io import discover_input_images
from src.pipeline_profile import GAUGE_PIPELINE_PROFILE_NAMES
from src.profile_comparison import compare_profile_payloads

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_NAMES = ("448", "640")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a full automated comparison of two gauge profiles"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input image directories")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--profiles",
        nargs=2,
        choices=GAUGE_PIPELINE_PROFILE_NAMES,
        default=DEFAULT_PROFILE_NAMES,
        metavar=("BASELINE", "CANDIDATE"),
        help="Two named profiles to compare (default: 448 640)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "pipeline-profile-comparison",
        help=(
            "Run-history root. Each invocation writes to a new timestamped "
            "subdirectory."
        ),
    )
    parser.add_argument(
        "--export-processing-stages",
        action="store_true",
        help="Pass lossless processing-stage export through to each batch run",
    )
    args = parser.parse_args()
    input_directories = [path.resolve() for path in args.inputs]
    for path in input_directories:
        if not path.is_dir():
            parser.error(f"Input is not a directory: {path}")
    output_dir = create_timestamped_run_directory(args.output_dir.resolve())
    profile_names = tuple(args.profiles)
    batches = [
        _run_batch(
            index,
            input_directory,
            output_dir,
            args.device,
            profile_names,
            export_processing_stages=args.export_processing_stages,
        )
        for index, input_directory in enumerate(input_directories, start=1)
    ]
    aggregate: Counter[str] = Counter()
    for batch in batches:
        summary = batch["comparison"]["summary"]
        for key, value in summary.items():
            aggregate[key] += int(value)
    payload = {
        "run_id": output_dir.name,
        "output_directory": str(output_dir),
        "profiles": list(profile_names),
        "inputs": [str(path) for path in input_directories],
        "summary": dict(aggregate),
        "run_warnings": [
            warning for batch in batches for warning in batch["run_warnings"]
        ],
        "batches": batches,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "profile-comparison.json"
    html_path = output_dir / "profile-comparison.html"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Comparison JSON: {json_path}")
    print(f"Comparison HTML: {html_path}")
    return 0


def _run_batch(
    index: int,
    input_directory: Path,
    output_dir: Path,
    device: str,
    profile_names: tuple[str, str],
    *,
    export_processing_stages: bool = False,
) -> dict[str, Any]:
    batch_key = _batch_key(index, input_directory.name)
    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, dict[str, str]] = {}
    run_warnings: list[dict[str, Any]] = []
    expected_images = [path.name for path in discover_input_images([input_directory])[0]]
    for profile_name in profile_names:
        profile_dir = output_dir / profile_name / batch_key
        json_path = profile_dir / "instrument-report.json"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "batch_instrument_report.py"),
            "--device",
            device,
            "--pipeline-profile",
            profile_name,
            "--output",
            str(json_path),
            str(input_directory),
        ]
        if export_processing_stages:
            command.insert(-1, "--export-processing-stages")
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode not in (0, -signal.SIGABRT):
            raise subprocess.CalledProcessError(completed.returncode, command)
        reports[profile_name] = load_completed_report(
            json_path,
            json_path.with_suffix(".html"),
            expected_profile=profile_name,
            expected_images=expected_images,
        )
        if completed.returncode == -signal.SIGABRT:
            warning = {
                "batch": input_directory.name,
                "profile": profile_name,
                "kind": "post_export_sigabrt",
                "message": (
                    "The batch process aborted during native-library shutdown after "
                    "fresh JSON and HTML artifacts were fully validated."
                ),
            }
            run_warnings.append(warning)
            print(f"Warning: {warning['message']}", file=sys.stderr)
        report_paths[profile_name] = {
            "json": str(json_path.relative_to(output_dir)),
            "html": str(json_path.with_suffix(".html").relative_to(output_dir)),
        }
    comparison = compare_profile_payloads(
        reports[profile_names[0]], reports[profile_names[1]]
    )
    image_paths = {path.name: path for path in input_directory.iterdir() if path.is_file()}
    for record in comparison["records"]:
        image_path = image_paths.get(str(record["image"]))
        record["thumbnail"] = _thumbnail_data_uri(image_path) if image_path else None
        record["profiles"] = list(profile_names)
    return {
        "batch_name": input_directory.name,
        "batch_key": batch_key,
        "input_directory": str(input_directory),
        "profiles": list(profile_names),
        "reports": report_paths,
        "run_warnings": run_warnings,
        "comparison": comparison,
    }


def load_completed_report(
    json_path: Path,
    html_path: Path,
    *,
    expected_profile: str,
    expected_images: list[str],
) -> dict[str, Any]:
    """Load report artifacts only after validating their content contract."""

    if not json_path.is_file() or not html_path.is_file():
        raise ValueError("Batch report artifacts are incomplete")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    profile_name = (payload.get("pipeline_profile") or {}).get("name")
    if profile_name != expected_profile:
        raise ValueError(
            f"Batch report profile mismatch: expected {expected_profile}, got {profile_name}"
        )
    image_order = (payload.get("input_contract") or {}).get("image_order")
    record_images = [record.get("image") for record in payload.get("records", [])]
    if image_order != expected_images or record_images != expected_images:
        raise ValueError("Batch report image order is incomplete or mismatched")
    if "</html>" not in html_path.read_text(encoding="utf-8").casefold():
        raise ValueError("Batch HTML report is incomplete")
    return payload


def create_timestamped_run_directory(
    output_root: Path, *, now: datetime | None = None
) -> Path:
    """Create an isolated output directory for one comparison invocation."""

    instant = now or datetime.now().astimezone()
    run_directory = output_root / instant.strftime("%Y%m%dT%H%M%S%f%z")
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _batch_key(index: int, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "batch"
    return f"{index:02d}-{safe}"


def _thumbnail_data_uri(path: Path) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((480, 320), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (480, 320), "#eef1ee")
    offset = ((480 - image.width) // 2, (320 - image.height) // 2)
    canvas.paste(image, offset)
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_html(payload: dict[str, Any]) -> str:
    batches = "".join(_render_batch(batch) for batch in payload["batches"])
    summary = payload["summary"]
    baseline_name, candidate_name = payload["profiles"]
    summary_items = "".join(
        f"<div><span>{html.escape(key)}</span><strong>{value}</strong></div>"
        for key, value in summary.items()
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(baseline_name)} / {html.escape(candidate_name)} 仪表管线比较</title>
<style>
body{{margin:0;background:#f2f4f2;color:#17211d;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:0 0 8px}}.muted{{color:#65716b}}
.summary{{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}}.summary div{{background:white;border:1px solid #d9dfdb;border-radius:8px;padding:10px 14px}}
.summary span{{margin-right:10px;color:#65716b}}section{{background:white;border:1px solid #d9dfdb;border-radius:12px;padding:18px;margin:18px 0}}
.links a{{margin-right:14px}}.record{{display:grid;grid-template-columns:360px 1fr;gap:18px;border-top:1px solid #e4e8e5;padding:18px 0}}
.record img{{width:100%;height:240px;object-fit:contain;background:#eef1ee;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #dce2de;padding:7px;text-align:left;vertical-align:top}}th{{background:#f5f7f5}}
.unchanged{{color:#2b7450}}.coverage_gain{{color:#1769aa}}.coverage_loss,.value_changed,.status_changed{{color:#b33a2f;font-weight:600}}.details_changed{{color:#6c5a24}}
@media(max-width:900px){{.record{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>{html.escape(baseline_name)} / {html.escape(candidate_name)} 仪表管线比较</h1><p class="muted">仅比较两次程序自动输出；不把任一 profile 当作人工真值。</p>
<div class="summary">{summary_items}</div>{batches}</main></body></html>"""


def _render_batch(batch: dict[str, Any]) -> str:
    links = []
    for profile_name in batch["profiles"]:
        report = batch["reports"][profile_name]
        links.append(
            f'<a href="{html.escape(report["html"])}">{profile_name} 完整报告</a>'
        )
    records = "".join(_render_record(record) for record in batch["comparison"]["records"])
    return (
        f'<section><h2>{html.escape(batch["batch_name"])}</h2>'
        f'<div class="links">{"".join(links)}</div>{records}</section>'
    )


def _render_record(record: dict[str, Any]) -> str:
    rows = "".join(_render_channel(channel) for channel in record["channels"])
    thumbnail = record.get("thumbnail")
    image = (
        f'<img src="{thumbnail}" alt="{html.escape(record["image"])}">'
        if thumbnail
        else ""
    )
    baseline_name, candidate_name = record["profiles"]
    return f"""<div class="record"><div>{image}<p>{html.escape(record["image"])}</p></div>
<table><thead><tr><th>实例 / 通道</th><th>{html.escape(baseline_name)}</th><th>{html.escape(candidate_name)}</th><th>变化</th><th>角度差</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""


def _render_channel(channel: dict[str, Any]) -> str:
    change = str(channel["change"])
    angle_delta = channel.get("angle_delta_degrees")
    return (
        "<tr>"
        f'<td>{html.escape(channel["instance_id"])}<br>{html.escape(channel["channel_id"])}</td>'
        f'<td>{_format_result(channel.get("baseline"))}</td>'
        f'<td>{_format_result(channel.get("candidate"))}</td>'
        f'<td class="{html.escape(change)}">{html.escape(change)}</td>'
        f'<td>{"-" if angle_delta is None else f"{angle_delta:.3f}°"}</td>'
        "</tr>"
    )


def _format_result(result: dict[str, Any] | None) -> str:
    if result is None:
        return "-"
    value = result.get("value")
    candidates = result.get("candidates") or []
    display = value if value is not None else "/".join(str(item) for item in candidates)
    return html.escape(f"{display if display != '' else '-'} ({result.get('status')})")


if __name__ == "__main__":
    raise SystemExit(main())
