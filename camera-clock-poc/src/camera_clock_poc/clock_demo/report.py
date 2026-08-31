"""Self-contained Chinese HTML report for a clock-camera experiment session."""

from __future__ import annotations

import base64
import statistics
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np

from camera_clock_poc.reusable.session import SessionRecord

from .reader import circular_seconds_error


def _summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        statistics.fmean(values),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 95)),
    )


def _data_uri(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _tracking_continuity(records: list[SessionRecord]) -> tuple[float, int, int]:
    errors: list[float] = []
    valid = [item for item in records if item["seconds"] is not None]
    for index, previous in enumerate(valid):
        previous_at = datetime.fromisoformat(previous["timestamp"])
        for current in valid[index + 1 :]:
            current_at = datetime.fromisoformat(current["timestamp"])
            elapsed = (current_at - previous_at).total_seconds()
            if elapsed < 1.8:
                continue
            if elapsed > 2.7:
                break
            assert previous["seconds"] is not None
            assert current["seconds"] is not None
            observed = (current["seconds"] - previous["seconds"] + 30.0) % 60.0 - 30.0
            errors.append(abs(observed - elapsed))
            break
    jumps = sum(error > 1.0 for error in errors)
    continuity = 0.0 if not errors else 1.0 - jumps / len(errors)
    return continuity, jumps, len(errors)


def write_clock_report(
    session_dir: Path,
    records: list[SessionRecord],
    source: str,
    sample_fps: float,
    privacy_mode: bool = False,
) -> Path:
    processing = [float(item["processing_ms"]) for item in records]
    valid = [item for item in records if item["seconds"] is not None]
    alarm = [item for item in records if item["alarm_state"] == "alarm"]
    detected = [item for item in records if bool(item["detected"])]
    privacy_applied_count = sum(bool(item.get("privacy_applied")) for item in records)
    mean_ms, p50_ms, p95_ms = _summary(processing) if processing else (0.0, 0.0, 0.0)
    continuity, jump_count, continuity_pairs = _tracking_continuity(records)
    continuity_text = f"{continuity:.1%}" if continuity_pairs else "样本不足"
    continuity_verdict = (
        "未判定" if not continuity_pairs else ("通过" if continuity >= 0.9 else "失败")
    )
    errors: list[float] = []
    tilts: list[float] = []
    for item in records:
        tilt_value = item.get("tilt_degrees")
        if tilt_value is not None:
            tilts.append(float(tilt_value))
    rectified_count = sum(bool(item.get("perspective_rectified")) for item in records)
    reference_records = sum(
        int(item.get("scale_reference_labels", 0)) > 0 for item in records
    )
    max_reference_labels = max(
        (int(item.get("scale_reference_labels", 0)) for item in records),
        default=0,
    )
    mean_tilt = statistics.fmean(tilts) if tilts else 0.0
    p95_tilt = (
        float(np.percentile(np.asarray(tilts, dtype=np.float64), 95)) if tilts else 0.0
    )
    for item in valid:
        seconds = item["seconds"]
        expected = item["expected_seconds"]
        if seconds is not None and expected is not None:
            errors.append(circular_seconds_error(seconds, expected))
    error_text = (
        f"{float(np.percentile(np.asarray(errors), 95)):.2f} 秒"
        if errors
        else "未启用系统时间对照"
    )

    cards: list[str] = []
    for item in records:
        screenshot = item["screenshot"]
        if screenshot is None:
            continue
        image_path = session_dir / str(screenshot)
        raw_screenshot = item.get("raw_screenshot")
        raw_link = (
            f'<a href="{escape(str(raw_screenshot))}">查看未标注帧</a>'
            if raw_screenshot
            else "旧会话未保存未标注帧"
        )
        tilt_value = item.get("tilt_degrees")
        tilt_text = f"{float(tilt_value):.1f}°" if tilt_value is not None else "无"
        cards.append(
            f"""
            <article>
              <img src="{_data_uri(image_path)}" alt="实验截图">
              <div><strong>{escape(str(item["timestamp"]))}</strong><br>
              检测：{"成功" if item["detected"] else "失败"}　秒数：{f"{float(item['seconds']):.2f}" if item["seconds"] is not None else "无"}　报警：{escape(str(item["alarm_state"]))}<br>
              表盘倾斜：{tilt_text}　正视校正：{"是" if item.get("perspective_rectified") else "否"}<br>
              数字刻度参照：{int(item.get("scale_reference_labels", 0))}个<br>
              本帧背景虚化：{"是" if item.get("privacy_applied") else "否"}<br>
              耗时：{float(item["processing_ms"]):.2f} 毫秒　原因：{escape(str(item["failure_reason"] or "无"))}<br>{raw_link}</div>
            </article>
            """
        )
    report_path = session_dir / "report.html"
    report_path.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>闹钟摄像头PoC实验报告</title>
<style>:root{{--ink:#16211d;--muted:#64736c;--paper:#f4f1e8;--panel:#fffef9;--line:#d8d4c8;--green:#13794b;--red:#b23a32}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1320px;margin:auto;padding:38px 24px 70px}}h1{{font-size:clamp(36px,6vw,70px);line-height:1;margin:8px 0 18px;letter-spacing:-.05em}}.lead{{color:var(--muted);line-height:1.7}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}}.stat,article{{background:var(--panel);border:1px solid var(--line)}}.stat{{padding:18px}}.stat strong{{display:block;font-size:28px}}.stat span{{color:var(--muted)}}.gallery{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}article{{overflow:hidden}}article img{{display:block;width:100%;height:320px;object-fit:contain;background:#18201c}}article div{{padding:12px;font-size:13px;line-height:1.6}}@media(max-width:760px){{.stats,.gallery{{grid-template-columns:1fr}}main{{padding:24px 12px}}}}</style></head>
<body><main><p class="lead">摄像头闹钟演示 · 完整画面输入</p><h1>闹钟摄像头<br>PoC实验报告</h1><p class="lead">来源：{escape(source)}　抽样频率：{sample_fps:g} 帧/秒。没有闹钟或无法识别时，报警状态必须为未知，不能视为正常。隐私背景虚化：{"条件开启；只在检测成功后虚化背景，漏检帧保留清晰原图用于诊断" if privacy_mode else "未开启"}。</p>
<section class="stats"><div class="stat"><strong>{len(records)}</strong><span>采样次数</span></div><div class="stat"><strong>{len(detected)}/{len(records)}</strong><span>闹钟检测成功</span></div><div class="stat"><strong>{len(valid)}/{len(records)}</strong><span>有效秒数</span></div><div class="stat"><strong>{len(alarm)}</strong><span>报警采样</span></div></section>
<section class="stats"><div class="stat"><strong>{privacy_applied_count}/{len(records)}</strong><span>检测成功后背景虚化</span></div><div class="stat"><strong>{len(records) - privacy_applied_count}</strong><span>保留清晰原图的帧</span></div><div class="stat"><strong>{"诊断优先" if privacy_mode else "未开启"}</strong><span>隐私模式</span></div><div class="stat"><strong>{"是" if privacy_mode else "否"}</strong><span>漏检帧可视</span></div></section>
<section class="stats"><div class="stat"><strong>{mean_ms:.1f}</strong><span>平均耗时（毫秒）</span></div><div class="stat"><strong>{p50_ms:.1f}</strong><span>p50（毫秒）</span></div><div class="stat"><strong>{p95_ms:.1f}</strong><span>p95（毫秒）</span></div><div class="stat"><strong>{error_text}</strong><span>秒数误差p95</span></div></section>
<section class="stats"><div class="stat"><strong>{continuity_text}</strong><span>两秒窗口进度一致性（目标≥90%）</span></div><div class="stat"><strong>{jump_count}</strong><span>冻结/跳轨窗口（误差&gt;1秒）</span></div><div class="stat"><strong>{continuity_pairs}</strong><span>参与进度比较的时间窗口</span></div><div class="stat"><strong>{continuity_verdict}</strong><span>连续跟踪判定</span></div></section>
<section class="stats"><div class="stat"><strong>{rectified_count}/{len(records)}</strong><span>完整透视校正次数</span></div><div class="stat"><strong>{mean_tilt:.1f}°</strong><span>平均估计倾斜角</span></div><div class="stat"><strong>{p95_tilt:.1f}°</strong><span>倾斜角p95</span></div><div class="stat"><strong>{reference_records}/{len(records)}，最多{max_reference_labels}个</strong><span>数字刻度参照</span></div></section>
<section class="gallery">{"".join(cards)}</section></main></body></html>""",
        encoding="utf-8",
    )
    return report_path
