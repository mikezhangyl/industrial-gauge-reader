"""Chinese visualization for the clock-only camera demonstration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera_clock_poc.reusable.types import Observation

from .alarm import AlarmDecision, AlarmState


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("没有找到中文字体")


def alarm_label(state: AlarmState) -> str:
    return {
        AlarmState.NORMAL: "正常",
        AlarmState.ALARM: "报警",
        AlarmState.UNKNOWN: "未知",
    }[state]


def _panel_origin(
    frame_size: tuple[int, int],
    panel_size: tuple[int, int],
    instrument_bbox: tuple[int, int, int, int] | None,
    margin: int = 10,
) -> tuple[int, int]:
    """Choose the corner with the least overlap and greatest distance from the dial."""

    width, height = frame_size
    panel_width, panel_height = panel_size
    candidates = (
        (margin, margin),
        (max(margin, width - panel_width - margin), margin),
        (margin, max(margin, height - panel_height - margin)),
        (
            max(margin, width - panel_width - margin),
            max(margin, height - panel_height - margin),
        ),
    )
    if instrument_bbox is None:
        return candidates[0]

    box_x1, box_y1, box_x2, box_y2 = instrument_bbox
    box_center = ((box_x1 + box_x2) * 0.5, (box_y1 + box_y2) * 0.5)

    def score(origin: tuple[int, int]) -> tuple[float, float]:
        x1, y1 = origin
        x2, y2 = x1 + panel_width, y1 + panel_height
        overlap_width = max(0, min(x2, box_x2) - max(x1, box_x1))
        overlap_height = max(0, min(y2, box_y2) - max(y1, box_y1))
        overlap = float(overlap_width * overlap_height)
        panel_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        distance_squared = (panel_center[0] - box_center[0]) ** 2 + (
            panel_center[1] - box_center[1]
        ) ** 2
        return (overlap, -distance_squared)

    return min(candidates, key=score)


def render_overlay(
    frame: np.ndarray,
    observation: Observation,
    alarm: AlarmDecision,
    expected_seconds: float | None = None,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    thickness = max(2, round(min(height, width) / 300))
    if observation.bbox is not None:
        x1, y1, x2, y2 = observation.bbox
        color = (0, 0, 255) if alarm.state == AlarmState.ALARM else (0, 210, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
    if observation.center is not None:
        center = tuple(round(value) for value in observation.center)
        cv2.circle(output, center, thickness * 3, (255, 80, 0), -1)
        if observation.pointer_tip is not None:
            tip = tuple(round(value) for value in observation.pointer_tip)
            cv2.arrowedLine(
                output,
                center,
                tip,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
                tipLength=0.16,
            )

    seconds = "无" if observation.value is None else f"{observation.value:.2f} 秒"
    angle = (
        "无"
        if observation.angle_degrees is None
        else f"{observation.angle_degrees:.2f}°"
    )
    confidence = (
        "无" if observation.confidence is None else f"{observation.confidence:.2f}"
    )
    tilt = (
        "无" if observation.tilt_degrees is None else f"{observation.tilt_degrees:.1f}°"
    )
    correction = "已校正" if observation.perspective_rectified else "未校正"
    reference = (
        f"{observation.scale_reference_labels}个数字"
        if observation.scale_reference_labels
        else "仅刻度"
    )
    tracking = "短时保持" if "短时预测保持" in observation.method else "实时"
    lines = [
        f"检测 {'成功' if observation.detected else '未检测'}  |  读数 {seconds}  |  报警 {alarm_label(alarm.state)}",
        f"角度 {angle}  |  置信 {confidence}  |  耗时 {observation.processing_ms:.1f}ms",
        f"倾斜 {tilt}/{correction}  |  参照 {reference}  |  跟踪 {tracking}",
    ]
    if expected_seconds is not None:
        lines.append(f"系统参考秒数：{expected_seconds:.2f} 秒")
    if observation.failure_reason:
        lines.append(f"失败原因：{observation.failure_reason}")

    canvas = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(max(14, min(22, round(width / 70))))
    text = "\n".join(lines)
    line_spacing = 2
    text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing)
    padding = max(6, font.size // 3)
    panel_width = min(width - 20, round(text_box[2] + 2 * padding))
    panel_height = min(height - 20, round(text_box[3] + 2 * padding))
    origin_x, origin_y = _panel_origin(
        (width, height),
        (panel_width, panel_height),
        observation.bbox,
    )
    panel = (
        origin_x,
        origin_y,
        origin_x + panel_width,
        origin_y + panel_height,
    )
    panel_color = (
        (150, 20, 15, 205) if alarm.state == AlarmState.ALARM else (0, 0, 0, 170)
    )
    draw.rounded_rectangle(panel, radius=8, fill=panel_color)
    draw.multiline_text(
        (origin_x + padding, origin_y + padding),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=line_spacing,
    )
    return cv2.cvtColor(
        np.asarray(Image.alpha_composite(canvas, overlay).convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )
