"""Pure comparison of two automated gauge-report payloads."""

from __future__ import annotations

from collections import Counter
from typing import Any


def compare_profile_payloads(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare profile outputs without treating either profile as ground truth."""

    candidate_records = {
        str(record["image"]): record for record in candidate.get("records", [])
    }
    compared_records: list[dict[str, Any]] = []
    change_counts: Counter[str] = Counter()
    for baseline_record in baseline.get("records", []):
        image_name = str(baseline_record["image"])
        candidate_record = candidate_records.pop(image_name, None)
        if candidate_record is None:
            raise ValueError(f"Candidate report is missing image: {image_name}")
        if baseline_record.get("image_sha256") != candidate_record.get("image_sha256"):
            raise ValueError(f"Cannot compare {image_name}: image SHA-256 differs")
        record = _compare_record(baseline_record, candidate_record)
        compared_records.append(record)
        change_counts.update(channel["change"] for channel in record["channels"])
    if candidate_records:
        extras = ", ".join(sorted(candidate_records))
        raise ValueError(f"Candidate report contains extra images: {extras}")

    channel_count = sum(len(record["channels"]) for record in compared_records)
    summary = {
        "images": len(compared_records),
        "channels": channel_count,
        "unchanged": change_counts["unchanged"],
        "value_changed": change_counts["value_changed"],
        "candidates_changed": change_counts["candidates_changed"],
        "status_changed": change_counts["status_changed"],
        "details_changed": change_counts["details_changed"],
        "coverage_gain": change_counts["coverage_gain"],
        "coverage_loss": change_counts["coverage_loss"],
        "baseline_only": change_counts["baseline_only"],
        "candidate_only": change_counts["candidate_only"],
    }
    return {
        "baseline_profile": baseline.get("pipeline_profile"),
        "candidate_profile": candidate.get("pipeline_profile"),
        "summary": summary,
        "records": compared_records,
    }


def _compare_record(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_channels = _channel_map(baseline)
    candidate_channels = _channel_map(candidate)
    channel_keys = list(baseline_channels)
    channel_keys.extend(key for key in candidate_channels if key not in baseline_channels)
    baseline_angles = _angle_map(baseline)
    candidate_angles = _angle_map(candidate)
    channels = []
    for instance_id, channel_id in channel_keys:
        baseline_channel = baseline_channels.get((instance_id, channel_id))
        candidate_channel = candidate_channels.get((instance_id, channel_id))
        change = _classify_change(baseline_channel, candidate_channel)
        channels.append(
            {
                "instance_id": instance_id,
                "channel_id": channel_id,
                "baseline": _automated_result(baseline_channel),
                "candidate": _automated_result(candidate_channel),
                "change": change,
                "angle_delta_degrees": _angle_delta(
                    baseline_angles.get(instance_id), candidate_angles.get(instance_id)
                ),
            }
        )
    return {
        "image": baseline["image"],
        "image_sha256": baseline.get("image_sha256"),
        "channels": channels,
    }


def _channel_map(record: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(channel["instance_id"]), str(channel["channel_id"])): channel
        for channel in record.get("channels", [])
    }


def _angle_map(record: dict[str, Any]) -> dict[str, float]:
    return {
        str(detection["instance_id"]): float(detection["angle_degrees"])
        for detection in record.get("detections", [])
        if detection.get("angle_degrees") is not None
    }


def _automated_result(channel: dict[str, Any] | None) -> dict[str, Any] | None:
    if channel is None:
        return None
    automated = channel.get("automated") or {}
    return {
        "status": automated.get("status"),
        "value": automated.get("value"),
        "candidates": automated.get("candidates") or [],
        "method": automated.get("method"),
        "note_zh": automated.get("note_zh"),
    }


def _classify_change(
    baseline: dict[str, Any] | None, candidate: dict[str, Any] | None
) -> str:
    if baseline is None:
        return "candidate_only"
    if candidate is None:
        return "baseline_only"
    first = _automated_result(baseline) or {}
    second = _automated_result(candidate) or {}
    if first == second:
        return "unchanged"
    first_status = first.get("status")
    second_status = second.get("status")
    available = {"recognized", "ambiguous"}
    if first_status not in available and second_status in available:
        return "coverage_gain"
    if first_status in available and second_status not in available:
        return "coverage_loss"
    if first_status != second_status:
        return "status_changed"
    if first.get("value") != second.get("value"):
        return "value_changed"
    if first.get("candidates") != second.get("candidates"):
        return "candidates_changed"
    return "details_changed"


def _angle_delta(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return round(abs((second - first + 180.0) % 360.0 - 180.0), 6)
