"""Evaluate a recorded clock session against manually labelled presence intervals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict


class Record(TypedDict):
    timestamp: str
    detected: bool
    seconds: float | None


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def continuity_rate(records: list[Record], tolerance_seconds: float = 1.0) -> float:
    errors: list[float] = []
    valid = [record for record in records if record["seconds"] is not None]
    for index, previous in enumerate(valid):
        previous_at = parse_timestamp(previous["timestamp"])
        for current in valid[index + 1 :]:
            current_at = parse_timestamp(current["timestamp"])
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
    return (
        0.0 if not errors else sum(e <= tolerance_seconds for e in errors) / len(errors)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--clock-start", required=True, type=parse_timestamp)
    parser.add_argument("--clock-end", required=True, type=parse_timestamp)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.02)
    parser.add_argument("--min-valid-rate", type=float, default=0.90)
    parser.add_argument("--min-continuity-rate", type=float, default=0.90)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in (args.session / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    clock_present: list[Record] = []
    clock_absent: list[Record] = []
    for record in records:
        captured_at = parse_timestamp(record["timestamp"])
        target = (
            clock_present
            if args.clock_start <= captured_at <= args.clock_end
            else clock_absent
        )
        target.append(record)

    false_positive_rate = (
        0.0
        if not clock_absent
        else sum(record["detected"] for record in clock_absent) / len(clock_absent)
    )
    valid_rate = (
        0.0
        if not clock_present
        else sum(record["seconds"] is not None for record in clock_present)
        / len(clock_present)
    )
    stable_rate = continuity_rate(clock_present)
    print(
        f"无闹钟误检率：{false_positive_rate:.1%}（目标 <= {args.max_false_positive_rate:.1%}）"
    )
    print(f"有闹钟有效读数率：{valid_rate:.1%}（目标 >= {args.min_valid_rate:.1%}）")
    print(
        f"两秒窗口进度一致性：{stable_rate:.1%}（目标 >= {args.min_continuity_rate:.1%}）"
    )
    passed = (
        false_positive_rate <= args.max_false_positive_rate
        and valid_rate >= args.min_valid_rate
        and stable_rate >= args.min_continuity_rate
    )
    print(f"会话判定：{'通过' if passed else '失败'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
