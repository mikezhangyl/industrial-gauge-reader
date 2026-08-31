"""Clock-only 50–59 second demonstration alarm."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AlarmState(StrEnum):
    NORMAL = "normal"
    ALARM = "alarm"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AlarmDecision:
    state: AlarmState
    changed: bool
    reason: str


class ClockDemoAlarm:
    def __init__(
        self, start_second: float = 50.0, consecutive_samples: int = 2
    ) -> None:
        if not 0.0 <= start_second < 60.0:
            raise ValueError("start_second 必须位于 [0, 60)")
        if consecutive_samples < 1:
            raise ValueError("consecutive_samples 必须大于0")
        self.start_second = start_second
        self.consecutive_samples = consecutive_samples
        self.state = AlarmState.UNKNOWN
        self._candidate: AlarmState | None = None
        self._candidate_count = 0

    def update(self, seconds: float | None) -> AlarmDecision:
        previous = self.state
        if seconds is None:
            self.state = AlarmState.UNKNOWN
            self._candidate = None
            self._candidate_count = 0
            return AlarmDecision(
                self.state, previous != self.state, "没有可靠秒数，报警状态未知"
            )

        target = (
            AlarmState.ALARM
            if self.start_second <= seconds < 60.0
            else AlarmState.NORMAL
        )
        if target == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = target
            self._candidate_count = 1
        if self._candidate_count >= self.consecutive_samples:
            self.state = target
        reason = (
            f"秒数 {seconds:.2f} 进入 {self.start_second:g}～59 秒演示报警区"
            if self.state == AlarmState.ALARM
            else f"秒数 {seconds:.2f} 不在演示报警区"
        )
        return AlarmDecision(self.state, previous != self.state, reason)
