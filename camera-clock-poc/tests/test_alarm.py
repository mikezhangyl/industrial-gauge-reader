from __future__ import annotations

import unittest

from camera_clock_poc.clock_demo.alarm import AlarmState, ClockDemoAlarm


class ClockDemoAlarmTests(unittest.TestCase):
    def test_requires_two_consecutive_samples_to_enter_and_leave_alarm(self) -> None:
        alarm = ClockDemoAlarm(start_second=50.0, consecutive_samples=2)
        self.assertEqual(alarm.update(49.0).state, AlarmState.UNKNOWN)
        self.assertEqual(alarm.update(49.5).state, AlarmState.NORMAL)
        self.assertEqual(alarm.update(50.0).state, AlarmState.NORMAL)
        self.assertEqual(alarm.update(51.0).state, AlarmState.ALARM)
        self.assertEqual(alarm.update(0.0).state, AlarmState.ALARM)
        self.assertEqual(alarm.update(1.0).state, AlarmState.NORMAL)

    def test_missing_reading_is_unknown_not_normal(self) -> None:
        alarm = ClockDemoAlarm(consecutive_samples=1)
        self.assertEqual(alarm.update(55.0).state, AlarmState.ALARM)
        self.assertEqual(alarm.update(None).state, AlarmState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
