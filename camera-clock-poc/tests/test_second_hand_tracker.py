from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from camera_clock_poc.clock_demo.reader import HandCandidate, SecondHandTracker


def hand(angle: float, score: float = 1.0) -> HandCandidate:
    return HandCandidate(
        tip=(100.0, 20.0),
        angle_degrees=angle % 360.0,
        length_ratio=0.85,
        line_score=score,
    )


class SecondHandTrackerTests(unittest.TestCase):
    def test_selects_six_degree_per_second_track_over_static_hands(self) -> None:
        tracker = SecondHandTracker(min_evidence_seconds=1.5)
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)
        selected = None
        for index in range(6):
            selected = tracker.update(
                [hand(60.0), hand(130.0), hand(300.0 + index * 3.0)],
                started + timedelta(seconds=index * 0.5),
            )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertAlmostEqual(selected.candidate.angle_degrees, 315.0, delta=1.0)
        self.assertAlmostEqual(selected.velocity_degrees_per_second, 6.0, delta=0.5)

    def test_returns_none_when_only_static_lines_exist(self) -> None:
        tracker = SecondHandTracker(min_evidence_seconds=1.0)
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)
        selected = None
        for index in range(5):
            selected = tracker.update(
                [hand(60.0), hand(130.0)],
                started + timedelta(seconds=index * 0.5),
            )
        self.assertIsNone(selected)

    def test_tracks_across_twelve_oclock_wrap(self) -> None:
        tracker = SecondHandTracker(min_evidence_seconds=1.0)
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)
        selected = None
        for index, angle in enumerate((354.0, 357.0, 0.0, 3.0, 6.0)):
            selected = tracker.update(
                [hand(angle)], started + timedelta(seconds=index * 0.5)
            )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertAlmostEqual(selected.velocity_degrees_per_second, 6.0, delta=0.5)

    def test_locked_hand_is_not_replaced_by_a_distant_competing_track(self) -> None:
        tracker = SecondHandTracker(min_evidence_seconds=1.0)
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)
        selected = None
        sequence = (
            [hand(0.0)],
            [hand(3.0)],
            [hand(6.0)],
            [hand(8.75), hand(180.0)],
            [hand(11.5), hand(183.0)],
            [hand(14.25), hand(186.0)],
        )
        for index, candidates in enumerate(sequence):
            selected = tracker.update(
                candidates, started + timedelta(seconds=index * 0.5)
            )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertAlmostEqual(selected.candidate.angle_degrees, 14.25, delta=1.0)

    def test_static_nearby_line_cannot_extend_prediction_hold_forever(self) -> None:
        tracker = SecondHandTracker(min_evidence_seconds=1.0, hold_seconds=1.25)
        started = datetime(2026, 8, 20, tzinfo=timezone.utc)
        sequence = (
            [hand(0.0)],
            [hand(3.0)],
            [hand(6.0)],
            [hand(6.0)],
            [hand(6.0)],
            [hand(6.0)],
            [hand(6.0)],
            [hand(6.0)],
        )
        selections = []
        for index, candidates in enumerate(sequence):
            selections.append(
                tracker.update(candidates, started + timedelta(seconds=index * 0.5))
            )

        self.assertIsNone(selections[5])
        self.assertIsNone(selections[-1])


if __name__ == "__main__":
    unittest.main()
