import random
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from failure import _failure_at, busy_failure, merge_plans, plan_state, replan_after_failure, sample_failure
from test_run import ScheduleResult, TaskSpec


@dataclass(frozen=True)
class ReleasedRobot:
    robot_id: int
    start_vertex: int
    battery: int
    release_time: int = 0
    entry_vertex: int | None = None


class FailureReleaseTests(unittest.TestCase):
    def delayed_plan(self):
        robot = ReleasedRobot(1, 1, 5, release_time=3, entry_vertex=0)
        result = ScheduleResult(True, 4, {
            1: [(1, None, None, 5)], 2: [(1, None, None, 5)],
            3: [(1, 0, None, 5)], 4: [(1, 1, 7, 4)],
        }, [], "FEASIBLE")
        return [robot], [TaskSpec(7, 0, 1)], result

    def test_state_uses_entry_at_zero_and_absence_before_release(self):
        robots, tasks, result = self.delayed_plan()
        self.assertIsNone(plan_state(robots, tasks, result, 0)[0][1])
        self.assertIsNone(plan_state(robots, tasks, result, 2)[0][1])
        self.assertEqual(plan_state(robots, tasks, result, 3)[0][1], 0)
        immediate = [ReleasedRobot(1, 3, 5, entry_vertex=0)]
        self.assertEqual(plan_state(immediate, tasks, result, 0)[0][1], 0)

    def test_failure_cannot_happen_before_robot_appears(self):
        robots, tasks, result = self.delayed_plan()
        with self.assertRaises(ValueError):
            _failure_at(robots, tasks, result, 1, 2)
        sampled = sample_failure(robots, tasks, result, 1, random.Random(0), only_moving=False)
        self.assertEqual((sampled.robot_id, sampled.tick, sampled.vertex), (1, 3, 0))
        self.assertIsNone(busy_failure(robots, tasks, result))

    def test_replan_preserves_future_entry_and_relative_release(self):
        robots = [ReleasedRobot(1, 0, 6), ReleasedRobot(2, 3, 5, entry_vertex=1),
                  ReleasedRobot(3, 3, 6, release_time=4, entry_vertex=0)]
        tasks = [TaskSpec(7, 3, 4)]
        result = ScheduleResult(True, 1, {1: [(1, 0, None, 6), (2, 2, None, 4),
                                             (3, None, None, 6)]}, [], "FEASIBLE")
        failure = _failure_at(robots, tasks, result, 1, 1)

        class Solver:
            def solve(self, alive, left, **kwargs):
                self.alive = alive
                self.kwargs = kwargs
                return ScheduleResult(True, 0, {}, [], "FEASIBLE")

        solver = Solver()
        with patch("failure.RobotSpec", ReleasedRobot):
            _, _, alive = replan_after_failure(solver, robots, tasks, result, failure, 10)
        self.assertEqual(alive, [ReleasedRobot(2, 2, 4), ReleasedRobot(3, 3, 6, 3, 0)])
        self.assertNotIn("blocked_cargo", solver.kwargs)

    def recovery_fixture(self):
        robots = [ReleasedRobot(1, 0, 5), ReleasedRobot(2, 3, 5)]
        tasks = [TaskSpec(7, 0, 2)]
        result = ScheduleResult(True, 2, {
            1: [(1, 1, 7, 4), (2, 3, None, 5)],
            2: [(1, 2, 7, 3), (2, 3, None, 5)],
        }, [], "FEASIBLE")
        return robots, tasks, result, _failure_at(robots, tasks, result, 1, 1)

    def test_replan_passes_only_the_failed_robots_actual_cargo(self):
        robots, tasks, result, failure = self.recovery_fixture()

        class Solver:
            def solve(self, alive, left, **kwargs):
                self.kwargs = kwargs
                return ScheduleResult(True, 0, {}, [], "FEASIBLE")

        solver = Solver()
        replan_after_failure(solver, robots, tasks, result, failure, 10)
        self.assertEqual(solver.kwargs["blocked_vertices"], {1: 1})
        self.assertEqual(solver.kwargs["blocked_cargo"], {1: 7})

    def test_frozen_robot_keeps_cargo_until_the_actual_receiver_pickup(self):
        robots, tasks, result, failure = self.recovery_fixture()
        sub = ScheduleResult(True, 3, {
            1: [(2, 2, None, 4)], 2: [(2, 1, None, 3)], 3: [(2, 2, 7, 2)],
        }, [], "FEASIBLE")
        merged = merge_plans(robots, result, failure, sub)
        self.assertEqual(merged.ticks[2][0], (1, 1, 7, 4))
        self.assertEqual(merged.ticks[3][0], (1, 1, 7, 4))
        self.assertEqual(merged.ticks[4][0], (1, 1, None, 4))
        self.assertIn("Такт 3: Груз #7 передан от Робота #1 -> Роботу #2 в вершине 1", merged.handoff_events)


if __name__ == "__main__":
    unittest.main()
