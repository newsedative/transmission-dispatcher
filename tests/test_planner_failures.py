import unittest
from unittest.mock import patch

from failure import run_with_failure
from greedy import GreedyHandoffSolver
from planners import create_solver
from test_run import RobotSpec, TaskSpec, cp_model
from visualize import failure_scenario, scenario


class PlannerFailureTests(unittest.TestCase):
    def setUp(self):
        forbidden = patch.object(cp_model, "CpSolver", side_effect=AssertionError("SAT called"))
        forbidden.start()
        self.addCleanup(forbidden.stop)

    def test_completed_greedy_plan_keeps_feasible_status_after_failure(self):
        robots = [RobotSpec(1, 0, 3)]
        tasks = [TaskSpec(1, 0, 1)]
        run = run_with_failure(
            2, [(0, 1)], robots, tasks, forced=(1, 1), planner="greedy"
        )
        self.assertEqual(run.original.solver_status, "FEASIBLE")
        self.assertEqual(run.replan_tasks, [])
        self.assertEqual(run.result.solver_status, "FEASIBLE")
        view = failure_scenario(2, [(0, 1)], robots, tasks, run, "done", coords=[[0, 0], [1, 0]])
        self.assertEqual(view["delivered"], 1)
        self.assertFalse(view["optimal"])

    def test_last_robot_failure_preserves_partial_plan(self):
        robots = [RobotSpec(1, 0, 5)]
        tasks = [TaskSpec(1, 0, 2)]
        run = run_with_failure(
            3, [(0, 1), (1, 2)], robots, tasks, forced=(1, 1), planner="greedy"
        )
        self.assertTrue(run.result.success)
        self.assertEqual(run.result.makespan, 1)
        self.assertEqual(run.result.solver_status, "FEASIBLE")
        self.assertEqual(run.replan_tasks, [TaskSpec(1, 1, 2)])
        self.assertEqual(run.result.ticks[1][0][1], 1)

    def test_failure_at_initial_tick_keeps_frozen_battery(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        robots = [RobotSpec(1, 0, 5), RobotSpec(2, 3, 5)]
        tasks = [TaskSpec(1, 1, 2)]
        run = run_with_failure(
            4, edges, robots, tasks, forced=(1, 0), planner="greedy", planner_seed=42
        )
        self.assertTrue(run.result.success)
        for rows in run.result.ticks.values():
            self.assertEqual(next(row for row in rows if row[0] == 1), (1, 0, None, 5))
        view = failure_scenario(4, edges, robots, tasks, run, "initial",
                                coords=[[v / 3, 0] for v in range(4)])
        self.assertEqual(view["delivered"], 1)
        self.assertFalse(view["optimal"])

    def test_failure_before_at_and_after_handoff_obeys_occupancy(self):
        edges = [(v, v + 1) for v in range(4)]
        robots = [RobotSpec(1, 0, 10), RobotSpec(2, 2, 10)]
        tasks = [TaskSpec(1, 0, 4)]
        original = GreedyHandoffSolver(5, edges, handoff_probability=1).solve(robots, tasks)
        self.assertTrue(original.handoff_events)
        for forced in [(1, 1), (1, 2), (2, 2), (2, 3)]:
            with self.subTest(forced=forced):
                if forced == (2, 2):
                    # Пустой приёмщик сломался до передачи: два робота уже
                    # находятся вместе, а фактической передачи не будет.
                    with self.assertRaisesRegex(RuntimeError, 'NO_PLAN'):
                        run_with_failure(5, edges, robots, tasks, original=original,
                                         forced=forced, planner='greedy', planner_seed=42)
                    continue
                run = run_with_failure(
                    5, edges, robots, tasks, original=original, forced=forced,
                    planner="greedy", planner_seed=42,
                )
                view = failure_scenario(
                    5, edges, robots, tasks, run, "handoff", coords=[[v / 4, 0] for v in range(5)]
                )
                self.assertEqual(view["delivered"], 1)
                self.assertEqual(run.result.solver_status, "FEASIBLE")

    def test_partial_replan_keeps_already_delivered_task(self):
        edges = [(v, v + 1) for v in range(4)]
        robots = [RobotSpec(1, 0, 10), RobotSpec(2, 4, 1)]
        tasks = [TaskSpec(1, 0, 1), TaskSpec(2, 1, 4)]
        run = run_with_failure(
            5, edges, robots, tasks, forced=(1, 2), planner="greedy", planner_seed=42
        )
        view = failure_scenario(
            5, edges, robots, tasks, run, "partial", coords=[[v / 4, 0] for v in range(5)]
        )
        self.assertEqual(view["delivered"], 1)
        self.assertEqual(view["failure"]["undelivered"], [1])
        self.assertEqual(run.result.solver_status, "FEASIBLE")

    def test_empty_greedy_plan_can_be_visualized(self):
        result = create_solver(1, [], planner="greedy").solve([], [])
        view = scenario(1, [], [], [], result, "empty", coords=[[0, 0]])
        self.assertEqual(view["delivered"], 0)
        self.assertEqual(view["makespan"], 0)
        self.assertFalse(view["optimal"])


if __name__ == "__main__":
    unittest.main()
