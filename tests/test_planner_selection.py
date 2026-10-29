import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from planners import create_solver
from test_run import GraphHandoffCpSatSolver, RobotSpec, TaskSpec, cp_model
import failure
import visualize


ROOT = Path(__file__).resolve().parents[1]


class PlannerSelectionTests(unittest.TestCase):
    def test_default_and_unknown_planner(self):
        self.assertIsInstance(create_solver(2, [(0, 1)]), GraphHandoffCpSatSolver)
        with self.assertRaises(ValueError):
            create_solver(2, [(0, 1)], planner="unknown")

    def test_greedy_solves_without_sat(self):
        with patch.object(cp_model, "CpSolver", side_effect=AssertionError("SAT called")):
            result = create_solver(2, [(0, 1)], planner="greedy", seed=42).solve(
                [RobotSpec(1, 0, 2)], [TaskSpec(1, 0, 1)], max_horizon=4
            )
        self.assertTrue(result.success)
        self.assertEqual(result.solver_status, "FEASIBLE")
        self.assertEqual(result.ticks[result.makespan][0][1], 1)

    def test_solver_cli_accepts_planner_and_seed(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder), \
                patch.object(sys, "argv", ["test_run.py", "--planner", "greedy", "--seed", "42"]), \
                patch("builtins.input", side_effect=["2 2", "1"]), \
                patch.object(cp_model, "CpSolver", side_effect=AssertionError("SAT called")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(ROOT / "test_run.py"), run_name="__main__")
        self.assertNotIn("Решение не найдено", output.getvalue())
        self.assertGreater(len(output.getvalue().splitlines()), 2)

    def test_generated_visualization_retains_planner(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder), \
                patch.object(cp_model, "CpSolver", side_effect=AssertionError("SAT called")):
            scenarios = visualize.generated_warehouse(
                2, 2, 1, seed=7, num_robots=1, battery=10, max_horizon=10,
                planner="greedy", planner_seed=42, failures=[{"p_fail": 0, "seed": 99}],
            )
            path = Path(visualize.export_html(scenarios, "demo.html"))
            payload = re.search(r"^const SCENARIOS = (.*);$", path.read_text(encoding="utf-8"), re.M)
            self.assertEqual(json.loads(payload.group(1)), scenarios)
        self.assertEqual(len(scenarios), 2)
        self.assertTrue(all(s["delivered"] == 1 and not s["optimal"] for s in scenarios))

    def test_failure_replans_with_greedy(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        robots = [RobotSpec(1, 0, 10), RobotSpec(2, 1, 10)]
        tasks = [TaskSpec(1, 1, 3)]
        with patch.object(cp_model, "CpSolver", side_effect=AssertionError("SAT called")):
            run = failure.run_with_failure(
                4, edges, robots, tasks, forced=(1, 1), max_horizon=10,
                planner="greedy", planner_seed=42,
            )
        scenario = visualize.failure_scenario(
            4, edges, robots, tasks, run, "failure", coords=[[v / 3, 0] for v in range(4)]
        )
        self.assertIsNotNone(run.failure)
        self.assertEqual(run.replan.solver_status, "FEASIBLE")
        self.assertEqual(scenario["delivered"], 1)
        self.assertFalse(scenario["optimal"])

    def test_passport_builders_forward_planner(self):
        with patch("visualize.create_solver") as factory, patch("visualize.scenario"):
            visualize.passport_reference(planner="greedy", planner_seed=42)
        self.assertEqual(factory.call_args.kwargs, {"planner": "greedy", "seed": 42})
        with patch("visualize.run_with_failure") as run, patch("visualize.failure_scenario"):
            visualize.passport_failure(planner="greedy", planner_seed=42)
        self.assertEqual(run.call_args.kwargs["planner"], "greedy")
        self.assertEqual(run.call_args.kwargs["planner_seed"], 42)

    def test_visualization_cli_documents_options(self):
        with patch.object(sys, "argv", ["visualize.py", "demo.html", "--help"]), \
                contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as stopped:
            runpy.run_path(str(ROOT / "visualize.py"), run_name="__main__")
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--planner", output.getvalue())
        self.assertIn("--seed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
