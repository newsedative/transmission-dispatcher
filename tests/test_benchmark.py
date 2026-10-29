import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from benchmark import main, matrix_cases, run_pair
from instances import load_instance
from test_run import ScheduleResult


INSTANCE = dict(num_vertices=2, edges=[[0, 1]],
                robots=[dict(robot_id=1, start_vertex=0, battery=5)],
                tasks=[dict(task_id=7, start_vertex=0, finish_vertex=1)],
                coords=[[0, 0], [1, 0]], cells=[[0, 0], [0, 1]],
                config=dict(geometry=dict(topology="long_aisles", n=1, m=2, seed=42),
                            geometry_seed=42, placement_seed=43))
RESULT = ScheduleResult(True, 1, {1: [(1, 1, 7, 4)]}, [], "OPTIMAL")


class FakeSolver:
    def __init__(self, result):
        self.result = result

    def _solve_high_level(self, *args, **kwargs):
        return dict(success=self.result.success, status="FEASIBLE", tasks=[])

    def _solve_low_level_cbs(self, *args, **kwargs):
        return self.result

    def solve(self, robots, tasks, **kwargs):
        high = self._solve_high_level(robots, tasks)
        return self._solve_low_level_cbs(robots, tasks, high)


class BenchmarkTests(unittest.TestCase):
    def test_pair_uses_independent_inputs_and_equal_parameters(self):
        instance = copy.deepcopy(INSTANCE)
        calls = []

        def factory(vertices, edges, planner, seed):
            self.assertEqual([list(edge) for edge in edges], [[0, 1]])
            edges.append([1, 1])
            solver = FakeSolver(copy.deepcopy(RESULT))
            solve = solver.solve

            def observe(robots, tasks, **kwargs):
                calls.append((planner, robots, tasks, kwargs))
                return solve(robots, tasks, **kwargs)

            solver.solve = observe
            return solver

        with tempfile.TemporaryDirectory() as folder, patch("benchmark.create_solver", side_effect=factory):
            output = Path(folder) / "pair"
            rows = run_pair(instance, output, max_horizon=12, time_limit_s=6, planner_seed=4)
            self.assertEqual(len(rows), 2)
            self.assertTrue((output / "input.json").exists())
            self.assertTrue((output / "dispatch.html").exists())
            html = (output / "dispatch.html").read_text(encoding="utf-8")
            views = json.loads(re.search(r"^const SCENARIOS = (.*);$", html, re.M).group(1))
            self.assertTrue(all(view["warehouse"]["n"] == 1 and view["warehouse"]["m"] == 2
                                and view["warehouse"]["topology"] == "long_aisles" for view in views))
            for row in rows:
                self.assertTrue(row["valid"], row.get("error"))
                self.assertEqual(row["status"], "FEASIBLE")
                self.assertEqual(row["last_delivery_tick"], 1)
                self.assertEqual(row["utilization"], 1)
                self.assertFalse(row["full_route_optimality_proven"])
                self.assertEqual(row['occupancy_rule'], 'one_robot_except_actual_transfer')
                self.assertIn("benchmark.py", row["source_sha256"])
                self.assertIn("python", row["runtime"])
                self.assertIn("ortools", row["runtime"])
                raw = json.loads((output / row["planner"] / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(raw["solver_status"], "OPTIMAL")
        self.assertEqual(instance, INSTANCE)
        self.assertEqual([call[0] for call in calls], ["greedy", "cp-sat"])
        self.assertIsNot(calls[0][1], calls[1][1])
        self.assertIsNot(calls[0][2], calls[1][2])
        self.assertEqual(calls[0][3], calls[1][3])
        self.assertEqual(calls[0][3], dict(max_horizon=12, time_limit_s=6))

    def test_failed_plans_keep_status_and_null_metrics(self):
        result = ScheduleResult(False, 0, {}, [], "UNKNOWN")
        with tempfile.TemporaryDirectory() as folder, \
                patch("benchmark.create_solver", side_effect=lambda *a, **k: FakeSolver(result)):
            rows = run_pair(copy.deepcopy(INSTANCE), Path(folder) / "pair")
        for row in rows:
            self.assertFalse(row["valid"])
            self.assertEqual(row["status"], "UNKNOWN")
            self.assertIsNone(row["last_delivery_tick"])
            self.assertIsNone(row["utilization"])

    def test_invalid_success_is_not_reported_as_a_completed_plan(self):
        result = copy.deepcopy(RESULT)
        result.ticks[1] = [(1, 1, 7, 5)]
        with tempfile.TemporaryDirectory() as folder, \
                patch("benchmark.create_solver", side_effect=lambda *a, **k: FakeSolver(result)):
            rows = run_pair(copy.deepcopy(INSTANCE), Path(folder) / "pair")
        for row in rows:
            self.assertEqual(row["status"], "INVALID")
            self.assertFalse(row["valid"])
            self.assertTrue(row["error"])
            self.assertIsNone(row["utilization"])

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "pair"
            output.mkdir()
            sentinel = output / "input.json"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_pair(copy.deepcopy(INSTANCE), output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_input_mutation_stops_the_comparison(self):
        instance = copy.deepcopy(INSTANCE)

        def tamper(*args, **kwargs):
            instance["config"]["tampered"] = True
            return FakeSolver(copy.deepcopy(RESULT))

        with tempfile.TemporaryDirectory() as folder, patch("benchmark.create_solver", side_effect=tamper) as factory:
            with self.assertRaisesRegex(RuntimeError, "shared input"):
                run_pair(instance, Path(folder) / "pair")
        self.assertEqual(factory.call_count, 1)

    def test_real_pair_records_native_cp_statistics_without_claiming_route_optimality(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = run_pair(copy.deepcopy(INSTANCE), Path(folder) / "pair", max_horizon=4, time_limit_s=6)
        self.assertTrue(all(row["valid"] and row["delivered"] == 1 for row in rows))
        cp = rows[1]
        self.assertEqual(cp["cp_sat"]["status"], "OPTIMAL")
        self.assertIsNotNone(cp["cp_sat"]["best_bound"])
        self.assertGreater(cp["timings_seconds"]["native_cp_sat"], 0)
        self.assertGreaterEqual(cp["timings_seconds"]["preprocessing_and_model"], 0)
        self.assertEqual(cp["status"], "FEASIBLE")
        self.assertFalse(cp["full_route_optimality_proven"])

    def test_geometry_config_can_set_dimensions_and_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            geometry = Path(folder) / "geometry.json"
            geometry.write_text(json.dumps(dict(n=2, m=3, seed=7)), encoding="utf-8")
            output = Path(folder) / "experiment"
            status = main(["--config", str(geometry), "--output", str(output),
                           "--topologies", "long_aisles", "--tasks", "1", "--robots", "1",
                           "--horizon", "10", "--time-limit", "6"])
            self.assertEqual(status, 0)
            saved = list(output.glob("*/input.json"))
            self.assertEqual(len(saved), 1)
            instance = load_instance(saved[0])
            self.assertEqual(instance["config"]["geometry_seed"], 7)
            self.assertEqual(instance["config"]["geometry"]["n"], 2)
            self.assertEqual(instance["config"]["geometry"]["m"], 3)

    def test_matrix_has_twelve_instances_and_twenty_four_solver_runs(self):
        cases = matrix_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual({case["topology"] for case in cases},
                         {"long_aisles", "cross_aisles", "zoned"})
        self.assertEqual({case["geometry_seed"] for case in cases}, {42, 43})
        self.assertEqual({case["num_tasks"] for case in cases}, {3, 8})
        self.assertEqual(len({(c["topology"], c["geometry_seed"], c["num_tasks"]) for c in cases}), 12)

    def test_placement_options_reach_the_saved_input(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "experiment"
            with patch("benchmark.run_pair", return_value=[]) as pair:
                main(["--output", str(output), "--topologies", "cross_aisles",
                      "--n", "8", "--m", "12", "--tasks", "4", "--robots", "3",
                      "--robot-placement", "corner", "--spawn-interval", "3",
                      "--task-placement", "distant", "--min-task-distance", "12"])
            configuration = json.loads((output / 'configuration.json').read_text())
            self.assertEqual(configuration['robot_placement'], 'corner')
            self.assertEqual(configuration['spawn_interval'], 3)
            instance = pair.call_args.args[0]
            self.assertEqual([r['release_time'] for r in instance['robots']], [0, 3, 6])
            self.assertEqual(len({r['entry_vertex'] for r in instance['robots']}), 1)
            self.assertEqual(instance['config']['placement']['min_task_distance'], 12)


if __name__ == "__main__":
    unittest.main()
