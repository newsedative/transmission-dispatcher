import unittest
from dataclasses import asdict

from greedy import GreedyHandoffSolver
from test_run import GraphHandoffCpSatSolver, RobotSpec, TaskSpec
from validation import verify_result


class RobotEntryTests(unittest.TestCase):
    edges = [(v, v + 1) for v in range(8)]

    def test_release_and_deployment_are_physical_in_both_planners(self):
        robot = RobotSpec(7, 2, 8, release_time=3, entry_vertex=0)
        task = TaskSpec(4, 3, 6)
        instance = dict(num_vertices=9, edges=self.edges,
                        robots=[asdict(robot)], tasks=[asdict(task)])
        for planner in (GreedyHandoffSolver, GraphHandoffCpSatSolver):
            with self.subTest(planner=planner.__name__):
                result = planner(9, self.edges).solve([robot], [task],
                                                     max_horizon=20, time_limit_s=2)
                self.assertTrue(result.success, result.solver_status)
                self.assertEqual(result.ticks[1], [(7, None, None, 8)])
                self.assertEqual(result.ticks[2], [(7, None, None, 8)])
                self.assertEqual(result.ticks[3], [(7, 0, None, 8)])
                metrics = verify_result(instance, result)
                self.assertEqual(metrics['last_delivery_tick'], 9)
                self.assertEqual(metrics['total_moves'], 6)

    def test_deployment_energy_is_not_available_for_delivery(self):
        robot = RobotSpec(1, 2, 5, release_time=1, entry_vertex=0)
        task = TaskSpec(0, 2, 6)
        for planner in (GreedyHandoffSolver, GraphHandoffCpSatSolver):
            with self.subTest(planner=planner.__name__):
                result = planner(9, self.edges).solve([robot], [task],
                                                     max_horizon=20, time_limit_s=2)
                self.assertFalse(result.success)

    def test_absent_robots_do_not_conflict(self):
        solver = GraphHandoffCpSatSolver(2, [(0, 1)])
        paths = {0: [0, 1, 1], 1: [None, None, 0],
                 2: [None, None, None], 3: [None, None, None]}
        self.assertIsNone(solver._find_vertex_conflict(paths, {}))

    def test_entry_blocked_at_release_is_not_silently_delayed(self):
        solver = GraphHandoffCpSatSolver(2, [(0, 1)])
        robot = RobotSpec(1, 1, 5, release_time=2, entry_vertex=0)
        self.assertIsNone(solver._plan_all_robots_chronological(
            [robot], [], {(0, 0, 2)}, 10, {}))

    def test_unused_robot_can_remain_outside_beyond_the_horizon(self):
        robots = [RobotSpec(1, 0, 2), RobotSpec(2, 0, 2, release_time=10, entry_vertex=0)]
        tasks = [TaskSpec(0, 0, 1)]
        instance = dict(num_vertices=2, edges=[(0, 1)], robots=[asdict(r) for r in robots],
                        tasks=[asdict(t) for t in tasks])
        for planner in (GreedyHandoffSolver, GraphHandoffCpSatSolver):
            with self.subTest(planner=planner.__name__):
                result = planner(2, [(0, 1)]).solve(robots, tasks, 5, time_limit_s=2)
                self.assertTrue(result.success, result.solver_status)
                self.assertEqual(verify_result(instance, result)['last_delivery_tick'], 1)
                self.assertTrue(all(row[1] is None for rows in result.ticks.values()
                                    for row in rows if row[0] == 2))

    def test_invalid_release_is_rejected(self):
        for release in (-1, 1.5, True):
            with self.subTest(release=release), self.assertRaises(ValueError):
                RobotSpec(1, 0, 5, release_time=release)

    def test_cp_accounts_for_approach_energy_when_a_handoff_is_needed(self):
        edges = [(r * 3 + c, nr * 3 + nc) for r in range(3) for c in range(3)
                 for nr, nc in ((r + 1, c), (r, c + 1)) if nr < 3 and nc < 3]
        robots = [RobotSpec(1, 2, 4), RobotSpec(2, 1, 4, release_time=2, entry_vertex=2)]
        tasks = [TaskSpec(1, 0, 8)]
        instance = dict(num_vertices=9, edges=edges, robots=[asdict(r) for r in robots],
                        tasks=[asdict(t) for t in tasks])
        result = GraphHandoffCpSatSolver(9, edges).solve(robots, tasks, 30, time_limit_s=2)
        self.assertTrue(result.success, result.solver_status)
        self.assertEqual(verify_result(instance, result)['handoffs'], 1)


if __name__ == '__main__':
    unittest.main()
