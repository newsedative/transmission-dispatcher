import unittest
from dataclasses import asdict

from greedy import GreedyHandoffSolver
from test_run import GraphHandoffCpSatSolver, RobotSpec, TaskSpec
from validation import verify_result


class HandoffCapacityTests(unittest.TestCase):
    def setUp(self):
        self.solver = GraphHandoffCpSatSolver(5, [(v, v + 1) for v in range(4)])

    def test_two_empty_robots_cannot_cross_in_one_cell(self):
        paths = {0: [0, 1, 2], 1: [2, 1, 0]}
        self.assertEqual(self.solver._find_vertex_conflict(paths, {}), (1, 1, [0, 1]))

    def test_only_the_actual_transfer_boundary_allows_two(self):
        paths = {0: [0, 1, 1], 1: [2, 1, 2]}
        held = {0: [None, 7, None], 1: [None, None, 7]}
        self.assertIsNone(self.solver._find_vertex_conflict(paths, {}, held))
        held[1][2] = None
        self.assertEqual(self.solver._find_vertex_conflict(paths, {}, held), (1, 1, [0, 1]))

    def test_waiting_together_before_or_after_a_handoff_is_a_conflict(self):
        paths = {0: [0, 1, 1, 1], 1: [2, 1, 1, 2]}
        held = {0: [None, 7, 7, None], 1: [None, None, None, 7]}
        self.assertEqual(self.solver._find_vertex_conflict(paths, {}, held), (1, 1, [0, 1]))
        held = {0: [None, 7, None, None], 1: [None, None, 7, 7]}
        self.assertEqual(self.solver._find_vertex_conflict(paths, {}, held), (1, 2, [0, 1]))

    def test_three_robots_are_forbidden_even_during_a_transfer(self):
        paths = {0: [0, 1, 1], 1: [2, 1, 2], 2: [3, 1, 3]}
        held = {0: [None, 7, None], 1: [None, None, 7], 2: [None] * 3}
        self.assertEqual(self.solver._find_vertex_conflict(paths, {}, held), (1, 1, [0, 1, 2]))

    def test_initial_overlap_and_parking_with_a_broken_robot_are_forbidden(self):
        self.assertEqual(self.solver._find_vertex_conflict({0: [1], 1: [1]}, {}), (1, 0, [0, 1]))
        self.assertEqual(self.solver._find_vertex_conflict({0: [1]}, {1: 1}), (1, 0, [0]))

    def test_idle_robot_can_clear_the_common_entry(self):
        edges = [(0, 1), (1, 2)]
        robots = [RobotSpec(1, 0, 5), RobotSpec(2, 1, 5, release_time=2, entry_vertex=0)]
        tasks = [TaskSpec(0, 1, 2)]
        instance = dict(num_vertices=3, edges=edges, robots=[asdict(r) for r in robots],
                        tasks=[asdict(t) for t in tasks])
        for planner in (GraphHandoffCpSatSolver, GreedyHandoffSolver):
            with self.subTest(planner=planner.__name__):
                result = planner(3, edges).solve(robots, tasks, max_horizon=12, time_limit_s=2)
                self.assertTrue(result.success, result.solver_status)
                self.assertEqual(verify_result(instance, result)['delivered'], 1)
                self.assertTrue(all(len({v for _, v, _, _ in rows if v is not None}) ==
                                    sum(v is not None for _, v, _, _ in rows)
                                    for rows in result.ticks.values()))

    def test_recovery_is_a_single_transfer_from_a_frozen_cargo_owner(self):
        paths = {0: [0, 1, 2]}
        held = {0: [None, None, 7]}
        self.assertIsNone(self.solver._find_vertex_conflict(paths, {1: 1}, held, {1: 7}))
        self.assertEqual(self.solver._find_vertex_conflict(paths, {1: 1}, held, {1: 8}),
                         (1, 1, [0]))
        paths[0] += [1, 2]
        held[0] += [None, 7]
        self.assertEqual(self.solver._find_vertex_conflict(paths, {1: 1}, held, {1: 7}),
                         (1, 3, [0]))

    def test_an_empty_mission_does_not_allow_initial_sharing(self):
        robots = [RobotSpec(1, 0, 3), RobotSpec(2, 0, 3)]
        for planner in (GraphHandoffCpSatSolver, GreedyHandoffSolver):
            with self.subTest(planner=planner.__name__):
                self.assertFalse(planner(2, [(0, 1)]).solve(robots, []).success)


if __name__ == '__main__':
    unittest.main()
