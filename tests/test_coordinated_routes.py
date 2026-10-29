import unittest
from dataclasses import asdict
from unittest.mock import patch

from coordinated_routes import plan_routes
from greedy import GreedyHandoffSolver
from test_run import GraphHandoffCpSatSolver, RobotSpec, TaskSpec
from validation import verify_result


class CoordinatedRouteTests(unittest.TestCase):
    def check(self, vertices, edges, robots, tasks, high, horizon=30):
        solver = GraphHandoffCpSatSolver(vertices, edges)
        with patch('test_run.cp_model.CpSolver', side_effect=AssertionError('Routing called SAT')):
            result = plan_routes(solver, robots, tasks, high, horizon, 5)
        self.assertIsNotNone(result)
        instance = dict(num_vertices=vertices, edges=edges,
                        robots=[asdict(r) for r in robots], tasks=[asdict(t) for t in tasks])
        checked = verify_result(instance, result)
        self.assertEqual(checked['delivered'], len(tasks))
        self.assertEqual(result.solver_status, 'FEASIBLE')
        return result, checked

    def test_handoff_shares_only_the_actual_transfer_boundary(self):
        tasks = [TaskSpec(0, 0, 2)]
        high = [dict(task_id=0, start=0, finish=2, relay=True, r1=0, r2=1,
                     handoff_v=1, t_start=0, t_mid=1, t_end=2)]
        _, checked = self.check(3, [(0, 1), (1, 2)],
                                [RobotSpec(1, 0, 1), RobotSpec(2, 1, 1)], tasks, high)
        self.assertEqual(checked['handoffs'], 1)

    def test_opposing_robots_cannot_meet_in_the_crossing(self):
        tasks = [TaskSpec(0, 0, 2), TaskSpec(1, 2, 0)]
        high = [dict(task_id=i, start=t.start_vertex, finish=t.finish_vertex,
                     relay=False, r1=i, t_start=0, t_end=2) for i, t in enumerate(tasks)]
        self.check(3, [(0, 1), (1, 2)],
                   [RobotSpec(1, 0, 5), RobotSpec(2, 2, 5)], tasks, high)

    def test_releases_and_idle_robot_clearing_the_entry(self):
        tasks = [TaskSpec(0, 1, 2)]
        high = [dict(task_id=0, start=1, finish=2, relay=False, r1=1, t_start=3, t_end=4)]
        result, _ = self.check(3, [(0, 1), (1, 2)],
                              [RobotSpec(1, 0, 6), RobotSpec(2, 1, 6, 2, 0)], tasks, high)
        self.assertEqual(result.ticks[1][1], (2, None, None, 6))
        self.assertEqual(result.ticks[2][1], (2, 0, None, 6))

    def test_routing_does_not_invent_energy(self):
        solver = GraphHandoffCpSatSolver(3, [(0, 1), (1, 2)])
        task = TaskSpec(0, 0, 2)
        high = [dict(task_id=0, start=0, finish=2, relay=False, r1=0, t_start=0, t_end=2)]
        self.assertIsNone(plan_routes(solver, [RobotSpec(1, 0, 1)], [task], high, 10, 2))

    def test_donor_does_not_block_the_receivers_previous_delivery(self):
        tasks = [TaskSpec(0, 0, 3), TaskSpec(1, 3, 1)]
        high = [dict(task_id=1, start=3, finish=1, relay=False, r1=1, t_start=0, t_end=2),
                dict(task_id=0, start=0, finish=3, relay=True, r1=0, r2=1,
                     handoff_v=1, t_start=0, t_mid=2, t_end=4)]
        _, checked = self.check(4, [(0, 1), (1, 2), (2, 3)],
                                [RobotSpec(1, 0, 1), RobotSpec(2, 3, 4)], tasks, high, 8)
        self.assertEqual(checked['handoffs'], 1)
        self.assertEqual(checked['total_moves'], 5)

    def test_waiting_donor_yields_to_a_chain_of_delivering_robots(self):
        tasks = [TaskSpec(0, 0, 3), TaskSpec(1, 1, 4), TaskSpec(2, 2, 0), TaskSpec(3, 4, 0)]
        high = [dict(task_id=k, start=tasks[k].start_vertex, finish=tasks[k].finish_vertex,
                     relay=False, r1=r, t_start=0, t_end=4) for k, r in ((0, 0), (2, 2), (3, 3))]
        high.append(dict(task_id=1, start=1, finish=4, relay=True, r1=1, r2=3,
                         handoff_v=0, t_start=0, t_mid=4, t_end=8))
        self.check(5, [(v, v + 1) for v in range(4)],
                   [RobotSpec(r + 1, v, 30) for r, v in enumerate((0, 1, 2, 4))], tasks, high)

    def test_receiver_can_finish_its_delivery_and_take_the_next_cargo(self):
        tasks = [TaskSpec(0, 2, 0), TaskSpec(1, 3, 1)]
        high = [dict(task_id=1, start=3, finish=1, relay=False, r1=1, t_start=0, t_end=2),
                dict(task_id=0, start=2, finish=0, relay=True, r1=0, r2=1,
                     handoff_v=1, t_start=0, t_mid=2, t_end=3)]
        _, checked = self.check(4, [(0, 1), (1, 2), (2, 3)],
                                [RobotSpec(1, 2, 1), RobotSpec(2, 3, 3)], tasks, high, 8)
        self.assertEqual(checked['handoffs'], 1)
        self.assertEqual(checked['total_moves'], 4)

    def test_public_planners_clear_a_busy_junction(self):
        edges = [(0, v) for v in range(1, 9)]
        robots = [RobotSpec(r, r, 100) for r in range(1, 9)]
        tasks = [TaskSpec(r, r, r % 8 + 1) for r in range(1, 9)]
        instance = dict(num_vertices=9, edges=edges, robots=[asdict(r) for r in robots],
                        tasks=[asdict(t) for t in tasks])
        for planner in (GreedyHandoffSolver, GraphHandoffCpSatSolver):
            with self.subTest(planner=planner.__name__):
                result = planner(9, edges).solve(robots, tasks, max_horizon=40, time_limit_s=3)
                self.assertTrue(result.success, result.solver_status)
                self.assertEqual(verify_result(instance, result)['delivered'], 8)


if __name__ == '__main__':
    unittest.main()
