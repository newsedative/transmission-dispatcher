import unittest
from unittest.mock import patch

import test_run
from test_run import GraphHandoffCpSatSolver, SpaceTimeAStar


class TimedArrivalTests(unittest.TestCase):
    def test_long_wait_does_not_expand_the_whole_time_grid(self):
        edges = [(v, v + d) for v in range(64) for d in (1, 8)
                 if v + d < 64 and (d != 1 or v // 8 == (v + 1) // 8)]
        solver = GraphHandoffCpSatSolver(64, edges)
        pop = test_run.heapq.heappop
        count = 0

        def limited_pop(queue):
            nonlocal count
            count += 1
            self.assertLess(count, 2000)
            return pop(queue)

        with patch.object(test_run.heapq, 'heappop', limited_pop):
            path = SpaceTimeAStar.plan(0, 0, 0, 100, 63, 500, 600,
                                      solver.adj, solver.oracle, {(0, 63, 500)}, 2, {})
        self.assertEqual(len(path), 502)
        self.assertEqual(path[-1][0], 63)

    def test_wait_avoids_unnecessary_battery_use(self):
        solver = GraphHandoffCpSatSolver(3, [(0, 1), (1, 2)])
        path = SpaceTimeAStar.plan(0, 0, 0, 4, 2, 4, 10,
                                  solver.adj, solver.oracle, {(0, 2, 3)}, 2, {})
        self.assertEqual(len(path), 5)
        self.assertEqual(path[-1], (2, 2))

    def test_insufficient_energy_does_not_search_a_long_horizon(self):
        solver = GraphHandoffCpSatSolver(4, [(0, 1), (1, 2), (2, 3)])
        pop = test_run.heapq.heappop
        count = 0

        def limited_pop(queue):
            nonlocal count
            count += 1
            self.assertLess(count, 32, "Waiting cannot replenish energy")
            return pop(queue)

        with patch.object(test_run.heapq, 'heappop', limited_pop):
            path = SpaceTimeAStar.plan(0, 0, 0, 1, 3, 0, 100000,
                                      solver.adj, solver.oracle, set(), 2, {})
        self.assertIsNone(path)


if __name__ == '__main__':
    unittest.main()


