import unittest

from validation import verify_result


class ReleaseValidationTests(unittest.TestCase):
    def fixture(self):
        instance = dict(num_vertices=3, edges=[[0, 1], [1, 2]],
                        robots=[dict(robot_id=1, start_vertex=1, entry_vertex=0, release_time=2, battery=3)],
                        tasks=[dict(task_id=7, start_vertex=1, finish_vertex=2)])
        result = dict(success=True, makespan=4, ticks={
            1: [[1, None, None, 3]], 2: [[1, 0, None, 3]],
            3: [[1, 1, None, 2]], 4: [[1, 2, 7, 1]]})
        return instance, result

    def test_appearance_is_free_and_absence_counts_in_denominator(self):
        report = verify_result(*self.fixture())
        self.assertEqual(report["delivered"], 1)
        self.assertEqual(report["total_moves"], 2)
        self.assertEqual(report["busy_robot_ticks"], 2)
        self.assertEqual(report["robot_ticks_denominator"], 4)
        self.assertEqual(report["utilization"], 0.5)

    def test_zero_release_uses_entry_vertex_at_tick_zero(self):
        instance, _ = self.fixture()
        instance["robots"][0]["release_time"] = 0
        result = dict(success=True, makespan=2, ticks={
            1: [[1, 1, None, 2]], 2: [[1, 2, 7, 1]]})
        self.assertEqual(verify_result(instance, result)["total_moves"], 2)

    def test_rejects_early_late_wrong_or_loaded_appearance(self):
        changes = [(1, 1, 0), (2, 1, None), (2, 1, 1), (1, 2, 7), (2, 2, 7),
                   (1, 3, 2), (2, 3, 2), (3, 1, None)]
        for tick, column, value in changes:
            with self.subTest(change=(tick, column, value)):
                instance, result = self.fixture()
                result["ticks"][tick][0][column] = value
                with self.assertRaises((AssertionError, ValueError)):
                    verify_result(instance, result)

    def test_absent_robots_do_not_occupy_a_vertex(self):
        instance = dict(num_vertices=3, edges=[[0, 1], [1, 2]], tasks=[], robots=[
            dict(robot_id=r, start_vertex=3 - r, entry_vertex=0, release_time=r, battery=3)
            for r in (1, 2, 3)])
        result = dict(success=True, makespan=3, ticks={
            1: [[1, 0, None, 3], [2, None, None, 3], [3, None, None, 3]],
            2: [[1, 1, None, 2], [2, 0, None, 3], [3, None, None, 3]],
            3: [[1, 2, None, 1], [2, 1, None, 2], [3, 0, None, 3]]})
        self.assertEqual(verify_result(instance, result)["max_vertex_occupancy"], 1)


if __name__ == "__main__":
    unittest.main()
