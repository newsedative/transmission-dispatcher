import unittest

from test_run import RobotSpec, ScheduleResult, TaskSpec
from validation import verify_result
from visualize import plan_from_ticks, validate


class TransferOccupancyTests(unittest.TestCase):
    def fixture(self):
        instance = dict(num_vertices=4, edges=[[0, 1], [1, 2], [1, 3]],
                        robots=[dict(robot_id=10, start_vertex=0, battery=5),
                                dict(robot_id=20, start_vertex=2, battery=5)],
                        tasks=[dict(task_id=7, start_vertex=0, finish_vertex=2)])
        result = dict(success=True, makespan=3, solver_status="FEASIBLE", handoff_events=[], ticks={
            1: [[10, 1, 7, 4], [20, 1, None, 4]],
            2: [[10, 0, None, 3], [20, 2, 7, 3]],
            3: [[10, 0, None, 3], [20, 2, None, 3]]})
        return instance, result

    def legacy(self, instance, result):
        robots = [RobotSpec(**r) for r in instance["robots"]]
        tasks = [TaskSpec(**t) for t in instance["tasks"]]
        pos, hold = plan_from_ticks(robots, tasks, ScheduleResult(**result))
        return validate(instance["num_vertices"], instance["edges"], robots, tasks,
                        pos, hold, result["makespan"])

    def rejected(self, instance, result):
        for name, check in (("physical", verify_result), ("visual", self.legacy)):
            with self.subTest(validator=name), self.assertRaises((AssertionError, ValueError)):
                check(instance, result)

    def test_two_robots_allowed_only_on_actual_transfer_boundary(self):
        instance, result = self.fixture()
        report = verify_result(instance, result)
        self.assertEqual((report["handoffs"], report["max_vertex_occupancy"]), (1, 2))
        self.assertEqual(self.legacy(instance, result), 1)

    def test_crossing_without_cargo_is_not_a_transfer_even_with_fake_log(self):
        instance, result = self.fixture()
        instance["tasks"] = []
        for rows in result["ticks"].values():
            for row in rows:
                row[2] = None
        result["handoff_events"] = ["1 7 10 20 1"]
        self.rejected(instance, result)

    def test_waiting_together_before_or_after_transfer_is_forbidden(self):
        for carrier in (10, 20):
            with self.subTest(waiting_carrier=carrier):
                instance, result = self.fixture()
                result["ticks"] = {
                    1: result["ticks"][1],
                    2: [[10, 1, 7 if carrier == 10 else None, 4],
                        [20, 1, 7 if carrier == 20 else None, 4]],
                    3: result["ticks"][2], 4: result["ticks"][3]}
                result["makespan"] = 4
                self.rejected(instance, result)

    def test_no_transfer_exception_at_initial_or_final_boundary(self):
        for initial in (True, False):
            with self.subTest(initial=initial):
                instance, result = self.fixture()
                instance["tasks"] = []
                if initial:
                    instance["robots"][1]["start_vertex"] = 0
                    rows = [[10, 0, None, 5], [20, 1, None, 4]]
                else:
                    rows = [[10, 1, None, 4], [20, 1, None, 4]]
                result.update(makespan=1, ticks={1: rows})
                self.rejected(instance, result)

    def test_third_robot_is_forbidden_even_during_transfer(self):
        instance, result = self.fixture()
        instance["robots"].append(dict(robot_id=30, start_vertex=3, battery=5,
                                       release_time=1, entry_vertex=1))
        for tick, rows in result["ticks"].items():
            rows.append([30, 1 if tick == 1 else 3, None, 5 if tick == 1 else 4])
        self.rejected(instance, result)


if __name__ == "__main__":
    unittest.main()
