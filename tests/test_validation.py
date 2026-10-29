import copy
from dataclasses import dataclass
import unittest

from validation import verify_result


INSTANCE = dict(num_vertices=3, edges=[[0, 1], [1, 2]],
                robots=[dict(robot_id=10, start_vertex=0, battery=5),
                        dict(robot_id=20, start_vertex=1, battery=5)],
                tasks=[dict(task_id=7, start_vertex=0, finish_vertex=2)])
RESULT = dict(success=True, makespan=4, solver_status="OPTIMAL", handoff_events=[],
              ticks={"1": [[10, 0, 7, 5], [20, 1, None, 5]],
                     "2": [[10, 1, 7, 4], [20, 1, None, 5]],
                     "3": [[10, 1, None, 4], [20, 2, 7, 4]],
                     "4": [[10, 1, None, 4], [20, 2, 7, 4]]})


@dataclass
class SavedResult:
    success: bool
    makespan: int
    ticks: dict
    handoff_events: list
    solver_status: str


class ValidationTests(unittest.TestCase):
    def test_physical_delivery_and_loaded_wait_utilization(self):
        report = verify_result(INSTANCE, RESULT)
        self.assertTrue(report["valid"])
        self.assertEqual(report["delivered"], 1)
        self.assertEqual(report["last_delivery_tick"], 3)
        self.assertEqual(report["handoffs"], 1)
        self.assertEqual(report["total_moves"], 2)
        self.assertEqual(report["max_vertex_occupancy"], 2)
        self.assertEqual(report["busy_robot_ticks"], 3)
        self.assertEqual(report["moving_robot_ticks"], 2)
        self.assertEqual(report["robot_ticks_denominator"], 6)
        self.assertEqual(report["utilization"], 0.5)
        self.assertEqual(report["movement_utilization"], 1 / 3)

    def test_dataclass_and_integer_tick_keys(self):
        result = copy.deepcopy(RESULT)
        result["ticks"] = {int(t): rows for t, rows in result["ticks"].items()}
        self.assertEqual(verify_result(INSTANCE, SavedResult(**result)),
                         verify_result(INSTANCE, RESULT))

    def test_trailing_empty_motion_does_not_extend_delivery_window(self):
        result = copy.deepcopy(RESULT)
        result["ticks"]["4"][0] = [10, 0, None, 3]
        result["ticks"]["4"][1] = [20, 2, None, 4]
        report = verify_result(INSTANCE, result)
        self.assertEqual(report["last_delivery_tick"], 3)
        self.assertEqual(report["total_moves"], 3)
        self.assertEqual(report["moving_robot_ticks"], 2)
        self.assertEqual(report["robot_ticks_denominator"], 6)

    def test_already_delivered_task_has_an_empty_window(self):
        instance = copy.deepcopy(INSTANCE)
        instance["tasks"][0]["finish_vertex"] = 0
        result = dict(success=True, makespan=0, ticks={})
        report = verify_result(instance, result)
        self.assertEqual(report["delivered"], 1)
        self.assertEqual(report["last_delivery_tick"], 0)
        self.assertEqual(report["robot_ticks_denominator"], 0)
        self.assertIsNone(report["utilization"])
        self.assertIsNone(report["movement_utilization"])

    def test_optional_horizon_checks_makespan_not_last_delivery(self):
        for config in ({}, {"max_horizon": 4}):
            with self.subTest(config=config):
                self.assertTrue(verify_result(dict(INSTANCE, config=config), RESULT)["valid"])
        with self.assertRaises((AssertionError, ValueError)):
            verify_result(dict(INSTANCE, config={"max_horizon": 3}), RESULT)

    def test_rejects_failed_result(self):
        with self.assertRaises((AssertionError, ValueError)):
            verify_result(INSTANCE, dict(RESULT, success=False))

    def test_rejects_broken_tick_rows(self):
        def missing_tick(result):
            result["ticks"].pop("2")

        def duplicate_tick(result):
            result["ticks"][1] = result["ticks"]["1"]

        def duplicate_robot(result):
            result["ticks"]["1"][1][0] = 10

        for mutation in (missing_tick, duplicate_tick, duplicate_robot):
            with self.subTest(mutation=mutation.__name__):
                result = copy.deepcopy(RESULT)
                mutation(result)
                with self.assertRaises((AssertionError, ValueError)):
                    verify_result(INSTANCE, result)

    def test_rejects_invalid_moves_batteries_and_cargo(self):
        mutations = {
            "nonedge_move": ("2", 0, 1, 2),
            "wrong_battery": ("1", 0, 3, 4),
            "unknown_cargo": ("1", 0, 2, 99),
            "duplicate_cargo": ("2", 1, 2, 7),
            "premature_floor_drop": ("2", 0, 2, None),
        }
        for label, (tick, row, column, value) in mutations.items():
            with self.subTest(label=label):
                result = copy.deepcopy(RESULT)
                result["ticks"][tick][row][column] = value
                with self.assertRaises((AssertionError, ValueError)):
                    verify_result(INSTANCE, result)

    def test_rejects_pickup_away_from_the_cargo(self):
        result = copy.deepcopy(RESULT)
        result["ticks"]["1"][0][2] = None
        result["ticks"]["1"][1][2] = 7
        with self.assertRaises((AssertionError, ValueError)):
            verify_result(INSTANCE, result)

    def test_rejects_transfer_without_collocation(self):
        result = copy.deepcopy(RESULT)
        for tick in ("1", "2"):
            result["ticks"][tick][1] = [20, 2, None, 4]
        with self.assertRaises((AssertionError, ValueError)):
            verify_result(INSTANCE, result)

    def test_rejects_exhausted_battery(self):
        instance = dict(num_vertices=2, edges=[[0, 1]],
                        robots=[dict(robot_id=1, start_vertex=0, battery=0)],
                        tasks=[dict(task_id=1, start_vertex=0, finish_vertex=1)])
        result = dict(success=True, makespan=1, ticks={1: [[1, 1, 1, -1]]})
        with self.assertRaises((AssertionError, ValueError)):
            verify_result(instance, result)

    def test_rejects_capacity_overflow_including_initial_state(self):
        for start in (0, 1):
            with self.subTest(third_robot_start=start):
                instance, result = copy.deepcopy(INSTANCE), copy.deepcopy(RESULT)
                instance["robots"].append(dict(robot_id=30, start_vertex=start, battery=5))
                if start == 0:
                    instance["robots"][1]["start_vertex"] = 0
                for rows in result["ticks"].values():
                    rows.append([30, start, None, 5])
                with self.assertRaises((AssertionError, ValueError)):
                    verify_result(instance, result)

    def test_rejects_incomplete_delivery(self):
        result = copy.deepcopy(RESULT)
        for tick in ("3", "4"):
            result["ticks"][tick][1] = [20, 1, 7, 5]
        with self.assertRaises((AssertionError, ValueError)):
            verify_result(INSTANCE, result)


if __name__ == "__main__":
    unittest.main()
