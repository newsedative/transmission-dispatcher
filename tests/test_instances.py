import copy
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import tempfile
import unittest

from instances import (add_geometry_arguments, create_instance, fingerprint,
                       geometry_options, load_instance, save_instance, solver_inputs)


class InstanceTests(unittest.TestCase):
    def make(self, **kwargs):
        options = dict(n=12, m=16, num_tasks=4, num_robots=3, battery=80)
        return create_instance(**(options | kwargs))

    def test_seeded_placements_use_only_walkable_vertices(self):
        first = self.make(geometry_seed=7, placement_seed=9)
        self.assertEqual(first, self.make(geometry_seed=7, placement_seed=9))
        other = self.make(geometry_seed=7, placement_seed=10)
        self.assertEqual(first['edges'], other['edges'])
        self.assertEqual(first['cells'], other['cells'])
        self.assertNotEqual(first['tasks'], other['tasks'])
        vertices = set(range(first['num_vertices']))
        self.assertEqual(len(first['coords']), len(vertices))
        self.assertEqual(len({r['start_vertex'] for r in first['robots']}), 3)
        for robot in first['robots']:
            self.assertIn(robot['start_vertex'], vertices)
        for task in first['tasks']:
            self.assertIn(task['start_vertex'], vertices)
            self.assertIn(task['finish_vertex'], vertices)
            self.assertNotEqual(task['start_vertex'], task['finish_vertex'])

    def test_round_trip_and_checksum_detect_tampering(self):
        instance = self.make()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'instance.json'
            self.assertEqual(save_instance(instance, path), fingerprint(instance))
            self.assertEqual(load_instance(path), instance)
            with self.assertRaises(FileExistsError):
                save_instance(instance, path)
            saved = json.loads(path.read_text(encoding='utf-8'))
            saved['instance']['robots'][0]['battery'] += 1
            path.write_text(json.dumps(saved), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum|SHA|контроль'):
                load_instance(path)

    def test_solver_inputs_are_independent(self):
        instance = self.make()
        before = copy.deepcopy(instance)
        first = solver_inputs(instance)
        second = solver_inputs(instance)
        self.assertEqual(first, second)
        first[1].clear()
        first[2].clear()
        first[3].clear()
        self.assertEqual(instance, before)
        self.assertTrue(second[1])
        self.assertEqual(len(second[2]), 3)
        self.assertEqual(len(second[3]), 4)

    def test_invalid_load_is_rejected(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make(num_robots=value)
        with self.assertRaises(ValueError):
            create_instance(n=2, m=2, num_robots=5)
        with self.assertRaises(ValueError):
            self.make(battery=-1)
        with self.assertRaises(ValueError):
            self.make(num_tasks=-1)

    def test_zero_tasks_and_smallest_map(self):
        instance = create_instance(n=2, m=2, num_tasks=0, num_robots=1)
        self.assertEqual(instance['tasks'], [])
        self.assertEqual(len(instance['robots']), 1)


class PlacementTests(unittest.TestCase):
    make = InstanceTests.make

    def test_random_defaults_keep_legacy_input_and_robot_fields(self):
        legacy = self.make()
        self.assertEqual(fingerprint(legacy),
                         "89fc2fc89a9f54e63d19ec9042050eba3544f27bbfb6c2765b58e24f371bd1d5")
        self.assertEqual(legacy, self.make(robot_placement="random", spawn_interval=2,
                                           task_placement="random", min_task_distance=None))
        self.assertTrue(all(set(robot) == {"robot_id", "start_vertex", "battery"}
                            for robot in legacy["robots"]))
        self.assertNotIn("placement", legacy["config"])

    def test_corner_robots_enter_together_and_park_right_to_left(self):
        instance = self.make(robot_placement="corner", spawn_interval=3)
        cells, robots = instance["cells"], instance["robots"]
        self.assertEqual([cells[r["start_vertex"]] for r in robots], [[0, 15], [0, 14], [0, 13]])
        self.assertEqual([r["release_time"] for r in robots], [0, 3, 6])
        self.assertTrue(all(cells[r["entry_vertex"]] == [0, 15] for r in robots))
        self.assertEqual(len({r["start_vertex"] for r in robots}), 3)
        self.assertEqual(instance["config"]["placement"]["spawn_interval"], 3)

    def test_distant_tasks_meet_exact_bfs_distance_and_balance_directions(self):
        instance = self.make(n=16, m=20, topology="cross_aisles", num_tasks=12,
                             task_placement="distant", min_task_distance=24)
        adjacency = [[] for _ in range(instance["num_vertices"])]
        for a, b in instance["edges"]:
            adjacency[a].append(b)
            adjacency[b].append(a)
        directions = []
        for task in instance["tasks"]:
            start, finish = task["start_vertex"], task["finish_vertex"]
            distance, queue = {start: 0}, deque([start])
            while queue:
                vertex = queue.popleft()
                for other in adjacency[vertex]:
                    if other not in distance:
                        distance[other] = distance[vertex] + 1
                        queue.append(other)
            self.assertGreaterEqual(distance[finish], 24)
            a, b = instance["cells"][start], instance["cells"][finish]
            directions.append((b[0] > a[0], b[1] > a[1]))
        self.assertEqual(Counter(directions), {(True, True): 3, (False, False): 3,
                                               (True, False): 3, (False, True): 3})
        self.assertEqual(instance["config"]["placement"]["min_task_distance"], 24)

    def test_new_modes_reproduce_and_default_distance_is_recorded(self):
        options = dict(robot_placement="corner", task_placement="distant", placement_seed=9)
        first = self.make(**options)
        self.assertEqual(first, self.make(**options))
        self.assertEqual(first["config"]["placement"]["min_task_distance"], 20)
        self.assertNotEqual(first["tasks"], self.make(**(options | {"placement_seed": 10}))["tasks"])

    def test_placement_cli_options_reach_instance_creation(self):
        parser = argparse.ArgumentParser()
        add_geometry_arguments(parser)
        args = parser.parse_args(["--robot-placement", "corner", "--spawn-interval", "4",
                                  "--task-placement", "distant", "--min-task-distance", "12"])
        instance = self.make(**geometry_options(args))
        self.assertEqual(instance["config"]["placement"],
                         dict(robot_placement="corner", spawn_interval=4,
                              task_placement="distant", min_task_distance=12))
        self.assertNotIn("robot_placement", instance["config"]["geometry"])

    def test_invalid_placement_settings_are_rejected(self):
        for options in (dict(robot_placement="unknown"), dict(task_placement="unknown"),
                        dict(spawn_interval=0), dict(spawn_interval=True),
                        dict(spawn_interval=1.5), dict(min_task_distance=0),
                        dict(robot_placement="corner", num_robots=17),
                        dict(task_placement="distant", min_task_distance=10000)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make(**options)


if __name__ == '__main__':
    unittest.main()
