"""Warehouse geometry, seeded placements and checksummed experiment inputs."""

import hashlib
import json
import math
from collections import deque
from pathlib import Path
import random

from warehouse import generate_warehouse


def add_geometry_arguments(parser):
    parser.add_argument("--topology", choices=("long_aisles", "cross_aisles", "zoned"),
                        default="long_aisles")
    parser.add_argument("--geometry-seed", type=int, default=42)
    parser.add_argument("--placement-seed", type=int, default=42)
    parser.add_argument("--robot-placement", choices=("random", "corner"), default="random")
    parser.add_argument("--task-placement", choices=("random", "distant"), default="random")
    parser.add_argument("--spawn-interval", type=int, default=2)
    parser.add_argument("--min-task-distance", type=int)
    for name in ("aisle-width", "rack-width", "rack-count", "rack-length", "cross-aisles",
                 "passage-width", "split-row", "split-col"):
        parser.add_argument("--" + name, type=int)
    for name in ("cross-positions", "passage-positions"):
        parser.add_argument("--" + name, type=int, nargs="+")


def geometry_options(args):
    names = ("topology", "aisle_width", "rack_width", "rack_count", "rack_length",
             "cross_aisles", "cross_positions", "passage_width", "passage_positions",
             "split_row", "split_col", "robot_placement", "spawn_interval",
             "task_placement", "min_task_distance")
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def create_instance(n=15, m=20, topology="long_aisles", num_tasks=10,
                    num_robots=None, battery=100, geometry_seed=42,
                    placement_seed=42, robot_placement="random", spawn_interval=2,
                    task_placement="random", min_task_distance=None, **geometry):
    if robot_placement not in ("random", "corner") or task_placement not in ("random", "distant"):
        raise ValueError("Unknown placement mode")
    if type(spawn_interval) is not int or spawn_interval < 1:
        raise ValueError("spawn_interval must be a positive integer")
    if min_task_distance is not None and (type(min_task_distance) is not int or min_task_distance < 1):
        raise ValueError("min_task_distance must be a positive integer")
    warehouse = generate_warehouse(n=n, m=m, topology=topology,
                                   seed=geometry_seed, **geometry)
    count = len(warehouse.graph)
    if num_robots is None:
        num_robots = max(1, math.floor(0.2 * math.sqrt(count)))
    for name, value in (("num_tasks", num_tasks), ("num_robots", num_robots),
                        ("battery", battery)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if num_robots > count:
        raise ValueError("More robots than walkable starting positions")
    if num_tasks and count < 2:
        raise ValueError("Tasks require at least two walkable vertices")
    rng = random.Random(placement_seed)
    if robot_placement == "corner":
        starts = sorted((v for v, (row, _) in enumerate(warehouse.cells) if row == 0),
                        key=lambda v: warehouse.cells[v][1], reverse=True)
        if num_robots > len(starts) or warehouse.cells[starts[0]] != (0, m - 1):
            raise ValueError("Not enough walkable top-row starting positions")
        robots = [dict(robot_id=i + 1, start_vertex=v, battery=battery,
                       release_time=i * spawn_interval, entry_vertex=starts[0])
                  for i, v in enumerate(starts[:num_robots])]
    else:
        robots = [dict(robot_id=i + 1, start_vertex=v, battery=battery)
                  for i, v in enumerate(rng.sample(range(count), num_robots))]
    tasks = []
    if task_placement == "distant":
        minimum = min_task_distance if min_task_distance is not None else math.ceil(0.75 * (n + m - 2))
        tasks = _distant_tasks(warehouse, num_tasks, placement_seed, minimum)
    else:
        minimum = min_task_distance
        for task_id in range(num_tasks):
            start, finish = rng.sample(range(count), 2)
            tasks.append(dict(task_id=task_id, start_vertex=start, finish_vertex=finish))
    instance = dict(
        schema_version=1, num_vertices=count,
        edges=[[u, v] for u, neighbors in warehouse.graph.items()
               for v, _ in neighbors if u < v],
        robots=robots, tasks=tasks, cells=warehouse.cells, coords=warehouse.coords,
        zones=warehouse.zones, passages=warehouse.passages,
        config=dict(geometry=warehouse.config,
                    load=dict(num_robots=num_robots, num_tasks=num_tasks, battery=battery),
                    geometry_seed=geometry_seed, placement_seed=placement_seed),
    )
    if (robot_placement != "random" or task_placement != "random"
            or spawn_interval != 2 or min_task_distance is not None):
        instance["config"]["placement"] = dict(robot_placement=robot_placement,
            spawn_interval=spawn_interval, task_placement=task_placement, min_task_distance=minimum)
    return json.loads(json.dumps(instance))


def _distant_tasks(warehouse, count, seed, minimum):
    """Alternate opposite corners; finite candidate sets, checked by exact BFS."""
    if count and minimum > len(warehouse.graph) - 1:
        raise ValueError("Requested task distance exceeds every possible graph path")
    n, m = warehouse.config["n"], warehouse.config["m"]
    margin_r, margin_c = (n - 1) // 8, (m - 1) // 8
    corners = ((0, 0), (n - 1, m - 1), (0, m - 1), (n - 1, 0))
    pools = [[v for v, (r, c) in enumerate(warehouse.cells)
              if abs(r - row) <= margin_r and abs(c - col) <= margin_c]
             for row, col in corners]
    directions = ((0, 1), (1, 0), (2, 3), (3, 2))
    rng, targets, tasks = random.Random(seed), {}, []

    def candidates(source, destination):
        key = source, destination
        if key not in targets:
            distance, queue = {source: 0}, deque([source])
            while queue:
                vertex = queue.popleft()
                for other, _ in warehouse.graph[vertex]:
                    if other not in distance:
                        distance[other] = distance[vertex] + 1
                        queue.append(other)
            targets[key] = [v for v in pools[destination]
                            if v != source and distance.get(v, -1) >= minimum]
        return targets[key]

    for task_id in range(count):
        for shift in range(4):
            origin, destination = directions[(task_id + shift) % 4]
            for source in rng.sample(pools[origin], len(pools[origin])):
                choices = candidates(source, destination)
                if choices:
                    tasks.append(dict(task_id=task_id, start_vertex=source,
                                      finish_vertex=rng.choice(choices)))
                    break
            else:
                continue
            break
        else:
            raise ValueError("No opposite-corner task pair meets min_task_distance")
    return tasks


def fingerprint(instance):
    raw = json.dumps(instance, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def save_instance(instance, path):
    checksum = fingerprint(instance)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(dict(sha256=checksum, instance=instance), stream,
                  ensure_ascii=False, indent=2)
    return checksum


def load_instance(path):
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    if fingerprint(saved["instance"]) != saved["sha256"]:
        raise ValueError("Instance checksum mismatch")
    return saved["instance"]


def solver_inputs(instance):
    from test_run import RobotSpec, TaskSpec

    return (instance["num_vertices"], [tuple(edge) for edge in instance["edges"]],
            [RobotSpec(**robot) for robot in instance["robots"]],
            [TaskSpec(**task) for task in instance["tasks"]])
