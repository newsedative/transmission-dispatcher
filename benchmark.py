"""Run both existing planners on saved, identical warehouse instances."""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import networkx
import ortools
from pathlib import Path
import sys
import time
from unittest.mock import patch
import uuid

from instances import create_instance, fingerprint, load_instance, save_instance, solver_inputs
from planners import create_solver
from test_run import cp_model
from validation import verify_result
from visualize import export_html, scenario


TOPOLOGIES = ("long_aisles", "cross_aisles", "zoned")
SOURCE_FILES = ("warehouse.py", "instances.py", "benchmark.py", "validation.py", "test_run.py",
                "greedy.py", "planners.py", "coordinated_routes.py", "visualize.py", "viz_ops_template.html")
METRICS = ("delivered", "last_delivery_tick", "utilization", "movement_utilization",
           "handoffs", "total_moves", "busy_robot_ticks", "moving_robot_ticks",
           "robot_ticks_denominator")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def matrix_cases(seeds=(42, 43)):
    profiles = (dict(profile="small", n=8, m=12, num_tasks=3, num_robots=2,
                     battery=500, max_horizon=256),
                dict(profile="medium", n=16, m=20, num_tasks=8, num_robots=4,
                     battery=2000, max_horizon=2048))
    return [dict(profile, topology=topology, geometry_seed=seed, placement_seed=seed)
            for topology in TOPOLOGIES for seed in seeds for profile in profiles]


def run_pair(instance, output, max_horizon=1024, time_limit_s=60, planner_seed=0):
    if max_horizon < 0 or time_limit_s <= 0:
        raise ValueError("Invalid horizon or time limit")
    output = Path(output)
    output.mkdir(parents=True)
    expected = fingerprint(instance)
    save_instance(instance, output / "input.json")
    input_bytes = (output / "input.json").read_bytes()
    source_root = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                     for name in SOURCE_FILES}
    rows, views = [], []

    for planner in ("greedy", "cp-sat"):
        print(f"{output.name}: {planner}", flush=True)
        destination = output / planner
        destination.mkdir()
        row = dict(case=output.name, planner=planner, max_horizon=max_horizon,
                   occupancy_rule="one_robot_except_actual_transfer",
                   time_limit_s=time_limit_s, planner_seed=planner_seed,
                   input_checksum=expected, input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                   phase="setup", status=None, high_level_status=None,
                   source_sha256=source_hashes, runtime=dict(python=sys.version, ortools=ortools.__version__,
                                                          networkx=networkx.__version__),
                   valid=False, error=None, full_route_optimality_proven=False,
                   **{key: None for key in METRICS})
        config = instance.get("config", {})
        geometry = config.get("geometry", config)
        row.update(topology=geometry.get("topology"), n=geometry.get("n"), m=geometry.get("m"),
                   geometry_seed=config.get("geometry_seed", geometry.get("seed")),
                   placement_seed=config.get("placement_seed"), vertices=instance["num_vertices"],
                   edges=len(instance["edges"]), robots=len(instance["robots"]), tasks=len(instance["tasks"]))
        timings, sat, phase = {}, {}, {}
        row.update(timings_seconds=timings, cp_sat=sat)
        write_json(destination / "metrics.json", row)
        before = time.perf_counter()
        vertices, edges, robots, tasks = solver_inputs(instance)
        solver = create_solver(vertices, edges, planner=planner, seed=planner_seed)
        timings["setup"] = time.perf_counter() - before

        def timed(name, method):
            def call(*args, **kwargs):
                row["phase"] = name
                before = time.perf_counter()
                if name == "assignment":
                    phase["assignment_start"] = before
                write_json(destination / "metrics.json", row)
                try:
                    value = method(*args, **kwargs)
                    if name == "assignment":
                        row["high_level_status"] = value.get("status")
                        write_json(destination / "high_plan.json", value)
                    return value
                finally:
                    timings[name] = time.perf_counter() - before
            return call

        solver._solve_high_level = timed("assignment", solver._solve_high_level)
        solver._solve_low_level_cbs = timed("routing", solver._solve_low_level_cbs)
        native_solve = cp_model.CpSolver.Solve

        def observe_sat(native, *args, **kwargs):
            before = time.perf_counter()
            timings["preprocessing_and_model"] = before - phase["assignment_start"]
            sat["parameters"] = str(native.parameters)
            sat["random_seed"] = native.parameters.random_seed
            sat["num_search_workers"] = native.parameters.num_search_workers
            status = native_solve(native, *args, **kwargs)
            timings["native_cp_sat"] = time.perf_counter() - before
            sat.update(status=native.StatusName(status), wall_time_seconds=native.WallTime(),
                       best_bound=native.BestObjectiveBound(), response_stats=native.ResponseStats(),
                       bound_scope="pruned high-level model")
            sat["objective"] = native.ObjectiveValue() if status in (cp_model.FEASIBLE, cp_model.OPTIMAL) else None
            return status

        before = time.perf_counter()
        result = None
        try:
            with patch.object(cp_model.CpSolver, "Solve", observe_sat):
                result = solver.solve(robots, tasks, max_horizon=max_horizon, time_limit_s=time_limit_s)
            timings["solve"] = time.perf_counter() - before
            row["reported_route_status"] = result.solver_status
            row["routing_method"] = getattr(solver, "last_routing_method", "cbs")
            row["status"] = "FEASIBLE" if result.success else result.solver_status
            write_json(destination / "result.json", asdict(result))
            if result.success:
                validation_start = time.perf_counter()
                if result.makespan > max_horizon:
                    raise ValueError("Route exceeds the requested horizon")
                checked = verify_result(instance, result)
                row.update({key: checked[key] for key in METRICS})
                row["valid"] = True
                row["verification"] = checked
                timings["validation"] = time.perf_counter() - validation_start
                view = scenario(vertices, instance["edges"], robots, tasks,
                                replace(result, solver_status="FEASIBLE"),
                                f"{output.name}: {planner}", coords=instance.get("coords"))
                view["warehouse"] = {**{key: geometry.get(key) for key in ("n", "m", "topology")},
                                     **{key: instance.get(key) for key in ("cells", "zones", "passages")}}
                views.append(view)
        except Exception as error:
            timings.setdefault("solve", time.perf_counter() - before)
            row.update(status="INVALID" if result and result.success else "ERROR",
                       valid=False, error=f"{type(error).__name__}: {error}")
            row.update({key: None for key in METRICS})
        row["phase"] = "complete"
        row["inputs_preserved"] = fingerprint(instance) == expected and (output / "input.json").read_bytes() == input_bytes
        row["source_files_preserved"] = all(hashlib.sha256((source_root / name).read_bytes()).hexdigest() == digest
                                             for name, digest in source_hashes.items())
        write_json(destination / "metrics.json", row)
        if not row["inputs_preserved"]:
            raise RuntimeError("Planner changed the shared input")
        rows.append(row)
        print(f"  {row['status']}: delivered={row['delivered']}, last={row['last_delivery_tick']}", flush=True)

    if views:
        export_html(views, str(output / "dispatch.html"))
    write_summary(rows, output)
    return rows


def write_summary(rows, output):
    write_json(output / "summary.json", dict(
        metric_definition="Mean moving-or-carrying robots / all robots, ticks 1..last physical delivery",
        time_unit="simulation tick", rows=rows))
    fields = ("case", "topology", "geometry_seed", "placement_seed", "n", "m", "vertices", "edges",
              "robots", "tasks", "planner", "max_horizon", "occupancy_rule", "routing_method", "status", "high_level_status", "valid", "delivered",
              "last_delivery_tick", "utilization", "movement_utilization", "handoffs", "solve_seconds",
              "preprocessing_and_model_seconds", "native_cp_sat_seconds", "cp_objective", "cp_best_bound", "error")
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row.get(key) for key in fields},
                             "solve_seconds": row["timings_seconds"].get("solve"),
                             "preprocessing_and_model_seconds": row["timings_seconds"].get("preprocessing_and_model"),
                             "native_cp_sat_seconds": row["timings_seconds"].get("native_cp_sat"),
                             "cp_objective": row["cp_sat"].get("objective"),
                             "cp_best_bound": row["cp_sat"].get("best_bound")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matrix", action="store_true", help="3 topologies x 2 seeds x 2 loads x 2 planners")
    parser.add_argument("--instance", type=Path, help="Replay one saved instance")
    parser.add_argument("--config", type=Path, help="JSON object of warehouse geometry parameters")
    parser.add_argument("--topologies", nargs="+", choices=TOPOLOGIES)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--placement-seed", type=int)
    parser.add_argument("--planner-seed", type=int, default=0)
    parser.add_argument("--robot-placement", choices=("random", "corner"))
    parser.add_argument("--spawn-interval", type=int)
    parser.add_argument("--task-placement", choices=("random", "distant"))
    parser.add_argument("--min-task-distance", type=int)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--battery", type=int, default=1000)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--time-limit", type=float)
    args = parser.parse_args(argv)
    if args.matrix and args.instance:
        parser.error("--matrix and --instance cannot be combined")
    output = args.output or Path("runs") / (datetime.now().strftime("warehouse_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    write_json(output / "configuration.json", {key: str(value) if isinstance(value, Path) else value
                                               for key, value in vars(args).items()})
    geometry = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(geometry, dict):
        parser.error("Geometry configuration must be a JSON object")
    configured_topology = geometry.pop("topology", None)
    base_seed = geometry.pop("geometry_seed", geometry.pop("seed", 42))
    rows = []
    time_limit = args.time_limit if args.time_limit is not None else (6 if args.matrix else 60)
    if args.instance:
        rows = run_pair(load_instance(args.instance), output / "replay",
                        args.horizon if args.horizon is not None else 1024, time_limit, args.planner_seed)
    else:
        seeds = args.seeds or ([base_seed, base_seed + 1] if args.matrix else [base_seed])
        topologies = args.topologies or ([configured_topology] if configured_topology else TOPOLOGIES)
        cases = matrix_cases(seeds) if args.matrix else [
            dict(topology=topology, geometry_seed=seed, placement_seed=seed, n=args.n, m=args.m,
                 num_tasks=args.tasks, num_robots=args.robots, battery=args.battery,
                 max_horizon=args.horizon if args.horizon is not None else 1024, profile="custom")
            for topology in topologies for seed in seeds]
        for case in cases:
            profile, horizon = case.pop("profile"), case.pop("max_horizon")
            if args.placement_seed is not None:
                case["placement_seed"] = args.placement_seed
            placement = {name: getattr(args, name) for name in
                         ("robot_placement", "spawn_interval", "task_placement", "min_task_distance")
                         if getattr(args, name) is not None}
            parameters = {**case, **geometry, **placement}
            name = f"{parameters['topology']}_seed{parameters['geometry_seed']}_{profile}"
            instance = create_instance(**parameters)
            rows.extend(run_pair(instance, output / name, args.horizon if args.horizon is not None else horizon,
                                 time_limit, args.planner_seed))
            write_summary(rows, output)
    write_summary(rows, output)
    print(f"Saved {len(rows)} runs: {output / 'summary.json'}", flush=True)
    return 0 if all(row["valid"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
