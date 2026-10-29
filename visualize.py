"""Интерактивный HTML-пульт для плана из test_run.GraphHandoffCpSatSolver.

    from generator import generate_graph, generate_tasks
    from test_run import prepare_data, GraphHandoffCpSatSolver
    from visualize import scenario, export_html

    n, m = 5, 6
    graph = generate_graph(n=n, m=m, padding=1, remove_prob=0.2, seed=42)
    raw_tasks = generate_tasks(graph, num_tasks=4, seed=42)
    V, edges, tasks, robots = prepare_data(graph, raw_tasks, default_battery=20)
    res = GraphHandoffCpSatSolver(V, edges).solve(robots, tasks, max_horizon=16)
    export_html([scenario(V, edges, robots, tasks, res, "Мой склад", coords=grid_coords(n, m))],
                "dispatch.html")

С отказом робота (см. failure.py) — тот же план, розыгрыш отказа и перестройка:

    run = run_with_failure(V, edges, robots, tasks, original=res, p_fail=0.06, seed=4, max_horizon=16)
    export_html([failure_scenario(V, edges, robots, tasks, run, "Мой склад, отказ",
                                  coords=grid_coords(n, m), p_fail=0.06)], "dispatch.html")

Открыть получившийся файл в браузере. Несколько сценариев в одном файле выбираются списком «Инстанс».
`python3 visualize.py` собирает dispatch.html для параметрического склада long_aisles.
Топология и размеры задаются аргументами командной строки.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from generator import generate_graph, generate_tasks
from test_run import GraphHandoffCpSatSolver, RobotSpec, ScheduleResult, TaskSpec, prepare_data
from failure import FailureRun, run_with_failure
from planners import create_solver
from instances import create_instance, solver_inputs
from validation import _check_occupancy, verify_result

TEMPLATE = Path(__file__).with_name("viz_ops_template.html")


def grid_coords(n: int, m: int) -> List[List[float]]:
    """Координаты вершин сетки из generator.py: вершина i*m + j стоит в столбце j, строке i."""
    return [[round((v % m) / max(1, m - 1), 5), round((v // m) / max(1, n - 1), 5)] for v in range(n * m)]


def auto_coords(num_vertices: int, edges: Sequence[Tuple[int, int]]) -> List[List[float]]:
    """Раскладка произвольного графа (нужен networkx)."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    G.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(G) if num_vertices <= 60 else nx.spring_layout(G, seed=7)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    def norm(a, lo, hi):
        return 0.5 if hi - lo < 1e-9 else (a - lo) / (hi - lo)
    return [[norm(pos[v][0], min(xs), max(xs)), norm(pos[v][1], min(ys), max(ys))] for v in range(num_vertices)]


def plan_from_ticks(robots: List[RobotSpec], tasks: List[TaskSpec], res: ScheduleResult):
    """ticks решателя -> pos[r][t] (вершина в момент t), hold[r][t] (индекс груза на шаге t -> t+1)."""
    T = res.makespan
    k_of = {t.task_id: k for k, t in enumerate(tasks)}
    pos = [[(r.start_vertex if getattr(r, "entry_vertex", None) is None else r.entry_vertex)
            if getattr(r, "release_time", 0) == 0 else None] for r in robots]
    hold: List[List[Optional[int]]] = [[] for _ in robots]
    for t in range(1, T + 1):
        rows = {row[0]: row for row in res.ticks[t]}
        for r, spec in enumerate(robots):
            _, v_next, task_id, _ = rows[spec.robot_id]
            pos[r].append(v_next)
            hold[r].append(None if task_id is None else k_of[task_id])
    for row in hold:
        row.append(None)
    return pos, hold


def events_from_plan(tasks: List[TaskSpec], pos, hold, T: int):
    """События для лога: [t, kind, k, who, v], kind = pickup | transfer | deliver."""
    events = []
    for k, task in enumerate(tasks):
        prev = None                                    # кто нёс груз на предыдущем шаге
        last = None                                    # последний носильщик
        for t in range(T):
            cur = next((r for r in range(len(pos)) if hold[r][t] == k), None)
            if cur is not None and cur != prev:
                v = pos[cur][t]
                if last is None or last == cur:          # первый подъём или снова поднял свой груз
                    events.append([t, "pickup", k, cur, v])
                else:                                    # груз перешёл к другому роботу
                    events.append([t, "transfer", k, [last, cur], v])
            if cur is None and prev is not None:
                v = pos[prev][t]
                if v == task.finish_vertex:
                    events.append([t, "deliver", k, prev, v])
            if cur is not None:
                last = cur
            prev = cur
        if prev is not None and pos[prev][T] == task.finish_vertex:
            events.append([T, "deliver", k, prev, pos[prev][T]])
    # повторная «сдача», если груз потом снова подняли с финиша, не считается доставкой
    final = {}
    for e in events:
        if e[1] == "deliver":
            final[e[2]] = e
    return [e for e in events if e[1] != "deliver" or final[e[2]] is e]


def validate(num_vertices: int, edges, robots: List[RobotSpec], tasks: List[TaskSpec], pos, hold, T: int,
             max_robots_per_vertex: int = 2) -> int:
    """Проверяет план по контракту паспорта, возвращает число доставленных грузов."""
    adj = {v: {v} for v in range(num_vertices)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for r, spec in enumerate(robots):
        release = getattr(spec, "release_time", 0)
        entry = spec.start_vertex if getattr(spec, "entry_vertex", None) is None else spec.entry_vertex
        assert type(release) is int and release >= 0, f"R{spec.robot_id}: неверное время выхода"
        assert pos[r][0] == (entry if release == 0 else None), f"R{spec.robot_id}: неверная начальная позиция"
        moves = 0
        for t in range(T):
            a, b = pos[r][t], pos[r][t + 1]
            if a is None:
                assert t + 1 <= release and b == (entry if t + 1 == release else None), f"R{spec.robot_id}: неверное появление на шаге {t}"
                assert hold[r][t] is None, f"R{spec.robot_id}: груз до появления"
            else:
                assert b in adj[a], f"R{spec.robot_id}: нет ребра {a}-{b} на шаге {t}"
                moves += a != b
        assert moves <= spec.battery, f"R{spec.robot_id}: потрачено {moves} > заряда {spec.battery}"
    delivered, transfers = 0, {}
    for k, task in enumerate(tasks):
        at, previous = task.start_vertex, None
        for t in range(T):
            carriers = [r for r in range(len(robots)) if hold[r][t] == k]
            assert len(carriers) <= 1, f"груз {task.task_id} несут несколько роботов, t={t}"
            if carriers:
                r = carriers[0]
                assert pos[r][t] == at, f"груз {task.task_id}: робот не там, где груз, t={t}"
                if previous is not None and previous != r:
                    assert pos[previous][t] == pos[r][t], f"груз {task.task_id}: передача без встречи, t={t}"
                    transfers.setdefault(t, set()).add((pos[r][t], frozenset((previous, r))))
                at = pos[r][t + 1]
            previous = carriers[0] if carriers else None
        delivered += at == task.finish_vertex
    for t in range(T + 1):
        count = _check_occupancy({r: pos[r][t] for r in range(len(robots))}, transfers.get(t, set()), t)
        assert count <= max_robots_per_vertex, f"больше {max_robots_per_vertex} роботов в вершине, t={t}"
    return delivered


def scenario(num_vertices: int, edges, robots: List[RobotSpec], tasks: List[TaskSpec], res: ScheduleResult,
             name: str, description: str = "", coords=None, labels=None) -> dict:
    """Упаковывает инстанс и найденный план для пульта (план предварительно проверяется)."""
    assert res.success, f"решение не найдено: {res.solver_status}"
    T = res.makespan
    pos, hold = plan_from_ticks(robots, tasks, res)
    delivered = validate(num_vertices, edges, robots, tasks, pos, hold, T)
    optimal = res.solver_status == "OPTIMAL"
    if not description:
        description = (f"{num_vertices} вершин, {len(robots)} роб., {len(tasks)} задач. "
                       f"Доставлено {delivered} из {len(tasks)} за {T} тактов"
                       + (" (оптимум)." if optimal else f" (статус {res.solver_status})."))
    return {
        "name": name,
        "description": description,
        "n": num_vertices,
        "edges": [[u, v, 1] for u, v in edges],        # в паспорте ребро = 1 такт = 1 ед. заряда
        "robots": [[r.start_vertex, r.battery] for r in robots],
        "robot_ids": [r.robot_id for r in robots],
        "release_times": [getattr(r, "release_time", 0) for r in robots],
        "entry_vertices": [getattr(r, "entry_vertex", None) for r in robots],
        "occupancy_rule": "one_robot_except_actual_transfer",
        "orders": [[t.start_vertex, t.finish_vertex] for t in tasks],
        "order_ids": [t.task_id for t in tasks],
        "coords": coords or auto_coords(num_vertices, edges),
        "labels": labels,
        "show_weights": False,
        "T": max(1, T),
        "pos": pos if T else [p * 2 for p in pos],
        "hold": hold if T else [[None, None] for _ in hold],
        "events": events_from_plan(tasks, pos, hold, T),
        "delivered": delivered,
        "makespan": T,
        "optimal": optimal,
        "upper_bound": T,
        "handoffs": res.handoff_events,
    }


def cargo_track(tasks: List[TaskSpec], pos, hold, T: int) -> List[int]:
    """Вершина каждого груза в момент T."""
    at = [task.start_vertex for task in tasks]
    for k in range(len(tasks)):
        for t in range(T):
            r = next((r for r in range(len(pos)) if hold[r][t] == k), None)
            if r is not None:
                at[k] = pos[r][t + 1]
    return at


def failure_scenario(num_vertices: int, edges, robots: List[RobotSpec], tasks: List[TaskSpec], run: FailureRun,
                     name: str, description: str = "", coords=None, labels=None,
                     p_fail: Optional[float] = None) -> dict:
    """Сценарий с отказом робота: итоговый план (до отказа исходный, после — перестроенный)
    плюс исходный план, чтобы пульт мог показать, что было отменено."""
    sc = scenario(num_vertices, edges, robots, tasks, run.result, name, description or " ", coords, labels)
    f = run.failure
    if f is None:
        sc["failure"] = None
        if not description:
            sc["description"] = (f"{num_vertices} вершин, {len(robots)} роб., {len(tasks)} задач. "
                                 f"Отказ не выпал (p = {p_fail}), план выполнен за {sc['makespan']} тактов.")
        return sc
    r_f = next(i for i, r in enumerate(robots) if r.robot_id == f.robot_id)
    k_of = {t.task_id: k for k, t in enumerate(tasks)}
    T0 = run.original.makespan
    opos, ohold = plan_from_ticks(robots, tasks, run.original)
    odone: List[Optional[int]] = [None] * len(tasks)
    for e in events_from_plan(tasks, opos, ohold, T0):
        if e[1] == "deliver":
            odone[e[2]] = e[0]
    final = cargo_track(tasks, sc["pos"], sc["hold"], sc["makespan"])
    undelivered = [k for k, task in enumerate(tasks) if final[k] != task.finish_vertex]
    sc["failure"] = {
        "r": r_f, "t": f.tick, "v": f.vertex,
        "task": None if f.carried_task is None else k_of[f.carried_task],
        "p": p_fail,
        "orig_T": T0, "orig_optimal": run.original.solver_status == "OPTIMAL",
        "orig_pos": opos, "orig_done": odone,
        "replanned": [k_of[t.task_id] for t in run.replan_tasks],
        "undelivered": undelivered,
        "alive": len(robots) - 1,
    }
    if not description:
        rid = f"R-{f.robot_id:02d}"
        cargo = f" с грузом #{f.carried_task} на борту" if f.carried_task is not None else ""
        sc["description"] = (f"{num_vertices} вершин, {len(robots)} роб., {len(tasks)} задач. "
                             f"На такте {f.tick} отказал {rid} (вершина {f.vertex}){cargo}; план перестроен "
                             f"для {len(robots) - 1} роб.: {T0} → {sc['makespan']} тактов, доставлено "
                             f"{sc['delivered']} из {len(tasks)}.")
    return sc


def export_html(scenarios: List[dict], path: str) -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(scenarios, ensure_ascii=False, separators=(",", ":"))
    Path(path).write_text(html.replace("/*__SCENARIOS__*/[]", data), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- готовые сценарии
def passport_reference(time_limit_s: float = 30.0, planner: str = "cp-sat",
                       planner_seed: int = 0) -> dict:
    """Малый эталон из паспорта: сетка 3×3 без центра, 3 робота, 2 встречные задачи."""
    #   0 1 2
    #   3 . 4
    #   5 6 7
    edges = [(0, 1), (1, 2), (0, 3), (2, 4), (3, 5), (4, 7), (5, 6), (6, 7)]
    tasks = [TaskSpec(1, 5, 2), TaskSpec(2, 2, 5)]
    robots = [RobotSpec(1, 5, 4), RobotSpec(2, 2, 2), RobotSpec(3, 0, 2)]
    res = create_solver(8, edges, planner=planner, seed=planner_seed).solve(
        robots, tasks, max_horizon=12, time_limit_s=time_limit_s)
    coords = [[0, 0], [0.5, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [0.5, 1], [1, 1]]
    return scenario(8, edges, robots, tasks, res, "Малый эталон из паспорта", coords=coords,
                    description="Кольцо 3×3 без центра. Груз #1 из левого нижнего угла в правый верхний, "
                                "груз #2 обратно; у R-02 и R-03 заряда только на 2 хода, поэтому грузы "
                                "передаются по эстафете.")


def generated_warehouse(n: int, m: int, num_tasks: int, seed: int = 42,
                        battery: int = 20, num_robots: Optional[int] = None,
                        max_horizon: int = 20, time_limit_s: float = 60.0, name: str = "",
                        failures: Sequence[dict] = (), planner: str = "cp-sat",
                        planner_seed: int = 0, topology: str = "long_aisles",
                        placement_seed: Optional[int] = None, geometry=None) -> List[dict]:
    """Параметрический склад; размещение воспроизводится независимо от геометрии."""
    instance = create_instance(n=n, m=m, topology=topology, num_tasks=num_tasks,
                               num_robots=num_robots, battery=battery, geometry_seed=seed,
                               placement_seed=seed if placement_seed is None else placement_seed,
                               **(geometry or {}))
    return instance_scenarios(instance, max_horizon, time_limit_s, name, failures,
                              planner, planner_seed)


def instance_scenarios(instance, max_horizon=1024, time_limit_s=60, name="", failures=(),
                       planner="cp-sat", planner_seed=0):
    V, edges, robots, tasks = solver_inputs(instance)
    res = create_solver(V, edges, planner=planner, seed=planner_seed).solve(
        robots, tasks, max_horizon=max_horizon, time_limit_s=time_limit_s)
    verify_result(instance, res)
    if res.makespan > max_horizon:
        raise ValueError("Result exceeds the planning horizon")
    config = instance["config"]["geometry"]
    name = name or f"{config['topology']}: {config['n']}×{config['m']}, {len(robots)} роб., {len(tasks)} задач"
    out = [scenario(V, edges, robots, tasks, replace(res, solver_status="FEASIBLE"),
                    name, coords=instance["coords"])]
    for fl in failures:
        run = run_with_failure(V, edges, robots, tasks, original=res, max_horizon=max_horizon,
                               planner=planner, planner_seed=planner_seed, **fl)
        run = replace(run, original=replace(run.original, solver_status="FEASIBLE"),
                      result=replace(run.result, solver_status="FEASIBLE"))
        tag = (f"отказ R-{run.failure.robot_id:02d} на такте {run.failure.tick}" if run.failure
               else "отказ не выпал")
        out.append(failure_scenario(V, edges, robots, tasks, run, f"{name} · {tag}",
                                    coords=instance["coords"], p_fail=fl.get("p_fail")))
    for view in out:
        view["warehouse"] = dict(n=config["n"], m=config["m"], topology=config["topology"],
                                  cells=instance["cells"], zones=instance["zones"],
                                  passages=instance["passages"])
    return out


def passport_failure(forced=(1, 2), planner: str = "cp-sat", planner_seed: int = 0) -> dict:
    """Малый эталон с отказом: у R-02 и R-03 по 2 ед. заряда, так что резерва почти нет."""
    edges = [(0, 1), (1, 2), (0, 3), (2, 4), (3, 5), (4, 7), (5, 6), (6, 7)]
    tasks = [TaskSpec(1, 5, 2), TaskSpec(2, 2, 5)]
    robots = [RobotSpec(1, 5, 4), RobotSpec(2, 2, 2), RobotSpec(3, 0, 2)]
    run = run_with_failure(8, edges, robots, tasks, max_horizon=12, forced=forced,
                           planner=planner, planner_seed=planner_seed)
    coords = [[0, 0], [0.5, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [0.5, 1], [1, 1]]
    return failure_scenario(8, edges, robots, tasks, run,
                            f"Малый эталон · отказ R-{forced[0]:02d} на такте {forced[1]}", coords=coords)


def _try(build, *args, **kwargs) -> List[dict]:
    """Собирает сценарий(и); если решатель не справился, пишет предупреждение и пропускает."""
    try:
        out = build(*args, **kwargs)
    except (AssertionError, RuntimeError, ValueError) as e:
        print(f"пропущен {build.__name__}{args}: {e}")
        return []
    return out if isinstance(out, list) else [out]


if __name__ == "__main__":
    import argparse
    from instances import add_geometry_arguments, geometry_options, load_instance

    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="dispatch.html")
    parser.add_argument("--planner", choices=("cp-sat", "greedy"), default="cp-sat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument("--robots", type=int)
    parser.add_argument("--battery", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--instance")
    add_geometry_arguments(parser)
    args = parser.parse_args()
    if args.instance:
        instance = load_instance(args.instance)
        scs = instance_scenarios(instance, args.horizon, args.time_limit,
                                 planner=args.planner, planner_seed=args.seed)
    else:
        geometry = geometry_options(args)
        topology = geometry.pop("topology")
        scs = generated_warehouse(args.n, args.m, args.tasks, seed=args.geometry_seed,
                                  battery=args.battery, num_robots=args.robots,
                                  max_horizon=args.horizon, time_limit_s=args.time_limit,
                                  topology=topology, geometry=geometry,
                                  placement_seed=args.placement_seed,
                                  planner=args.planner, planner_seed=args.seed)
    for s in scs:
        f = s.get("failure")
        extra = (f", отказ R{s['robot_ids'][f['r']]} t={f['t']} v={f['v']}, план {f['orig_T']}→{s['makespan']}"
                 if f else "")
        print(f"{s['name']}: T={s['makespan']}, доставлено {s['delivered']}/{len(s['orders'])}, "
              f"оптимум={s['optimal']}, событий {len(s['events'])}{extra}")
    print(export_html(scs, args.output))
