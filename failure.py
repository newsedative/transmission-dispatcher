"""Отказ робота во время выполнения плана и перестройка диспетчеризации.

Модель отказа: на каждом такте t каждый робот, который по плану должен сейчас ехать, с вероятностью
p_fail ломается (первый сломавшийся — единственный отказ за прогон). Сломанный робот навсегда остаётся
в своей вершине и продолжает занимать в ней одно из двух мест. Груз, который он вёз, остаётся у него
на борту, и его может забрать любой исправный робот, подъехав в эту вершину (передача логическая).

В момент отказа диспетчер берёт текущее состояние (вершины роботов, остаток заряда, где лежит каждый
груз) и заново запускает GraphHandoffCpSatSolver для оставшихся роботов и недоставленных грузов.
Если оставшимся роботам не хватает заряда на всё, решатель довозит максимум грузов.

    from failure import run_with_failure
    run = run_with_failure(V, edges, robots, tasks, p_fail=0.05, seed=1, max_horizon=30)
    run.failure   # кто, когда, где сломался (None, если отказа не случилось)
    run.result    # итоговый склеенный план: до отказа — исходный, после — перестроенный

`python3 failure.py` — пример на складе из генератора, вывод в формате контракта.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from test_run import GraphHandoffCpSatSolver, RobotSpec, ScheduleResult, TaskSpec
from planners import create_solver


@dataclass(frozen=True)
class FailureEvent:
    robot_id: int
    tick: int                      # момент t: робот стоит в вершине vertex и больше не двигается
    vertex: int
    carried_task: Optional[int]    # груз, оставшийся у него на борту


@dataclass
class FailureRun:
    original: ScheduleResult                  # план без отказа
    failure: Optional[FailureEvent]
    replan: Optional[ScheduleResult]          # план оставшихся роботов с момента отказа (такты с 1)
    replan_tasks: List[TaskSpec]              # что перепланировали: старт = где груз лежал в момент отказа
    result: ScheduleResult                    # итоговый план, такты 1..makespan


def plan_state(robots: List[RobotSpec], tasks: List[TaskSpec], res: ScheduleResult, t: int):
    """Состояние по плану в момент t: вершины роботов, остаток заряда, вершина каждого груза,
    кто вёз груз на шаге t-1 -> t."""
    pos = {r.robot_id: (None if getattr(r, "release_time", 0) > 0 else
                        r.start_vertex if getattr(r, "entry_vertex", None) is None else r.entry_vertex)
           for r in robots}
    battery = {r.robot_id: r.battery for r in robots}
    cargo = {task.task_id: task.start_vertex for task in tasks}
    carrier: Dict[int, Optional[int]] = {task.task_id: None for task in tasks}
    for step in range(1, t + 1):
        for robot_id, v_next, task_id, batt in res.ticks[step]:
            pos[robot_id] = v_next
            battery[robot_id] = batt
            if task_id is not None:
                cargo[task_id] = v_next
        carrier = {task.task_id: None for task in tasks}
        for robot_id, _, task_id, _ in res.ticks[step]:
            if task_id is not None:
                carrier[task_id] = robot_id
    return pos, battery, cargo, carrier


def _failure_at(robots: List[RobotSpec], tasks: List[TaskSpec], res: ScheduleResult, robot_id: int,
                t: int) -> FailureEvent:
    if not 0 <= t <= res.makespan:
        raise ValueError("Такт отказа вне плана")
    pos, _, cargo, carrier = plan_state(robots, tasks, res, t)
    if pos.get(robot_id) is None:
        raise ValueError("Отказ возможен только после выхода робота")
    finish = {task.task_id: task.finish_vertex for task in tasks}
    # груз, который робот как раз довёз до финиша, считается сданным
    held = next((k for k, c in carrier.items() if c == robot_id and cargo[k] != finish[k]), None)
    return FailureEvent(robot_id, t, pos[robot_id], held)


def sample_failure(robots: List[RobotSpec], tasks: List[TaskSpec], res: ScheduleResult, p_fail: float,
                   rng: random.Random, only_moving: bool = True,
                   guarantee: bool = False) -> Optional[FailureEvent]:
    """Разыгрывает отказ по плану res: на такте t (1 <= t < T) каждый робот, который едет на шаге
    t -> t+1 (или любой, если only_moving=False), ломается с вероятностью p_fail.

    guarantee=True — отказ обязательно случается: розыгрыш повторяется, пока кто-нибудь не сломается
    (то есть момент и робот берутся из того же распределения при условии, что отказ был)."""
    if not 0 <= p_fail <= 1:
        raise ValueError("p_fail должна быть от 0 до 1")
    for _ in range(10_000 if guarantee and p_fail > 0 else 1):
        failure = _sample_once(robots, tasks, res, p_fail, rng, only_moving)
        if failure is not None:
            return failure
    return None


def busy_failure(robots: List[RobotSpec], tasks: List[TaskSpec], res: ScheduleResult) -> Optional[Tuple[int, int]]:
    """Детерминированный «худший» отказ для демонстрации: первый робот, который начиная с трети плана
    везёт недоставленный груз. Возвращает (robot_id, t) для run_with_failure(forced=...)."""
    for t in range(max(1, res.makespan // 3), res.makespan):
        for r in robots:
            if getattr(r, "release_time", 0) > t:
                continue
            f = _failure_at(robots, tasks, res, r.robot_id, t)
            if f.carried_task is not None:
                return r.robot_id, t
    return None


def _sample_once(robots, tasks, res, p_fail, rng, only_moving):
    for t in range(1, res.makespan):
        pos, _, _, carrier = plan_state(robots, tasks, res, t)
        nxt = {row[0]: row[1] for row in res.ticks[t + 1]}
        ids = [r.robot_id for r in robots]
        rng.shuffle(ids)
        for robot_id in ids:
            if pos[robot_id] is None:
                continue
            if only_moving and nxt[robot_id] == pos[robot_id]:
                continue
            if rng.random() < p_fail:
                return _failure_at(robots, tasks, res, robot_id, t)
    return None


def replan_after_failure(solver: GraphHandoffCpSatSolver, robots: List[RobotSpec], tasks: List[TaskSpec],
                         res: ScheduleResult, failure: FailureEvent, max_horizon: int,
                         time_limit_s: float = 30.0) -> Tuple[ScheduleResult, List[TaskSpec], List[RobotSpec]]:
    """Перестраивает план с момента отказа для исправных роботов и недоставленных грузов."""
    pos, battery, cargo, _ = plan_state(robots, tasks, res, failure.tick)
    if pos.get(failure.robot_id) is None:
        raise ValueError("Отказ возможен только после выхода робота")
    alive = []
    for r in robots:
        if r.robot_id == failure.robot_id:
            continue
        if pos[r.robot_id] is None:
            alive.append(RobotSpec(r.robot_id, r.start_vertex, battery[r.robot_id],
                                   release_time=r.release_time - failure.tick, entry_vertex=r.entry_vertex))
        else:
            alive.append(RobotSpec(r.robot_id, pos[r.robot_id], battery[r.robot_id]))
    left = [TaskSpec(task.task_id, cargo[task.task_id], task.finish_vertex)
            for task in tasks if cargo[task.task_id] != task.finish_vertex]
    if not left:
        return ScheduleResult(True, 0, {}, [], res.solver_status), left, alive
    kw = dict(max_horizon=max_horizon, time_limit_s=time_limit_s, blocked_vertices={failure.vertex: 1})
    if failure.carried_task is not None:
        kw["blocked_cargo"] = {failure.vertex: failure.carried_task}
    sub = solver.solve(alive, left, **kw)
    if not sub.success:
        # всё довезти нельзя (не хватает заряда или груз заперт) — довозим сколько получится
        sub = solver.solve(alive, left, allow_undelivered=True, **kw)
    return sub, left, alive


def merge_plans(robots: List[RobotSpec], res: ScheduleResult, failure: FailureEvent,
                sub: ScheduleResult) -> ScheduleResult:
    """Такты 1..t_f — исходный план, дальше — перестроенный; сломанный робот стоит на месте."""
    t_f = failure.tick
    ticks = {t: list(res.ticks[t]) for t in range(1, t_f + 1)}
    frozen_battery = next(r.battery for r in robots if r.robot_id == failure.robot_id)
    if t_f:
        frozen_battery = next(row[3] for row in res.ticks[t_f] if row[0] == failure.robot_id)
    order = {r.robot_id: i for i, r in enumerate(robots)}
    frozen_cargo, recovery = failure.carried_task, None
    previous_pos = plan_state(robots, [], res, t_f)[0]
    for t in range(1, sub.makespan + 1):
        rows = list(sub.ticks[t])
        receiver = next((rid for rid, _, cargo, _ in rows if cargo == frozen_cargo), None) if frozen_cargo is not None else None
        if receiver is not None:
            if previous_pos[receiver] != failure.vertex:
                raise ValueError("Передача груза отказавшего робота без встречи")
            recovery = (t_f + t - 1, frozen_cargo, receiver)
            frozen_cargo = None
        previous_pos.update({rid: vertex for rid, vertex, _, _ in rows})
        rows.append((failure.robot_id, failure.vertex, frozen_cargo, frozen_battery))
        ticks[t_f + t] = sorted(rows, key=lambda row: order[row[0]])
    shift = lambda s: re.sub(r"^Такт (\d+)", lambda m: f"Такт {int(m.group(1)) + t_f}", s)
    events = [e for e in res.handoff_events if int(re.match(r"Такт (\d+)", e).group(1)) < t_f]
    events.append(f"Такт {t_f}: ОТКАЗ Робота #{failure.robot_id} в вершине {failure.vertex}"
                  + (f", на борту груз #{failure.carried_task}" if failure.carried_task is not None else "")
                  + " — план перестроен")
    events += [shift(e) for e in sub.handoff_events]
    if recovery is not None:
        tick, cargo, receiver = recovery
        events.append(f"Такт {tick}: Груз #{cargo} передан от Робота #{failure.robot_id} "
                      f"-> Роботу #{receiver} в вершине {failure.vertex}")
        events.sort(key=lambda event: int(re.match(r"Такт (\d+)", event).group(1)))
    return ScheduleResult(True, t_f + sub.makespan, ticks, events, sub.solver_status)


def run_with_failure(num_vertices: int, edges: Sequence[Tuple[int, int]], robots: List[RobotSpec],
                     tasks: List[TaskSpec], p_fail: float = 0.05, seed: int = 0, max_horizon: int = 30,
                     time_limit_s: float = 60.0, replan_time_limit_s: float = 30.0,
                     original: Optional[ScheduleResult] = None,
                     forced: Optional[Tuple[int, int]] = None, guarantee: bool = False,
                     planner: str = "cp-sat", planner_seed: int = 0) -> FailureRun:
    """Строит план, разыгрывает отказ и, если он случился, перестраивает план.

    original — уже найденный план без отказа (чтобы не решать заново);
    forced = (robot_id, t) — отказ задан вручную вместо розыгрыша, forced="busy" — см. busy_failure;
    guarantee=True — розыгрыш, при котором отказ обязательно случается.
    """
    solver = create_solver(num_vertices, list(edges), planner=planner, seed=planner_seed)
    if original is None:
        original = solver.solve(robots, tasks, max_horizon=max_horizon, time_limit_s=time_limit_s)
    if not original.success:
        raise RuntimeError(f"исходный план не найден: {original.solver_status}")
    if forced == "busy":
        forced = busy_failure(robots, tasks, original)
    if forced is not None:
        failure = _failure_at(robots, tasks, original, *forced)
    else:
        failure = sample_failure(robots, tasks, original, p_fail, random.Random(seed), guarantee=guarantee)
    if failure is None:
        return FailureRun(original, None, None, [], original)
    sub, left, _ = replan_after_failure(solver, robots, tasks, original, failure,
                                        max_horizon=max_horizon, time_limit_s=replan_time_limit_s)
    if not sub.success:
        raise RuntimeError(f"перестроенный план не найден: {sub.solver_status}")
    return FailureRun(original, failure, sub, left, merge_plans(robots, original, failure, sub))


if __name__ == "__main__":
    from generator import generate_graph, generate_tasks
    from test_run import prepare_data, print_output

    n, m, num_tasks, p_fail, seed = 6, 7, 4, 0.08, 1
    graph = generate_graph(n=n, m=m, padding=1, remove_prob=0.2, seed=3)
    raw_tasks = generate_tasks(graph, num_tasks=num_tasks, seed=3)
    V, edges, tasks, _ = prepare_data(graph, raw_tasks)
    robots = [RobotSpec(i + 1, i, 20) for i in range(4)]
    run = run_with_failure(V, edges, robots, tasks, p_fail=p_fail, seed=seed, max_horizon=24,
                           guarantee=True)
    print(f"план без отказа: T = {run.original.makespan}")
    if run.failure is None:
        print("отказа не случилось")
    else:
        f = run.failure
        print(f"отказ: робот #{f.robot_id}, такт {f.tick}, вершина {f.vertex}, груз на борту: {f.carried_task}")
        print(f"перестроенный план: T = {run.result.makespan} ({run.replan.solver_status})")
        for e in run.result.handoff_events:
            print("  *", e)
    print_output(run.result)
