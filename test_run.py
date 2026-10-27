"""
test_run.py

Двухуровневый решатель диспетчеризации роботов на графе (MAPD с передачей грузов):
1. Верхний уровень (CP-SAT): распределение задач, выбор эстафет (handoff) и временных окон.
2. Нижний уровень (CBS + Space-Time A*): потактовое построение бесконфликтных путей
   с синхронизацией встреч, контролем расхода батареи и вместимости вершин (<= 2).

Соответствует контракту паспорта проекта и интерфейсам failure.py и visualize.py.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model
from generator import generate_graph, generate_tasks


@dataclass(frozen=True)
class RobotSpec:
    robot_id: int
    start_vertex: int
    battery: int


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    start_vertex: int
    finish_vertex: int


@dataclass
class ScheduleResult:
    success: bool
    makespan: int
    # Потактовый план: timestep t -> список (robot_id, next_vertex, carried_task_id, battery_left)
    ticks: Dict[int, List[Tuple[int, int, Optional[int], int]]]
    handoff_events: List[str]
    solver_status: str


def compute_all_pairs_distances(
    num_vertices: int, adj: Dict[int, List[int]]
) -> List[List[int]]:
    """BFS для вычисления матрицы кратчайших расстояний между всеми вершинами."""
    INF = 10**9
    dist = [[INF] * num_vertices for _ in range(num_vertices)]
    for s in range(num_vertices):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            d = dist[s][u]
            for v in adj[u]:
                if v != u and dist[s][v] == INF:
                    dist[s][v] = d + 1
                    q.append(v)
    return dist


class SpaceTimeAStar:
    """Низкоуровневый планировщик пути одного робота в пространстве-времени."""

    @staticmethod
    def plan(
        robot_id: int,
        start_v: int,
        start_t: int,
        start_batt: int,
        goal_v: int,
        min_goal_t: int,
        max_horizon: int,
        adj: Dict[int, List[int]],
        dist_matrix: List[List[int]],
        constraints: Set[Tuple[int, int, int]],  # (robot_id, v, t)
        reservation: Dict[Tuple[int, int], int],  # (v, t) -> robot count
        max_capacity: int,
        blocked_vertices: Dict[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Возвращает путь как список пар (vertex, battery_left) для t от start_t до t_arrival.
        """
        counter = 0
        h0 = max(dist_matrix[start_v][goal_v], max(0, min_goal_t - start_t))
        pq = [(start_t + h0, start_t, counter, start_v, start_batt, [(start_v, start_batt)])]
        visited: Dict[Tuple[int, int], int] = {}

        while pq:
            f, t, _, u, batt, path = heapq.heappop(pq)

            if u == goal_v and t >= min_goal_t:
                return path

            if visited.get((u, t), -1) >= batt:
                continue
            visited[(u, t)] = batt

            if t >= max_horizon:
                continue

            for v in adj[u]:
                next_t = t + 1
                is_move = u != v
                next_batt = batt - 1 if is_move else batt
                if next_batt < 0:
                    continue

                if (robot_id, v, next_t) in constraints:
                    continue

                cap_limit = max_capacity - blocked_vertices.get(v, 0)
                if reservation.get((v, next_t), 0) >= cap_limit:
                    continue

                counter += 1
                h = max(dist_matrix[v][goal_v], max(0, min_goal_t - next_t))
                next_f = next_t + h
                heapq.heappush(
                    pq,
                    (
                        next_f,
                        next_t,
                        counter,
                        v,
                        next_batt,
                        path + [(v, next_batt)],
                    ),
                )

        return None


class GraphHandoffCpSatSolver:
    """
    Двухуровневый решатель:
    1. CP-SAT выбирает распределение задач, точки эстафет и временные рамки.
    2. CBS + Space-Time A* находит потактовые траектории роботов.
    """

    def __init__(
        self,
        num_vertices: int,
        edges: List[Tuple[int, int]],
        max_robots_per_vertex: int = 2,
    ) -> None:
        self.vertices = list(range(num_vertices))
        self.num_vertices = num_vertices
        self.max_robots_per_vertex = max_robots_per_vertex

        self.adj: Dict[int, List[int]] = {v: [v] for v in self.vertices}
        for u, v in edges:
            if v not in self.adj[u]:
                self.adj[u].append(v)
            if u not in self.adj[v]:
                self.adj[v].append(u)

        self.dist = compute_all_pairs_distances(self.num_vertices, self.adj)

    def solve(
        self,
        robots: List[RobotSpec],
        tasks: List[TaskSpec],
        max_horizon: int = 30,
        allow_floor_drop: bool = False,
        time_limit_s: float = 30.0,
        blocked_vertices: Optional[Dict[int, int]] = None,
        allow_undelivered: bool = False,
    ) -> ScheduleResult:
        start_time = time.time()
        blocked = blocked_vertices or {}
        R = len(robots)
        K = len(tasks)

        if K == 0 or R == 0:
            return ScheduleResult(
                success=True,
                makespan=0,
                ticks={},
                handoff_events=[],
                solver_status="OPTIMAL",
            )

        # ---------------------------------------------------------------------
        # ЭТАП 1: Верхний уровень (CP-SAT) — планирование эстафеты и задач
        # ---------------------------------------------------------------------
        high_level_plan = self._solve_high_level(
            robots,
            tasks,
            max_horizon,
            blocked,
            allow_undelivered,
            time_limit_s=min(10.0, time_limit_s * 0.4),
        )

        if not high_level_plan["success"]:
            return ScheduleResult(
                success=False,
                makespan=0,
                ticks={},
                handoff_events=[],
                solver_status=high_level_plan["status"],
            )

        # ---------------------------------------------------------------------
        # ЭТАП 2: Нижний уровень (CBS + Space-Time A*) — потактовые траектории
        # ---------------------------------------------------------------------
        remaining_time = max(1.0, time_limit_s - (time.time() - start_time))
        low_level_result = self._solve_low_level_cbs(
            robots,
            tasks,
            high_level_plan,
            max_horizon,
            blocked,
            time_limit_s=remaining_time,
        )

        return low_level_result

    def _solve_high_level(
        self,
        robots: List[RobotSpec],
        tasks: List[TaskSpec],
        max_horizon: int,
        blocked: Dict[int, int],
        allow_undelivered: bool,
        time_limit_s: float,
    ) -> dict:
        """Решает задачу назначения, выбора узлов передач и интервалов (строго линейно)."""
        model = cp_model.CpModel()
        R = len(robots)
        K = len(tasks)
        INF = 10**9

        # Кандидаты в узлы передачи: точки на кратчайших путях
        candidates: Dict[int, List[int]] = {}
        for k, task in enumerate(tasks):
            A, B = task.start_vertex, task.finish_vertex
            d_AB = self.dist[A][B]
            cands = []
            if d_AB < INF:
                for v in self.vertices:
                    if v != A and v != B and blocked.get(v, 0) == 0:
                        detour = self.dist[A][v] + self.dist[v][B]
                        if detour <= d_AB + 2:
                            cands.append((detour, v))
                if not cands and d_AB > 2:
                    for v in self.vertices:
                        if v != A and v != B and blocked.get(v, 0) == 0:
                            detour = self.dist[A][v] + self.dist[v][B]
                            if detour <= d_AB + 4:
                                cands.append((detour, v))
            cands.sort()
            candidates[k] = [v for _, v in cands[:15]]

        delivered = [model.NewBoolVar(f"deliv_{k}") for k in range(K)]

        # direct[k, r] = 1: робот r везёт груз k от A до B напрямую
        direct = {
            (k, r): model.NewBoolVar(f"dir_{k}_{r}") for k in range(K) for r in range(R)
        }

        # relay_h[k, h] = 1: задача k передаётся в узле h
        relay_h = {
            (k, h): model.NewBoolVar(f"rel_h_{k}_{h}")
            for k in range(K)
            for h in candidates[k]
        }

        # leg1[k, r, h] = 1: робот r везёт груз k от A до h
        leg1 = {
            (k, r, h): model.NewBoolVar(f"l1_{k}_{r}_{h}")
            for k in range(K)
            for r in range(R)
            for h in candidates[k]
        }

        # leg2[k, r, h] = 1: робот r везёт груз k от h до B
        leg2 = {
            (k, r, h): model.NewBoolVar(f"l2_{k}_{r}_{h}")
            for k in range(K)
            for r in range(R)
            for h in candidates[k]
        }

        for k in range(K):
            if not allow_undelivered:
                model.Add(delivered[k] == 1)

            modes = [direct[k, r] for r in range(R)] + [
                relay_h[k, h] for h in candidates[k]
            ]
            if modes:
                model.Add(sum(modes) == delivered[k])
            else:
                model.Add(delivered[k] == 0)

            for h in candidates[k]:
                model.Add(sum(leg1[k, r, h] for r in range(R)) == relay_h[k, h])
                model.Add(sum(leg2[k, r, h] for r in range(R)) == relay_h[k, h])
                for r in range(R):
                    model.Add(leg1[k, r, h] + leg2[k, r, h] <= relay_h[k, h])

        # Ограничение по батарее (строго линейно: BoolVar * int)
        for r, r_spec in enumerate(robots):
            energy_terms = []
            for k, task in enumerate(tasks):
                A, B = task.start_vertex, task.finish_vertex
                if self.dist[A][B] < INF:
                    energy_terms.append(direct[k, r] * self.dist[A][B])
                for h in candidates[k]:
                    if self.dist[A][h] < INF:
                        energy_terms.append(leg1[k, r, h] * self.dist[A][h])
                    if self.dist[h][B] < INF:
                        energy_terms.append(leg2[k, r, h] * self.dist[h][B])

            model.Add(sum(energy_terms) <= r_spec.battery)

        makespan = model.NewIntVar(0, max_horizon, "makespan")
        num_delivered = sum(delivered)
        model.Maximize(num_delivered * 10000 - makespan * 100)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {"success": False, "status": solver.StatusName(status)}

        plan_tasks = []
        for k, task in enumerate(tasks):
            if solver.BooleanValue(delivered[k]):
                is_dir = False
                for r in range(R):
                    if solver.BooleanValue(direct[k, r]):
                        plan_tasks.append(
                            {
                                "task_id": task.task_id,
                                "k": k,
                                "start": task.start_vertex,
                                "finish": task.finish_vertex,
                                "relay": False,
                                "r1": r,
                                "r2": None,
                                "handoff_v": None,
                            }
                        )
                        is_dir = True
                        break

                if not is_dir:
                    for h in candidates[k]:
                        if solver.BooleanValue(relay_h[k, h]):
                            r1 = next(
                                r
                                for r in range(R)
                                if solver.BooleanValue(leg1[k, r, h])
                            )
                            r2 = next(
                                r
                                for r in range(R)
                                if solver.BooleanValue(leg2[k, r, h])
                            )
                            plan_tasks.append(
                                {
                                    "task_id": task.task_id,
                                    "k": k,
                                    "start": task.start_vertex,
                                    "finish": task.finish_vertex,
                                    "relay": True,
                                    "r1": r1,
                                    "r2": r2,
                                    "handoff_v": h,
                                }
                            )
                            break

        return {
            "success": True,
            "tasks": plan_tasks,
            "status": solver.StatusName(status),
        }

    def _solve_low_level_cbs(
        self,
        robots: List[RobotSpec],
        tasks: List[TaskSpec],
        high_level: dict,
        max_horizon: int,
        blocked: Dict[int, int],
        time_limit_s: float,
    ) -> ScheduleResult:
        start_time = time.time()
        R = len(robots)
        scheduled_tasks = high_level["tasks"]

        robot_legs: Dict[int, List[dict]] = {r: [] for r in range(R)}
        for item in scheduled_tasks:
            if not item["relay"]:
                robot_legs[item["r1"]].append(
                    {
                        "type": "direct",
                        "task_id": item["task_id"],
                        "pickup": item["start"],
                        "dropoff": item["finish"],
                        "partner": None,
                    }
                )
            else:
                robot_legs[item["r1"]].append(
                    {
                        "type": "handoff_drop",
                        "task_id": item["task_id"],
                        "pickup": item["start"],
                        "dropoff": item["handoff_v"],
                        "partner": item["r2"],
                    }
                )
                robot_legs[item["r2"]].append(
                    {
                        "type": "handoff_pickup",
                        "task_id": item["task_id"],
                        "pickup": item["handoff_v"],
                        "dropoff": item["finish"],
                        "partner": item["r1"],
                    }
                )

        constraints: Set[Tuple[int, int, int]] = set()
        cbs_queue = [(0, 0, constraints)]
        node_counter = 0

        while cbs_queue and (time.time() - start_time) < time_limit_s:
            _, _, cur_constraints = heapq.heappop(cbs_queue)

            plan_res = self._plan_all_robots(
                robots, robot_legs, cur_constraints, max_horizon, blocked
            )
            if plan_res is None:
                continue

            paths, task_held = plan_res
            conflict = self._find_vertex_conflict(paths, blocked)
            if conflict is None:
                return self._build_schedule_result(
                    robots, tasks, paths, task_held, "OPTIMAL"
                )

            v_conf, t_conf, conf_robots = conflict
            for r_conf in conf_robots:
                new_constr = set(cur_constraints)
                new_constr.add((r_conf, v_conf, t_conf))
                node_counter += 1
                cost = max(len(p) for p in paths.values())
                heapq.heappush(cbs_queue, (cost, node_counter, new_constr))

        return ScheduleResult(
            success=False,
            makespan=0,
            ticks={},
            handoff_events=[],
            solver_status="TIMEOUT",
        )

    def _plan_all_robots(
        self,
        robots: List[RobotSpec],
        robot_legs: Dict[int, List[dict]],
        constraints: Set[Tuple[int, int, int]],
        max_horizon: int,
        blocked: Dict[int, int],
    ) -> Optional[Tuple[Dict[int, List[int]], Dict[int, List[Optional[int]]]]]:
        R = len(robots)
        paths: Dict[int, List[int]] = {r: [robots[r].start_vertex] for r in range(R)}
        batt: Dict[int, List[int]] = {r: [robots[r].battery] for r in range(R)}
        task_held: Dict[int, List[Optional[int]]] = {r: [None] for r in range(R)}
        reservation: Dict[Tuple[int, int], int] = {
            (robots[r].start_vertex, 0): 1 for r in range(R)
        }

        legs_queue = {r: list(robot_legs[r]) for r in range(R)}
        current_leg = {r: legs_queue[r].pop(0) if legs_queue[r] else None for r in range(R)}
        handoff_arrival: Dict[Tuple[int, int], int] = {}

        deadlock_guard = 0
        while any(current_leg.values()) and deadlock_guard < max_horizon * 2:
            deadlock_guard += 1
            progress = False

            for r in range(R):
                leg = current_leg[r]
                if leg is None:
                    continue

                curr_t = len(paths[r]) - 1
                curr_v = paths[r][-1]
                curr_b = batt[r][-1]

                if leg["type"] == "handoff_pickup":
                    key = (leg["task_id"], leg["pickup"])
                    if key not in handoff_arrival:
                        if curr_t < max_horizon:
                            self._append_wait(r, paths, batt, task_held, reservation)
                            progress = True
                        continue

                    meet_t = handoff_arrival[key]
                    if curr_t < meet_t or curr_v != leg["pickup"]:
                        p = SpaceTimeAStar.plan(
                            r,
                            curr_v,
                            curr_t,
                            curr_b,
                            leg["pickup"],
                            meet_t,
                            max_horizon,
                            self.adj,
                            self.dist,
                            constraints,
                            reservation,
                            self.max_robots_per_vertex,
                            blocked,
                        )
                        if p is None:
                            return None
                        self._apply_subpath(r, p, None, paths, batt, task_held, reservation)
                        curr_t = len(paths[r]) - 1
                        curr_v = paths[r][-1]
                        curr_b = batt[r][-1]

                    p_drop = SpaceTimeAStar.plan(
                        r,
                        curr_v,
                        curr_t,
                        curr_b,
                        leg["dropoff"],
                        curr_t + 1,
                        max_horizon,
                        self.adj,
                        self.dist,
                        constraints,
                        reservation,
                        self.max_robots_per_vertex,
                        blocked,
                    )
                    if p_drop is None:
                        return None
                    self._apply_subpath(
                        r, p_drop, leg["task_id"], paths, batt, task_held, reservation
                    )
                    current_leg[r] = legs_queue[r].pop(0) if legs_queue[r] else None
                    progress = True
                    continue

                if curr_v != leg["pickup"]:
                    p_pick = SpaceTimeAStar.plan(
                        r,
                        curr_v,
                        curr_t,
                        curr_b,
                        leg["pickup"],
                        curr_t,
                        max_horizon,
                        self.adj,
                        self.dist,
                        constraints,
                        reservation,
                        self.max_robots_per_vertex,
                        blocked,
                    )
                    if p_pick is None:
                        return None
                    self._apply_subpath(r, p_pick, None, paths, batt, task_held, reservation)
                    curr_t = len(paths[r]) - 1
                    curr_v = paths[r][-1]
                    curr_b = batt[r][-1]

                p_carry = SpaceTimeAStar.plan(
                    r,
                    curr_v,
                    curr_t,
                    curr_b,
                    leg["dropoff"],
                    curr_t + 1,
                    max_horizon,
                    self.adj,
                    self.dist,
                    constraints,
                    reservation,
                    self.max_robots_per_vertex,
                    blocked,
                )
                if p_carry is None:
                    return None
                self._apply_subpath(
                    r, p_carry, leg["task_id"], paths, batt, task_held, reservation
                )

                if leg["type"] == "handoff_drop":
                    handoff_arrival[(leg["task_id"], leg["dropoff"])] = len(paths[r]) - 1

                current_leg[r] = legs_queue[r].pop(0) if legs_queue[r] else None
                progress = True

            if not progress:
                min_len = min(len(paths[r]) for r in range(R))
                for r in range(R):
                    if len(paths[r]) == min_len and len(paths[r]) <= max_horizon:
                        self._append_wait(r, paths, batt, task_held, reservation)

        max_T = max(len(paths[r]) for r in range(R))
        for r in range(R):
            while len(paths[r]) < max_T:
                self._append_wait(r, paths, batt, task_held, reservation)

        return paths, task_held

    def _append_wait(self, r, paths, batt, task_held, reservation):
        v = paths[r][-1]
        b = batt[r][-1]
        t = len(paths[r])
        paths[r].append(v)
        batt[r].append(b)
        task_held[r].append(None)
        reservation[(v, t)] = reservation.get((v, t), 0) + 1

    def _apply_subpath(self, r, subpath, carried_task, paths, batt, task_held, reservation):
        for v, b in subpath[1:]:
            t = len(paths[r])
            paths[r].append(v)
            batt[r].append(b)
            task_held[r].append(carried_task)
            reservation[(v, t)] = reservation.get((v, t), 0) + 1

    def _find_vertex_conflict(
        self, paths: Dict[int, List[int]], blocked: Dict[int, int]
    ) -> Optional[Tuple[int, int, List[int]]]:
        T = max(len(p) for p in paths.values())
        for t in range(T):
            occ: Dict[int, List[int]] = {}
            for r, p in paths.items():
                if t < len(p):
                    occ.setdefault(p[t], []).append(r)
            for v, rob_list in occ.items():
                limit = self.max_robots_per_vertex - blocked.get(v, 0)
                if len(rob_list) > limit:
                    return v, t, rob_list
        return None

    def _build_schedule_result(
        self,
        robots: List[RobotSpec],
        tasks: List[TaskSpec],
        paths: Dict[int, List[int]],
        task_held: Dict[int, List[Optional[int]]],
        solver_status: str,
    ) -> ScheduleResult:
        R = len(robots)
        T = max(len(paths[r]) for r in range(R)) - 1
        ticks: Dict[int, List[Tuple[int, int, Optional[int], int]]] = {}
        battery_left = {r: robots[r].battery for r in range(R)}
        handoff_events: List[str] = []

        prev_carrier: Dict[int, Optional[int]] = {task.task_id: None for task in tasks}

        for t in range(1, T + 1):
            row = []
            for r in range(R):
                u_curr = paths[r][t - 1]
                v_next = paths[r][t]
                if u_curr != v_next:
                    battery_left[r] -= 1
                task_id = task_held[r][t] if t < len(task_held[r]) else None
                row.append((robots[r].robot_id, v_next, task_id, battery_left[r]))

                if task_id is not None:
                    if (
                        prev_carrier[task_id] is not None
                        and prev_carrier[task_id] != robots[r].robot_id
                    ):
                        v_h = paths[r][t - 1]
                        handoff_events.append(
                            f"Такт {t-1}: Груз #{task_id} передан от Робота #{prev_carrier[task_id]} "
                            f"-> Роботу #{robots[r].robot_id} в вершине {v_h}"
                        )
                    prev_carrier[task_id] = robots[r].robot_id

            ticks[t] = row

        return ScheduleResult(
            success=True,
            makespan=T,
            ticks=ticks,
            handoff_events=handoff_events,
            solver_status=solver_status,
        )


def print_passport_output(result: ScheduleResult) -> None:
    """Форматированный вывод согласно контракту из Паспорта проекта."""
    if not result.success:
        print(f"Решение не найдено (статус: {result.solver_status})")
        return

    print(f"Время требуемое для решения T = {result.makespan}")
    if result.handoff_events:
        print("\nЗафиксированные передачи/обмены грузов:")
        for ev in result.handoff_events:
            print("  *", ev)

    print("\nПотактовый план переходов (№ робота; вершина перехода):")
    for t, rows in result.ticks.items():
        print(f"--- Такт {t} ---")
        for r_id, v_next, task_id, batt in rows:
            cargo_info = f" [несёт груз #{task_id}]" if task_id is not None else ""
            print(f"{r_id}; {v_next}   (заряд: {batt}){cargo_info}")


def print_output(result: ScheduleResult) -> None:
    """Форматированный вывод: номер_робота следующая_вершина задача."""
    if not result.success:
        print(f"Решение не найдено (статус: {result.solver_status})")
        return

    print(f"{result.makespan} {len(result.ticks[1]) if result.ticks else 0}")
    for t in sorted(result.ticks.keys()):
        for r_id, v_next, task_id, _batt in result.ticks[t]:
            carried = task_id if task_id is not None else "-1"
            print(f"{r_id} {v_next} {carried}")


def prepare_data(graph: Dict, raw_tasks: List[Dict], default_battery: int = 30):
    """Адаптирует сырые данные генератора под запуск решателя."""
    num_vertices = len(graph)
    edges = []
    for u, neighbors in graph.items():
        for item in neighbors:
            v = item[0] if isinstance(item, (tuple, list)) else item
            if u < v:
                edges.append((u, v))
    edges.sort()

    tasks = [
        TaskSpec(task_id=t["id"], start_vertex=t["A"], finish_vertex=t["B"])
        for t in raw_tasks
    ]

    k_robots = math.floor(0.2 * math.sqrt(num_vertices))
    k_robots = max(1, k_robots)

    robots = [
        RobotSpec(robot_id=i + 1, start_vertex=i, battery=default_battery)
        for i in range(k_robots)
    ]

    return num_vertices, edges, tasks, robots


if __name__ == "__main__":
    print("введите n, m:")
    n, m = map(int, input().split())
    print("введите число задач:")
    num_tasks = int(input())

    raw_graph = generate_graph(n=n, m=m, padding=3, remove_prob=0.2, seed=42)
    raw_tasks = generate_tasks(raw_graph, num_tasks=num_tasks, seed=42)

    num_vertices, edges, tasks, robots = prepare_data(
        raw_graph, raw_tasks, default_battery=num_tasks * n * 2
    )

    solver = GraphHandoffCpSatSolver(
        num_vertices=num_vertices,
        edges=edges,
        max_robots_per_vertex=2,
    )

    res = solver.solve(
        robots=robots,
        tasks=tasks,
        max_horizon=num_tasks * n * 2,
        time_limit_s=60,
    )
    print_output(res)