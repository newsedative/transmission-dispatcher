"""
test_run.py

Двухуровневый решатель диспетчеризации роботов на графе (MAPD с передачей грузов):
1. Верхний уровень (CP-SAT): балансировка нагрузки (VRP/дизъюнктивное расписание),
   выбор эстафет (handoff), расчет холостых переходов и минимизация реального makespan.
2. Нижний уровень (CBS + Space-Time A*): синхронизированное потактовое планирование
   траекторий в хронологическом порядке, контроль вместимости вершин (<= 2) и батареи.

Полностью совместим с failure.py и visualize.py.
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
        max_horizon: int = 40,
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
        # 1. Верхний уровень (CP-SAT): распределение, транзиты, makespan
        # ---------------------------------------------------------------------
        high_level_plan = self._solve_high_level(
            robots,
            tasks,
            max_horizon,
            blocked,
            allow_undelivered,
            time_limit_s=min(12.0, time_limit_s * 0.5),
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
        # 2. Нижний уровень (CBS + Space-Time A*): синхронизация в lock-step
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
        """
        Дизъюнктивная модель расписания:
        - Связывает makespan со временем реального окончания задач каждым роботом
        - Учитывает холостые переходы между задачами
        """
        model = cp_model.CpModel()
        R = len(robots)
        K = len(tasks)
        INF = 10**9

        candidates: Dict[int, List[int]] = {}
        for k, task in enumerate(tasks):
            A, B = task.start_vertex, task.finish_vertex
            d_AB = self.dist[A][B]
            cands = []
            if d_AB < INF and d_AB > 2:
                for v in self.vertices:
                    if v != A and v != B and blocked.get(v, 0) == 0:
                        detour = self.dist[A][v] + self.dist[v][B]
                        if detour <= d_AB + 2:
                            cands.append((detour, v))
                cands.sort()
                candidates[k] = [v for _, v in cands[:8]]
            else:
                candidates[k] = []

        delivered = [model.NewBoolVar(f"deliv_{k}") for k in range(K)]
        direct = {
            (k, r): model.NewBoolVar(f"dir_{k}_{r}") for k in range(K) for r in range(R)
        }
        relay_h = {
            (k, h): model.NewBoolVar(f"rel_h_{k}_{h}")
            for k in range(K)
            for h in candidates[k]
        }
        leg1 = {
            (k, r, h): model.NewBoolVar(f"l1_{k}_{r}_{h}")
            for k in range(K)
            for r in range(R)
            for h in candidates[k]
        }
        leg2 = {
            (k, r, h): model.NewBoolVar(f"l2_{k}_{r}_{h}")
            for k in range(K)
            for r in range(R)
            for h in candidates[k]
        }

        # Временные интервалы выполнения задач
        t_start = [model.NewIntVar(0, max_horizon, f"ts_{k}") for k in range(K)]
        t_dur = [model.NewIntVar(0, max_horizon, f"td_{k}") for k in range(K)]
        t_end = [model.NewIntVar(0, max_horizon, f"te_{k}") for k in range(K)]

        for k, task in enumerate(tasks):
            A, B = task.start_vertex, task.finish_vertex
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

            dur_terms = []
            if self.dist[A][B] < INF:
                dur_terms.append(sum(direct[k, r] for r in range(R)) * self.dist[A][B])
            for h in candidates[k]:
                d_split = self.dist[A][h] + self.dist[h][B]
                dur_terms.append(relay_h[k, h] * d_split)

            model.Add(t_dur[k] == sum(dur_terms))
            model.Add(t_end[k] == t_start[k] + t_dur[k])

            # Время выезда со стартовой позиции робота к первой задаче
            for r, r_spec in enumerate(robots):
                if self.dist[r_spec.start_vertex][A] < INF:
                    model.Add(
                        t_start[k] >= self.dist[r_spec.start_vertex][A]
                    ).OnlyEnforceIf(direct[k, r])
                    for h in candidates[k]:
                        model.Add(
                            t_start[k] >= self.dist[r_spec.start_vertex][A]
                        ).OnlyEnforceIf(leg1[k, r, h])

        # Дизъюнкции: последовательность задач на каждом роботе с учётом перехода
        for r in range(R):
            for j in range(K):
                for k in range(j + 1, K):
                    task_j = tasks[j]
                    task_k = tasks[k]

                    # Участвует ли робот r в задаче j и задаче k?
                    in_j = [direct[j, r]] + [leg1[j, r, h] for h in candidates[j]] + [
                        leg2[j, r, h] for h in candidates[j]
                    ]
                    in_k = [direct[k, r]] + [leg1[k, r, h] for h in candidates[k]] + [
                        leg2[k, r, h] for h in candidates[k]
                    ]

                    r_in_j = model.NewBoolVar(f"r{r}_in_{j}")
                    r_in_k = model.NewBoolVar(f"r{r}_in_{k}")
                    model.Add(sum(in_j) == r_in_j)
                    model.Add(sum(in_k) == r_in_k)

                    both = model.NewBoolVar(f"both_r{r}_{j}_{k}")
                    model.AddBoolAnd([r_in_j, r_in_k]).OnlyEnforceIf(both)
                    model.AddBoolOr([r_in_j.Not(), r_in_k.Not()]).OnlyEnforceIf(both.Not())

                    j_before_k = model.NewBoolVar(f"ord_r{r}_{j}_{k}")

                    # Оценка дистанции переезда от сдачи j до забора k
                    d_jk = self.dist[task_j.finish_vertex][task_k.start_vertex]
                    d_kj = self.dist[task_k.finish_vertex][task_j.start_vertex]
                    if d_jk >= INF:
                        d_jk = 0
                    if d_kj >= INF:
                        d_kj = 0

                    model.Add(t_start[k] >= t_end[j] + d_jk).OnlyEnforceIf(
                        [both, j_before_k]
                    )
                    model.Add(t_start[j] >= t_end[k] + d_kj).OnlyEnforceIf(
                        [both, j_before_k.Not()]
                    )

        # Ограничения батареи
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

        # РЕАЛЬНЫЙ makespan: верхняя граница окончания всех доставленных задач
        makespan = model.NewIntVar(0, max_horizon, "makespan")
        for k in range(K):
            model.Add(makespan >= t_end[k]).OnlyEnforceIf(delivered[k])

        num_delivered = sum(delivered)
        # Приоритет: 1) Максимум доставленных, 2) Минимум makespan (балансировка!), 3) Экономия пути
        model.Minimize(
            (K - num_delivered) * 1_000_000 + makespan * 1000 + sum(t_dur)
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {"success": False, "status": solver.StatusName(status)}

        plan_tasks = []
        for k, task in enumerate(tasks):
            if solver.BooleanValue(delivered[k]):
                start_t_val = solver.Value(t_start[k])
                end_t_val = solver.Value(t_end[k])

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
                                "t_start": start_t_val,
                                "t_end": end_t_val,
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
                            t_mid = start_t_val + self.dist[task.start_vertex][h]
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
                                    "t_start": start_t_val,
                                    "t_mid": t_mid,
                                    "t_end": end_t_val,
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

        # Формируем этапы
        all_legs = []
        for item in scheduled_tasks:
            if not item["relay"]:
                all_legs.append(
                    {
                        "task_id": item["task_id"],
                        "carrier": item["r1"],
                        "pickup": item["start"],
                        "dropoff": item["finish"],
                        "type": "direct",
                        "partner": None,
                        "t_est_start": item["t_start"],
                        "t_est_end": item["t_end"],
                    }
                )
            else:
                all_legs.append(
                    {
                        "task_id": item["task_id"],
                        "carrier": item["r1"],
                        "pickup": item["start"],
                        "dropoff": item["handoff_v"],
                        "type": "handoff_drop",
                        "partner": item["r2"],
                        "t_est_start": item["t_start"],
                        "t_est_end": item["t_mid"],
                    }
                )
                all_legs.append(
                    {
                        "task_id": item["task_id"],
                        "carrier": item["r2"],
                        "pickup": item["handoff_v"],
                        "dropoff": item["finish"],
                        "type": "handoff_pickup",
                        "partner": item["r1"],
                        "t_est_start": item["t_mid"],
                        "t_est_end": item["t_end"],
                    }
                )

        # Сортируем этапы строго по времени начала (хронологический порядок)
        all_legs.sort(key=lambda leg: leg["t_est_start"])

        constraints: Set[Tuple[int, int, int]] = set()
        cbs_queue = [(0, 0, constraints)]
        node_counter = 0

        while cbs_queue and (time.time() - start_time) < time_limit_s:
            _, _, cur_constraints = heapq.heappop(cbs_queue)

            plan_res = self._plan_all_robots_chronological(
                robots, all_legs, cur_constraints, max_horizon, blocked
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

    def _plan_all_robots_chronological(
        self,
        robots: List[RobotSpec],
        all_legs: List[dict],
        constraints: Set[Tuple[int, int, int]],
        max_horizon: int,
        blocked: Dict[int, int],
    ) -> Optional[Tuple[Dict[int, List[int]], Dict[int, List[Optional[int]]]]]:
        """Синхронное пошаговое выполнение этапов с rendezvous в точках эстафеты."""
        R = len(robots)
        paths: Dict[int, List[int]] = {r: [robots[r].start_vertex] for r in range(R)}
        batt: Dict[int, List[int]] = {r: [robots[r].battery] for r in range(R)}
        task_held: Dict[int, List[Optional[int]]] = {r: [None] for r in range(R)}
        reservation: Dict[Tuple[int, int], int] = {
            (robots[r].start_vertex, 0): 1 for r in range(R)
        }

        handoff_events_arrival: Dict[Tuple[int, int], int] = {}

        for leg in all_legs:
            r = leg["carrier"]
            curr_t = len(paths[r]) - 1
            curr_v = paths[r][-1]
            curr_b = batt[r][-1]

            if leg["type"] == "handoff_pickup":
                key = (leg["task_id"], leg["pickup"])
                t_partner_arr = handoff_events_arrival.get(key, 0)

                # Путь до точки встречи
                p_to_meet = SpaceTimeAStar.plan(
                    r,
                    curr_v,
                    curr_t,
                    curr_b,
                    leg["pickup"],
                    t_partner_arr,
                    max_horizon,
                    self.adj,
                    self.dist,
                    constraints,
                    reservation,
                    self.max_robots_per_vertex,
                    blocked,
                )
                if p_to_meet is None:
                    return None
                self._apply_subpath(r, p_to_meet, None, paths, batt, task_held, reservation)
                curr_t = len(paths[r]) - 1
                curr_v = paths[r][-1]
                curr_b = batt[r][-1]

                # Синхронизация встречи: если приехал раньше напарника, ждёт в точке
                t_meet = max(curr_t, t_partner_arr)
                while curr_t < t_meet:
                    self._append_wait(r, paths, batt, task_held, reservation)
                    curr_t += 1

                # Доставка груза до финиша
                p_deliv = SpaceTimeAStar.plan(
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
                if p_deliv is None:
                    return None
                self._apply_subpath(
                    r, p_deliv, leg["task_id"], paths, batt, task_held, reservation
                )

            else:
                # Прямая доставка или первый этап эстафеты (handoff_drop)
                if curr_v != leg["pickup"]:
                    p_to_pick = SpaceTimeAStar.plan(
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
                    if p_to_pick is None:
                        return None
                    self._apply_subpath(
                        r, p_to_pick, None, paths, batt, task_held, reservation
                    )
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
                    handoff_events_arrival[(leg["task_id"], leg["dropoff"])] = (
                        len(paths[r]) - 1
                    )

        # Выравниваем по длине все траектории до единого глобального makespan
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
    if not result.success:
        print(f"Решение не найдено (статус: {result.solver_status})")
        return

    print(f"{result.makespan} {len(result.ticks[1]) if result.ticks else 0}")
    for t in sorted(result.ticks.keys()):
        for r_id, v_next, task_id, _batt in result.ticks[t]:
            carried = task_id if task_id is not None else "-1"
            print(f"{r_id} {v_next} {carried}")


def prepare_data(graph: Dict, raw_tasks: List[Dict], default_battery: int = 30):
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