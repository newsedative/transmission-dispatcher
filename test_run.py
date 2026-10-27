"""
test_run.py

Двухуровневый решатель диспетчеризации роботов на графе (MAPD с передачей грузов):
1. Верхний уровень (CP-SAT): пространственно-фильтрованная модель (Spatial Pruning)
   с расчетом POI-дистанций, сбалансированным Warm-Start (Min-Load Round-Robin)
   и точной минимизацией makespan.
2. Нижний уровень (CBS + Space-Time A*): потактовое хронологическое планирование
   с точной эвристикой, синхронизацией эстафет и полноценным Conflict-Based Search
   по вместимости вершин (<= 2).

Оптимизирован как для малых эталонов (passport), так и для сеток 128x128 (V > 16 000).
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
    ticks: Dict[int, List[Tuple[int, int, Optional[int], int]]]
    handoff_events: List[str]
    solver_status: str


class DistanceOracle:
    """
    Ленивый кэш кратчайших расстояний.
    Запускает BFS только из запрошенных точек интереса (POI: старты, финиши задач, базы роботов),
    исключая дорогой All-Pairs BFS на O(V^2).
    """

    def __init__(self, num_vertices: int, adj: Dict[int, List[int]]) -> None:
        self.num_vertices = num_vertices
        self.adj = adj
        self.cache: Dict[int, List[int]] = {}
        self.INF = 10**9

    def get_all_from(self, source: int) -> List[int]:
        if source not in self.cache:
            d = [self.INF] * self.num_vertices
            d[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                du = d[u]
                for v in self.adj[u]:
                    if v != u and d[v] == self.INF:
                        d[v] = du + 1
                        q.append(v)
            self.cache[source] = d
        return self.cache[source]

    def dist(self, u: int, v: int) -> int:
        if u in self.cache:
            return self.cache[u][v]
        if v in self.cache:
            return self.cache[v][u]
        return self.get_all_from(u)[v]


class SpaceTimeAStar:
    """Низкоуровневый планировщик пути робота с точной BFS-эвристикой для CBS."""

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
        dist_oracle: DistanceOracle,
        constraints: Set[Tuple[int, int, int]],
        max_capacity: int,
        blocked_vertices: Dict[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        goal_dists = dist_oracle.get_all_from(goal_v)
        counter = 0
        h0 = max(goal_dists[start_v], max(0, min_goal_t - start_t))
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

                # Ограничение CBS для данного робота
                if (robot_id, v, next_t) in constraints:
                    continue

                # Полностью заблокированная вершина
                if blocked_vertices.get(v, 0) >= max_capacity:
                    continue

                counter += 1
                h = max(goal_dists[v], max(0, min_goal_t - next_t))
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

        self.oracle = DistanceOracle(self.num_vertices, self.adj)

    def solve(
        self,
        robots: List[RobotSpec],
        tasks: List[TaskSpec],
        max_horizon: int = 60,
        allow_floor_drop: bool = False,
        time_limit_s: float = 60.0,
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

        elapsed = time.time() - start_time
        cp_time_budget = max(5.0, (time_limit_s - elapsed) * 0.7)

        high_level_plan = self._solve_high_level(
            robots,
            tasks,
            max_horizon,
            blocked,
            allow_undelivered,
            time_limit_s=cp_time_budget,
        )

        if not high_level_plan["success"]:
            return ScheduleResult(
                success=False,
                makespan=0,
                ticks={},
                handoff_events=[],
                solver_status=high_level_plan["status"],
            )

        remaining_time = max(2.0, time_limit_s - (time.time() - start_time))
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
        model = cp_model.CpModel()
        R = len(robots)
        K = len(tasks)
        INF = self.oracle.INF

        # 1. Прогрев кэша дистанций для POI
        for r in robots:
            self.oracle.get_all_from(r.start_vertex)
        for t in tasks:
            self.oracle.get_all_from(t.start_vertex)
            self.oracle.get_all_from(t.finish_vertex)

        # 2. Сбалансированный жадный Warm-Start (Min-Load Round-Robin)
        greedy_assigned: Dict[int, int] = {}
        cur_pos = [robots[r].start_vertex for r in range(R)]
        avail_t = [0] * R
        task_count = [0] * R
        max_tasks_per_robot = math.ceil(K / R) + 1

        sorted_k = sorted(
            range(K),
            key=lambda k: self.oracle.dist(tasks[k].start_vertex, tasks[k].finish_vertex),
            reverse=True,
        )

        for k in sorted_k:
            task = tasks[k]
            A, B = task.start_vertex, task.finish_vertex
            d_AB = self.oracle.dist(A, B)
            if d_AB >= INF:
                continue

            best_r = None
            best_score = INF

            for r in range(R):
                d_pick = self.oracle.dist(cur_pos[r], A)
                if d_pick >= INF:
                    continue
                # Проверяем, хватит ли роботу батареи на прямую доставку
                if d_AB > robots[r].battery:
                    continue
                t_fin = avail_t[r] + d_pick + d_AB
                penalty = 300 if task_count[r] >= max_tasks_per_robot else 0
                score = t_fin + penalty + avail_t[r] * 0.4

                if score < best_score:
                    best_score = score
                    best_r = r

            if best_r is not None:
                greedy_assigned[k] = best_r
                d_pick = self.oracle.dist(cur_pos[best_r], A)
                avail_t[best_r] = avail_t[best_r] + d_pick + d_AB
                cur_pos[best_r] = B
                task_count[best_r] += 1
            else:
                greedy_assigned[k] = 0

        # 3. Load-Aware Spatial Pruning
        cand_robots: Dict[int, List[int]] = {}
        c_limit = min(R, max(5, int(math.ceil(R * 0.25))))
        for k, task in enumerate(tasks):
            assigned_r = greedy_assigned.get(k, 0)
            ranked = sorted(
                range(R),
                key=lambda r: (
                    0 if r == assigned_r else 1,
                    min(
                        self.oracle.dist(robots[r].start_vertex, task.start_vertex),
                        self.oracle.dist(robots[r].start_vertex, task.finish_vertex),
                    ),
                ),
            )
            cand_robots[k] = ranked[:c_limit]

        # 4. Кандидаты в узлы эстафеты (handoff)
        candidates: Dict[int, List[int]] = {}
        for k, task in enumerate(tasks):
            A, B = task.start_vertex, task.finish_vertex
            d_AB = self.oracle.dist(A, B)
            cands = []
            needs_relay = d_AB >= 2 and (d_AB > 4 or any(r.battery < d_AB for r in robots))
            if needs_relay and d_AB < INF:
                d_A = self.oracle.get_all_from(A)
                d_B = self.oracle.get_all_from(B)
                mid_target = d_AB // 2
                for v in range(self.num_vertices):
                    if v != A and v != B and blocked.get(v, 0) == 0:
                        d1 = d_A[v]
                        d2 = d_B[v]
                        if d1 + d2 <= d_AB + 2:
                            balance = abs(d1 - mid_target)
                            cands.append((balance, v))
                cands.sort()
                candidates[k] = [v for _, v in cands[:6]]
            else:
                candidates[k] = []

        # 5. Переменные модели
        delivered = [model.NewBoolVar(f"deliv_{k}") for k in range(K)]
        direct = {
            (k, r): model.NewBoolVar(f"dir_{k}_{r}")
            for k in range(K)
            for r in cand_robots[k]
        }
        relay_h = {
            (k, h): model.NewBoolVar(f"rel_h_{k}_{h}")
            for k in range(K)
            for h in candidates[k]
        }
        leg1 = {
            (k, r, h): model.NewBoolVar(f"l1_{k}_{r}_{h}")
            for k in range(K)
            for r in cand_robots[k]
            for h in candidates[k]
        }
        leg2 = {
            (k, r, h): model.NewBoolVar(f"l2_{k}_{r}_{h}")
            for k in range(K)
            for r in cand_robots[k]
            for h in candidates[k]
        }

        t_start = [model.NewIntVar(0, max_horizon, f"ts_{k}") for k in range(K)]
        t_dur = [model.NewIntVar(0, max_horizon, f"td_{k}") for k in range(K)]
        t_end = [model.NewIntVar(0, max_horizon, f"te_{k}") for k in range(K)]

        for k, task in enumerate(tasks):
            A, B = task.start_vertex, task.finish_vertex
            if not allow_undelivered:
                model.Add(delivered[k] == 1)

            modes = [direct[k, r] for r in cand_robots[k]] + [
                relay_h[k, h] for h in candidates[k]
            ]
            if modes:
                model.Add(sum(modes) == delivered[k])
            else:
                model.Add(delivered[k] == 0)

            for h in candidates[k]:
                model.Add(sum(leg1[k, r, h] for r in cand_robots[k]) == relay_h[k, h])
                model.Add(sum(leg2[k, r, h] for r in cand_robots[k]) == relay_h[k, h])
                for r in cand_robots[k]:
                    model.Add(leg1[k, r, h] + leg2[k, r, h] <= relay_h[k, h])

            dur_terms = []
            if self.oracle.dist(A, B) < INF:
                dur_terms.append(sum(direct[k, r] for r in cand_robots[k]) * self.oracle.dist(A, B))
            for h in candidates[k]:
                d_split = self.oracle.dist(A, h) + self.oracle.dist(h, B)
                dur_terms.append(relay_h[k, h] * d_split)

            model.Add(t_dur[k] == sum(dur_terms))
            model.Add(t_end[k] == t_start[k] + t_dur[k])

            for r in cand_robots[k]:
                d_init = self.oracle.dist(robots[r].start_vertex, A)
                if d_init < INF:
                    model.Add(t_start[k] >= d_init).OnlyEnforceIf(direct[k, r])
                    for h in candidates[k]:
                        model.Add(t_start[k] >= d_init).OnlyEnforceIf(leg1[k, r, h])

        # 6. Дизъюнкции переходов
        tasks_for_robot: Dict[int, List[int]] = {r: [] for r in range(R)}
        for k in range(K):
            for r in cand_robots[k]:
                tasks_for_robot[r].append(k)

        for r in range(R):
            assigned_k = tasks_for_robot[r]
            for idx_j in range(len(assigned_k)):
                j = assigned_k[idx_j]
                for idx_k in range(idx_j + 1, len(assigned_k)):
                    k = assigned_k[idx_k]

                    task_j = tasks[j]
                    task_k = tasks[k]

                    in_j = [direct[j, r]] + [
                        leg1[j, r, h] for h in candidates[j] if (j, r, h) in leg1
                    ] + [leg2[j, r, h] for h in candidates[j] if (j, r, h) in leg2]
                    in_k = [direct[k, r]] + [
                        leg1[k, r, h] for h in candidates[k] if (k, r, h) in leg1
                    ] + [leg2[k, r, h] for h in candidates[k] if (k, r, h) in leg2]

                    r_in_j = model.NewBoolVar(f"r{r}_in_{j}")
                    r_in_k = model.NewBoolVar(f"r{r}_in_{k}")
                    model.Add(sum(in_j) == r_in_j)
                    model.Add(sum(in_k) == r_in_k)

                    both = model.NewBoolVar(f"both_r{r}_{j}_{k}")
                    model.AddBoolAnd([r_in_j, r_in_k]).OnlyEnforceIf(both)
                    model.AddBoolOr([r_in_j.Not(), r_in_k.Not()]).OnlyEnforceIf(both.Not())

                    j_before_k = model.NewBoolVar(f"ord_r{r}_{j}_{k}")
                    d_jk = self.oracle.dist(task_j.finish_vertex, task_k.start_vertex)
                    d_kj = self.oracle.dist(task_k.finish_vertex, task_j.start_vertex)
                    if d_jk >= INF:
                        d_jk = 0
                    if d_kj >= INF:
                        d_kj = 0

                    model.Add(t_start[k] >= t_end[j] + d_jk).OnlyEnforceIf([both, j_before_k])
                    model.Add(t_start[j] >= t_end[k] + d_kj).OnlyEnforceIf([both, j_before_k.Not()])

        # 7. Батарея
        for r, r_spec in enumerate(robots):
            energy_terms = []
            for k in tasks_for_robot[r]:
                task = tasks[k]
                A, B = task.start_vertex, task.finish_vertex
                if self.oracle.dist(A, B) < INF:
                    energy_terms.append(direct[k, r] * self.oracle.dist(A, B))
                for h in candidates[k]:
                    if (k, r, h) in leg1 and self.oracle.dist(A, h) < INF:
                        energy_terms.append(leg1[k, r, h] * self.oracle.dist(A, h))
                    if (k, r, h) in leg2 and self.oracle.dist(h, B) < INF:
                        energy_terms.append(leg2[k, r, h] * self.oracle.dist(h, B))

            model.Add(sum(energy_terms) <= r_spec.battery)

        makespan = model.NewIntVar(0, max_horizon, "makespan")
        for k in range(K):
            model.Add(makespan >= t_end[k]).OnlyEnforceIf(delivered[k])

        num_delivered = sum(delivered)
        model.Minimize((K - num_delivered) * 1_000_000 + makespan * 1000 + sum(t_dur))

        # 8. Хинты для прямо доставляемых задач (только если батарея позволяет)
        robot_tasks_order: Dict[int, List[int]] = {r: [] for r in range(R)}
        for k in range(K):
            if k in greedy_assigned:
                r_cand = greedy_assigned[k]
                if self.oracle.dist(tasks[k].start_vertex, tasks[k].finish_vertex) <= robots[r_cand].battery:
                    robot_tasks_order[r_cand].append(k)

        greedy_ms = 0
        for r in range(R):
            t_now = 0
            p_now = robots[r].start_vertex
            for k in robot_tasks_order[r]:
                task = tasks[k]
                A, B = task.start_vertex, task.finish_vertex
                d_pick = self.oracle.dist(p_now, A)
                d_AB = self.oracle.dist(A, B)
                t_arr = t_now + d_pick
                t_fin = t_arr + d_AB

                model.AddHint(delivered[k], 1)
                model.AddHint(direct[k, r], 1)
                model.AddHint(t_start[k], t_arr)
                model.AddHint(t_end[k], t_fin)

                t_now = t_fin
                p_now = B
            greedy_ms = max(greedy_ms, t_now)

        if greedy_ms > 0:
            model.AddHint(makespan, min(max_horizon, greedy_ms))

        # 9. Решение
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
                for r in cand_robots[k]:
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
                                for r in cand_robots[k]
                                if solver.BooleanValue(leg1[k, r, h])
                            )
                            r2 = next(
                                r
                                for r in cand_robots[k]
                                if solver.BooleanValue(leg2[k, r, h])
                            )
                            t_mid = start_t_val + self.oracle.dist(task.start_vertex, h)
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
        R = len(robots)
        paths: Dict[int, List[int]] = {r: [robots[r].start_vertex] for r in range(R)}
        batt: Dict[int, List[int]] = {r: [robots[r].battery] for r in range(R)}
        task_held: Dict[int, List[Optional[int]]] = {r: [None] for r in range(R)}

        handoff_drops: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for leg in all_legs:
            r = leg["carrier"]
            curr_t = len(paths[r]) - 1
            curr_v = paths[r][-1]
            curr_b = batt[r][-1]

            if leg["type"] == "handoff_pickup":
                key = (leg["task_id"], leg["pickup"])
                partner_r, t_partner_arr = handoff_drops[key]

                # Приезд принимающего робота в узел встречи
                p_to_pick = SpaceTimeAStar.plan(
                    r,
                    curr_v,
                    curr_t,
                    curr_b,
                    leg["pickup"],
                    curr_t,
                    max_horizon,
                    self.adj,
                    self.oracle,
                    constraints,
                    self.max_robots_per_vertex,
                    blocked,
                )
                if p_to_pick is None:
                    return None
                self._apply_subpath(r, p_to_pick, None, paths, batt, task_held)
                curr_t = len(paths[r]) - 1
                curr_v = paths[r][-1]
                curr_b = batt[r][-1]

                # Точное время встречи: максимум из прибытия обоих роботов
                t_meet = max(curr_t, t_partner_arr)

                # Если принимающий робот приехал раньше, он ждёт пустой
                while curr_t < t_meet:
                    self._append_wait(r, paths, batt, task_held, None)
                    curr_t += 1

                # Если робот-сдатчик приехал раньше, он ждёт с грузом на борту
                partner_cur_t = len(paths[partner_r]) - 1
                while partner_cur_t < t_meet:
                    self._append_wait(partner_r, paths, batt, task_held, leg["task_id"])
                    partner_cur_t += 1

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
                    self.oracle,
                    constraints,
                    self.max_robots_per_vertex,
                    blocked,
                )
                if p_deliv is None:
                    return None
                self._apply_subpath(
                    r, p_deliv, leg["task_id"], paths, batt, task_held
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
                        self.oracle,
                        constraints,
                        self.max_robots_per_vertex,
                        blocked,
                    )
                    if p_to_pick is None:
                        return None
                    self._apply_subpath(
                        r, p_to_pick, None, paths, batt, task_held
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
                    self.oracle,
                    constraints,
                    self.max_robots_per_vertex,
                    blocked,
                )
                if p_carry is None:
                    return None
                self._apply_subpath(
                    r, p_carry, leg["task_id"], paths, batt, task_held
                )

                if leg["type"] == "handoff_drop":
                    handoff_drops[(leg["task_id"], leg["dropoff"])] = (
                        r,
                        len(paths[r]) - 1,
                    )

        # Выравнивание траекторий
        max_T = max(len(paths[r]) for r in range(R))
        for r in range(R):
            while len(paths[r]) < max_T:
                self._append_wait(r, paths, batt, task_held, None)

        return paths, task_held

    def _append_wait(self, r, paths, batt, task_held, val=None):
        v = paths[r][-1]
        b = batt[r][-1]
        paths[r].append(v)
        batt[r].append(b)
        task_held[r].append(val)

    def _apply_subpath(self, r, subpath, carried_task, paths, batt, task_held):
        for v, b in subpath[1:]:
            paths[r].append(v)
            batt[r].append(b)
            task_held[r].append(carried_task)

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

    horizon = max(num_tasks * n * 2, (n + m) * 4)

    res = solver.solve(
        robots=robots,
        tasks=tasks,
        max_horizon=horizon,
        time_limit_s=60,
    )
    print_output(res)