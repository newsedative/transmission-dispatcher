"""Ближайший доставщик и случайная передача груза, без CP-SAT."""

import random

from test_run import DistanceOracle, GraphHandoffCpSatSolver, ScheduleResult


class GreedyHandoffSolver(GraphHandoffCpSatSolver):
    def __init__(self, num_vertices, edges, max_robots_per_vertex=2,
                 seed=0, handoff_probability=0.5):
        super().__init__(num_vertices, edges, max_robots_per_vertex)
        if not 0 <= handoff_probability <= 1:
            raise ValueError("handoff_probability должна быть от 0 до 1")
        self.seed = seed
        self.handoff_probability = handoff_probability

    def solve(self, robots, tasks, max_horizon=60, allow_floor_drop=False,
              time_limit_s=60.0, blocked_vertices=None, allow_undelivered=False,
              blocked_cargo=None):
        if tasks and not robots:
            status = "FEASIBLE" if allow_undelivered else "NO_PLAN"
            return ScheduleResult(allow_undelivered, 0, {}, [], status)
        result = super().solve(robots, tasks, max_horizon, allow_floor_drop,
                               time_limit_s, blocked_vertices, allow_undelivered, blocked_cargo)
        if result.success:
            result.solver_status = "FEASIBLE"
        return result

    def _solve_high_level(self, robots, tasks, max_horizon, blocked,
                          allow_undelivered, time_limit_s):
        rng = random.Random(self.seed)
        positions = [r.start_vertex for r in robots]
        available, battery = self._deployment_state(robots)
        if any((b < 0 or t > max_horizon) and r.release_time <= max_horizon
               for r, t, b in zip(robots, available, battery)):
            return {"success": False, "status": "NO_PLAN"}
        plan = []
        closed = {v for v, count in blocked.items() if count >= self.max_robots_per_vertex}
        oracle = self.oracle
        if closed:
            adj = {v: [u for u in neighbors if u not in closed] if v not in closed else []
                   for v, neighbors in self.adj.items()}
            oracle = DistanceOracle(self.num_vertices, adj)

        for task in tasks:
            start, finish = task.start_vertex, task.finish_vertex
            if start == finish:
                continue
            distance = oracle.dist(start, finish)
            if distance >= oracle.INF:
                if allow_undelivered:
                    continue
                return {"success": False, "status": "NO_PLAN"}

            from_start = oracle.get_all_from(start)
            to_finish = oracle.get_all_from(finish)

            nearest = sorted(range(len(robots)),
                             key=lambda r: oracle.dist(positions[r], start))
            for carrier in nearest:
                pickup = oracle.dist(positions[carrier], start)
                if pickup >= oracle.INF or pickup > battery[carrier]:
                    continue
                pickup_time = available[carrier] + pickup
                end = pickup_time + distance
                direct = pickup + distance <= battery[carrier] and end <= max_horizon
                partners = []
                for other in range(len(robots)):
                    if other == carrier:
                        continue
                    from_receiver = oracle.get_all_from(positions[other])
                    best = None
                    for vertex in self.vertices:
                        if (vertex in (start, finish)
                                or self.max_robots_per_vertex - blocked.get(vertex, 0) < 2):
                            continue
                        first_leg = from_start[vertex]
                        second_leg = to_finish[vertex]
                        travel = from_receiver[vertex]
                        if max(first_leg, second_leg, travel) >= oracle.INF:
                            continue
                        meeting = max(pickup_time + first_leg, available[other] + travel)
                        finish_time = meeting + second_leg
                        if (pickup + first_leg > battery[carrier]
                                or travel + second_leg > battery[other]
                                or finish_time > max_horizon):
                            continue
                        movement = pickup + first_leg + travel + second_leg
                        candidate = (finish_time, movement, vertex, first_leg,
                                     travel, meeting, second_leg)
                        if best is None or candidate < best:
                            best = candidate
                    if best is not None:
                        partners.append((other, best))
                if not direct and not partners:
                    continue

                relay = bool(partners) and (not direct or rng.random() < self.handoff_probability)
                receiver = None
                meeting = None
                if relay:
                    receiver, (end, _, vertex, first_leg, travel,
                               meeting, second_leg) = rng.choice(partners)
                    positions[carrier] = vertex
                    battery[carrier] -= pickup + first_leg
                    available[carrier] = meeting
                    positions[receiver] = finish
                    battery[receiver] -= travel + second_leg
                    available[receiver] = end
                else:
                    positions[carrier] = finish
                    battery[carrier] -= pickup + distance
                    available[carrier] = end

                plan.append({
                    "task_id": task.task_id, "start": start, "finish": finish,
                    "relay": relay, "r1": carrier, "r2": receiver,
                    "handoff_v": vertex if relay else None,
                    "t_start": pickup_time, "t_mid": meeting, "t_end": end,
                })
                break
            else:
                if not allow_undelivered:
                    return {"success": False, "status": "NO_PLAN"}

        return {"success": True, "tasks": plan, "status": "FEASIBLE"}
