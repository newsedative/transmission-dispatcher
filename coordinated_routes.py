"""Coordinate the next moves of fixed task assignments without SAT."""

import time

import networkx as nx


def plan_routes(solver, robots, tasks, scheduled, max_horizon, time_limit_s):
    """Keep assignments and robot task order; coordinate moves and handoffs."""
    deadline = time.perf_counter() + time_limit_s
    count = len(robots)
    goals = [[(robot.start_vertex, None, 'deploy')] for robot in robots]
    legs = []
    for item in scheduled:
        task, first = item['task_id'], item['r1']
        if item['relay']:
            meeting, second = item['handoff_v'], item['r2']
            legs.append((item['t_start'], first,
                         [(item['start'], task, 'pickup'), (meeting, task, 'give')]))
            legs.append((item['t_mid'], second,
                         [(meeting, task, 'take'), (item['finish'], task, 'deliver')]))
        else:
            legs.append((item['t_start'], first,
                         [(item['start'], task, 'pickup'), (item['finish'], task, 'deliver')]))
    for _, robot, actions in sorted(legs, key=lambda leg: leg[0]):
        goals[robot].extend(actions)
    suffix = []
    for route in goals:
        remaining = [0] * len(route)
        for i in range(len(route) - 2, -1, -1):
            remaining[i] = remaining[i + 1] + solver.oracle.dist(route[i][0], route[i + 1][0])
        suffix.append(remaining)

    entries = [r.entry_vertex if r.entry_vertex is not None else r.start_vertex for r in robots]
    positions = [entries[r] if robot.release_time == 0 else None for r, robot in enumerate(robots)]
    present = [v for v in positions if v is not None]
    if len(set(present)) != len(present):
        return None
    battery = [r.battery for r in robots]
    carrying, index, age = [None] * count, [0] * count, [0] * count
    paths = {r: [positions[r]] for r in range(count)}
    held = {r: [None] for r in range(count)}
    delivered = {task.task_id for task in tasks if task.start_vertex == task.finish_vertex}
    required = {item['task_id'] for item in scheduled}

    def advance():
        for r in range(count):
            while positions[r] is not None and index[r] < len(goals[r]):
                vertex, task, action = goals[r][index[r]]
                if positions[r] != vertex or action in ('give', 'take'):
                    break
                if action == 'pickup':
                    carrying[r] = task
                elif action == 'deliver':
                    delivered.add(task)
                    carrying[r] = None
                index[r] += 1

    advance()
    for tick in range(1, max_horizon + 1):
        if time.perf_counter() >= deadline:
            return None
        active = [r for r in range(count) if robots[r].release_time <= tick]
        choices, costs, distance_before, goal_distances = {}, {}, {}, {}
        for r in active:
            born = robots[r].release_time == tick
            source = positions[r]
            head = goals[r][index[r]] if index[r] < len(goals[r]) else None
            distances = solver.oracle.get_all_from(head[0]) if head else None
            goal_distances[r] = distances
            distance_before[r] = distances[source] if distances and source is not None else 0
            choices[r], costs[r] = [], {}
            for vertex in [entries[r]] if born else solver.adj[source]:
                moved = source is not None and source != vertex
                reserve = distances[vertex] + suffix[r][index[r]] if head else 0
                if battery[r] - moved < reserve:
                    continue
                choices[r].append(vertex)
                costs[r][vertex] = 10 * moved
            if not choices[r]:
                return None

        donors, receivers = {}, {}
        for r in active:
            if positions[r] is None or index[r] >= len(goals[r]):
                continue
            vertex = goals[r][index[r]][0]
            future_cargo = carrying[r]
            for target, task, action in goals[r][index[r]:]:
                if target != vertex:
                    break
                if action == 'give' and carrying[r] == task:
                    donors[task] = r
                elif action == 'take' and future_cargo is None:
                    receivers[task] = r
                if action in ('give', 'take'):
                    break
                if action == 'deliver':
                    future_cargo = None
                elif action == 'pickup':
                    future_cargo = task
        pairs = [(donors[task], receivers[task], task) for task in donors.keys() & receivers.keys()]
        pairs.sort(key=lambda pair: (-max(age[pair[0]], age[pair[1]]), pair[2]))
        groups, occupied = [], set()
        births = {entries[r] for r in active if robots[r].release_time == tick}
        for first, second, task in pairs:
            vertex = goals[first][index[first]][0]
            if (vertex == goals[second][index[second]][0] and vertex not in occupied
                    and vertex in choices[first] and vertex in choices[second]
                    and vertex not in births and solver.max_robots_per_vertex >= 2):
                groups.append((first, second, task, vertex))
                occupied.add(vertex)

        productive, approaching = [], []
        for r in active:
            if index[r] == len(goals[r]):
                continue
            _, task, action = goals[r][index[r]]
            partner = (receivers if action == 'give' else donors).get(task)
            if action not in ('give', 'take'):
                productive.append(r)
            elif partner is not None and distance_before[r] > 1:
                approaching.append(r)
        priorities = {r: 1000 * 2 ** rank for rank, r in
                      enumerate(sorted(productive, key=lambda r: (age[r], -r)))}
        priorities.update({r: 100 for r in approaching})
        for r in active:
            if goal_distances[r] is not None:
                for vertex in choices[r]:
                    costs[r][vertex] += goal_distances[r][vertex] * priorities.get(r, 1)

        # A paired move occupies one cell and commits the transfer on arrival.
        while True:
            bundled = {r for first, second, _, _ in groups for r in (first, second)}
            units = []
            for r in active:
                if r in bundled:
                    continue
                units.append((('robot', r), choices[r], costs[r]))
            units += [(('pair', task), [v], {v: 0}) for _, _, task, v in groups]
            graph = nx.DiGraph()
            sink = ('sink',)
            graph.add_node(sink, demand=len(units))
            for unit, vertices, prices in units:
                graph.add_node(unit, demand=-1)
                for vertex in vertices:
                    cell = ('cell', vertex)
                    graph.add_edge(unit, cell, capacity=1, weight=prices[vertex])
                    graph.add_edge(cell, sink, capacity=1, weight=0)
            try:
                _, flow = nx.network_simplex(graph)
                destinations = {unit: next(cell[1] for cell, amount in flow[unit].items() if amount)
                                for unit, _, _ in units}
                break
            except nx.NetworkXUnfeasible:
                if not groups:
                    return None
                groups.pop()

        before_index = index.copy()
        for r in range(count):
            vertex = destinations.get(('robot', r))
            for first, second, task, _ in groups:
                if r in (first, second):
                    vertex = destinations[('pair', task)]
                    break
            if r in choices:
                born = robots[r].release_time == tick
                battery[r] -= positions[r] is not None and positions[r] != vertex
                held[r].append(None if born else carrying[r])
            else:
                held[r].append(None)
            positions[r] = vertex
            paths[r].append(vertex)
        advance()
        for first, second, task, _ in groups:
            carrying[first], carrying[second] = None, task
            index[first] += 1
            index[second] += 1
        advance()
        for r in active:
            age[r] = 0 if index[r] != before_index[r] else age[r] + 1
        if required <= delivered:
            if solver._find_vertex_conflict(paths, {}, held) is not None:
                return None
            return solver._build_schedule_result(robots, tasks, paths, held, 'FEASIBLE')
    return None
