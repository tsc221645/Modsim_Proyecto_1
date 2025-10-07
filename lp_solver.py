import pulp
import numpy as np


def solve_tsp_lp(D, time_limit=60):
    n = len(D)
    model = pulp.LpProblem("TSP", pulp.LpMinimize)

    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n)] for i in range(n)]
    u = [pulp.LpVariable(f"u_{i}", lowBound=0, upBound=n - 1, cat="Integer") for i in range(n)]

    # Fijar u[0] = 0 para estabilidad
    model += (u[0] == 0)

    model += pulp.lpSum(D[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)

    for i in range(n):
        model += pulp.lpSum(x[i][j] for j in range(n) if i != j) == 1
        model += pulp.lpSum(x[j][i] for j in range(n) if i != j) == 1

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model += u[i] - u[j] + n * x[i][j] <= n - 1

    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit)

    try:
        model.solve(solver)
    except Exception as e:
        print(f"[LP ERROR] CBC falló o se bloqueó: {e}")
        return [], None, 0, 0


    status = pulp.LpStatus[model.status]
    print(f"[LP] Estado: {status}")

    if status not in ["Optimal", "Integer Feasible"]:
        print("[LP] No se encontró solución óptima dentro del límite de tiempo.")
        return [], None, len(model.variables()), len(model.constraints)

    visited = [0]
    last = 0
    while len(visited) < n:
        next_city = None
        for j in range(n):
            if j not in visited and pulp.value(x[last][j]) > 0.5:
                next_city = j
                break
        if next_city is None:
            for j in range(n):
                if j not in visited and pulp.value(x[last][j]) > 0.1:
                    next_city = j
                    break
        if next_city is None:
            break
        visited.append(next_city)
        last = next_city
    visited.append(0)

    total_cost = pulp.value(pulp.lpSum(D[i][j] * x[i][j] for i in range(n) for j in range(n)))

    num_vars = len(model.variables())
    num_constraints = len(model.constraints)

    return visited, total_cost, num_vars, num_constraints
