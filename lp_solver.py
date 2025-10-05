import pulp
from distance import compute_tour_length

def solve_tsp_lp(D):
    n = len(D)
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat="Binary")
    u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n-1, cat="Integer")

    prob += pulp.lpSum(D[i][j] * x[i][j] for i in range(n) for j in range(n))

    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(n) if j != i) == 1
        prob += pulp.lpSum(x[j][i] for j in range(n) if j != i) == 1

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i][j] <= n-1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    tour = [0]
    while len(tour) < n:
        last = tour[-1]
        for j in range(n):
            if pulp.value(x[last][j]) == 1:
                tour.append(j)
                break
    return tour, compute_tour_length(D, tour)
