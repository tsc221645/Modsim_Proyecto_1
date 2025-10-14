import numpy as np
import random
from distance import total_distance


# =========================
# Utilidades de robustez
# =========================
def is_valid_perm(route, n):
    """Verifica si una ruta es una permutación válida de 0..n-1."""
    if route is None or len(route) != n:
        return False
    try:
        if min(route) < 0 or max(route) >= n:
            return False
    except ValueError:
        return False
    return len(set(route)) == n


def repair_permutation(route, n):
    """Repara duplicados y faltantes en O(n)."""
    if route is None:
        return list(range(n))

    seen = set()
    dup_positions = []
    for i, g in enumerate(route):
        if g in seen or g < 0 or g >= n:
            dup_positions.append(i)
        else:
            seen.add(g)

    missing = list(set(range(n)) - seen)
    random.shuffle(missing)

    for i, pos in enumerate(dup_positions):
        if i < len(missing):
            route[pos] = missing[i]
        else:
            route[pos] = random.randint(0, n - 1)
    return route


def safe_total_distance(D, route, big_penalty=1e12):
    """Evalúa distancia total de una ruta; repara si es necesario."""
    n = D.shape[0]
    if route is None or len(route) != n:
        return big_penalty

    try:
        route = [int(x) for x in route]
    except Exception:
        return big_penalty

    if not is_valid_perm(route, n):
        route = repair_permutation(route, n)
    if not is_valid_perm(route, n):
        return big_penalty

    try:
        idx = np.asarray(route, dtype=np.int32)
        nxt = np.roll(idx, -1)
        dist = float(D[idx, nxt].sum())
    except Exception:
        return big_penalty

    if dist <= 0.0 or np.isnan(dist):
        return big_penalty
    return dist


# =========================
# Selección
# =========================
def tournament_selection(pop, fitness, k=3):
    selected = random.sample(list(zip(pop, fitness)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0][:]


# =========================
# Cruces
# =========================
def order_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = p1[a:b]
    used = set(p1[a:b])

    p2_filtered = [gene for gene in p2 if gene not in used]
    idx = 0
    for i in range(n):
        if child[i] == -1 and idx < len(p2_filtered):
            child[i] = p2_filtered[idx]
            idx += 1
    return repair_permutation(child, n)


def pmx_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = p1[a:b]
    mapping = {p2[i]: p1[i] for i in range(a, b)}

    for i in range(n):
        if a <= i < b:
            continue
        val = p2[i]
        visited = set()
        while val in mapping and val not in visited:
            visited.add(val)
            val = mapping[val]
        child[i] = val

    return repair_permutation(child, n)


# =========================
# Mutaciones
# =========================
def swap_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]


def reverse_mutation(route):
    a, b = sorted(random.sample(range(len(route)), 2))
    route[a:b] = reversed(route[a:b])


# =========================
# Heurísticas iniciales
# =========================
def nearest_neighbor_route(D, start=0):
    n = D.shape[0]
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    while unvisited:
        last = route[-1]
        next_city = min(unvisited, key=lambda j: D[last, j])
        route.append(next_city)
        unvisited.remove(next_city)
    return route


# =========================
# Refinamiento local (2-opt)
# =========================
def two_opt(route, D, max_trials=150):
    """Refina una ruta intercambiando segmentos si reduce la distancia."""
    n = len(route)
    best = route[:]
    best_dist = safe_total_distance(D, best)
    for _ in range(max_trials):
        i, j = sorted(random.sample(range(1, n), 2))
        if j - i <= 1:
            continue
        new_route = best[:]
        new_route[i:j] = reversed(best[i:j])
        new_dist = safe_total_distance(D, new_route)
        if new_dist < best_dist:
            best, best_dist = new_route, new_dist
    return best


# =========================
# Algoritmo Genético
# =========================
def genetic_algorithm_tsp(
    D,
    N=400,
    maxIter=1200,
    crossover_rate=0.85,
    mutation_rate=0.28,
    elite_k=5,
    seed=None,
    log_every=100,
):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = D.shape[0]

    # Población inicial: 70 % aleatoria, 20 % NN, 10 % greedy inversa
    pop = [list(np.random.permutation(n)) for _ in range(int(N * 0.7))]
    for i in range(int(N * 0.2)):
        pop.append(nearest_neighbor_route(D, start=i % n))
    for i in range(int(N * 0.1)):
        start = random.randint(0, n - 1)
        route = list(np.argsort(D[start]))[::-1]
        pop.append(route)

    pop = [repair_permutation(p, n) for p in pop]
    fitness = [safe_total_distance(D, p) for p in pop]

    # Mejor inicial
    best_idx = int(np.argmin(fitness))
    best_route = pop[best_idx][:]
    best_cost = fitness[best_idx]
    best_history = [best_cost]
    stagnation = 0

    for it in range(maxIter):
        elites = sorted(zip(pop, fitness), key=lambda x: x[1])[:elite_k]
        new_pop = [e[0][:] for e in elites]

        # Mutación adaptativa
        current_mut_rate = min(mutation_rate * (1.0 + stagnation / 40.0), 0.9)
        remaining = N - elite_k
        num_cross = int(round(remaining * crossover_rate))
        num_mut = remaining - num_cross

        # Cruces
        for _ in range(num_cross):
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            if random.random() < 0.6:
                child = order_crossover(p1, p2)
            else:
                child = pmx_crossover(p1, p2)
            new_pop.append(child)

        # Mutaciones
        for _ in range(num_mut):
            indiv = tournament_selection(pop, fitness)
            mutated = indiv[:]
            if random.random() < current_mut_rate:
                if random.random() < 0.5:
                    swap_mutation(mutated)
                else:
                    reverse_mutation(mutated)
            new_pop.append(mutated)

        # Reinyección si hay estancamiento
        if stagnation > 200:
            print(f"[GA] Reinitializing partial population at iter {it}")
            for k in range(int(N * 0.3)):
                pos = random.randint(elite_k, N - 1)
                new_pop[pos] = list(np.random.permutation(n))
            stagnation = 0

        # Mantener tamaño
        if len(new_pop) < N:
            while len(new_pop) < N:
                new_pop.append(list(np.random.permutation(n)))
        elif len(new_pop) > N:
            new_pop = new_pop[:N]

        new_pop[0] = best_route[:]

        # Evaluación
        pop = [repair_permutation(p, n) for p in new_pop]
        fitness = [safe_total_distance(D, p) for p in pop]

        # Mejor actual
        current_best_idx = int(np.argmin(fitness))
        current_best_route = pop[current_best_idx][:]
        current_best_cost = fitness[current_best_idx]

        # Refinamiento local cada 50 iteraciones
        if it % 50 == 0:
            refined = two_opt(best_route, D, max_trials=150)
            refined_cost = safe_total_distance(D, refined)
            if refined_cost < current_best_cost:
                current_best_route, current_best_cost = refined, refined_cost

        # Actualizar mejor global
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_route = current_best_route[:]
            stagnation = 0
        else:
            stagnation += 1

        best_history.append(best_cost)

        if log_every and (it % log_every == 0):
            print(f"[GA] iter {it:4d} | best={best_cost:.4f} | mut={current_mut_rate:.2f}")

    return best_route, best_cost, best_history
