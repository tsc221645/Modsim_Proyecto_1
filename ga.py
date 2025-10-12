import numpy as np
import random
from distance import total_distance  # opcional; usamos safe_total_distance internamente


# =========================
# Utilidades de robustez
# =========================
def is_valid_perm(route, n):
    return len(route) == n and len(set(route)) == n and min(route) >= 0 and max(route) < n


def repair_permutation(route, n):
    seen = set()
    missing = [g for g in range(n) if g not in route]
    miss_idx = 0
    for i, g in enumerate(route):
        if g in seen or g < 0 or g >= n:
            route[i] = missing[miss_idx]
            miss_idx += 1
        else:
            seen.add(g)
    return route


def safe_total_distance(D, route, big_penalty=1e12):
    n = D.shape[0]
    if not is_valid_perm(route, n):
        route = repair_permutation(route, n)
    if not is_valid_perm(route, n):
        return big_penalty

    dist = 0.0
    for i in range(n):
        dist += D[route[i], route[(i + 1) % n]]
    if dist <= 0.0:
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

    # Copiar segmento de p1
    child[a:b] = p1[a:b]
    used = set(p1[a:b])
    p2_filtered = [gene for gene in p2 if gene not in used]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            if idx < len(p2_filtered):
                child[i] = p2_filtered[idx]
                idx += 1
            else:
                break  
    repair_permutation(child, n)
    return child


def pmx_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n

    # Copiar segmento central de p1
    child[a:b] = p1[a:b]

    # Mapeo entre segmentos p2 -> p1
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

    # Relleno y reparación por seguridad
    repair_permutation(child, n)
    return child


# =========================
# Mutaciones
# =========================
def swap_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    repair_permutation(route, len(route))


def reverse_mutation(route):
    a, b = sorted(random.sample(range(len(route)), 2))
    route[a:b] = reversed(route[a:b])
    repair_permutation(route, len(route))


# =========================
# Heurística para población inicial
# =========================
def nearest_neighbor_route(D, start=0):
    n = D.shape[0]
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    while unvisited:
        last = route[-1]
        # ciudad más cercana aún no visitada
        next_city = min(unvisited, key=lambda j: D[last, j])
        route.append(next_city)
        unvisited.remove(next_city)
    return route


# =========================
# Algoritmo Genético
# =========================
def genetic_algorithm_tsp(
    D,
    N=200,
    maxIter=800,
    crossover_rate=0.7,
    mutation_rate=0.4,
    elite_k=3,
    seed=None,
    log_every=0,  # 0 = sin logs, >0 imprime cada 'log_every' iteraciones
):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = D.shape[0]

    # --- Población inicial: 80% aleatoria + 20% heurística NN
    pop = [list(np.random.permutation(n)) for _ in range(int(N * 0.8))]
    for i in range(int(N * 0.2)):
        pop.append(nearest_neighbor_route(D, start=i % n))

    # Reparación por seguridad
    for i in range(N):
        repair_permutation(pop[i], n)

    # Evaluación inicial segura
    fitness = [safe_total_distance(D, p) for p in pop]

    # Mejor global
    best_idx = int(np.argmin(fitness))
    best_route = pop[best_idx][:]
    best_cost = fitness[best_idx]
    best_history = [best_cost]

    stagnation = 0  # generaciones sin mejorar

    for it in range(maxIter):
        # Elitismo (preservar mejores)
        elites = sorted(zip(pop, fitness), key=lambda x: x[1])[:elite_k]
        new_pop = [e[0][:] for e in elites]

        # Mutación adaptativa si no mejora
        current_mut_rate = min(mutation_rate * (1.0 + stagnation / 50.0), 0.9)

        remaining = N - elite_k
        num_cross = int(round(remaining * crossover_rate))
        num_mut = remaining - num_cross

        # --- Cruce
        for _ in range(num_cross):
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            if random.random() < 0.6:
                child = order_crossover(p1, p2)
            else:
                child = pmx_crossover(p1, p2)
            new_pop.append(child)

        # --- Mutación (probabilística con tasa adaptativa)
        for _ in range(num_mut):
            indiv = tournament_selection(pop, fitness)
            mutated = indiv[:]
            if random.random() < current_mut_rate:
                if random.random() < 0.5:
                    swap_mutation(mutated)
                else:
                    reverse_mutation(mutated)
            # reparar por seguridad (no debería ser necesario, pero es barato)
            repair_permutation(mutated, n)
            new_pop.append(mutated)

        # Inyección de diversidad cada 100 iteraciones
        if it > 0 and it % 100 == 0:
            for _ in range(max(1, int(N * 0.1))):
                pos = random.randint(0, N - 1)
                new_pop[pos] = list(np.random.permutation(n))

        # Asegurar tamaño
        if len(new_pop) < N:
            while len(new_pop) < N:
                new_pop.append(list(np.random.permutation(n)))
        elif len(new_pop) > N:
            new_pop = new_pop[:N]

        # Preservar explícitamente el mejor global
        new_pop[0] = best_route[:]

        # Evaluar generación
        pop = new_pop
        fitness = [safe_total_distance(D, p) for p in pop]

        # Actualizar mejor
        current_best_idx = int(np.argmin(fitness))
        current_best_route = repair_permutation(pop[current_best_idx][:], n)
        current_best_cost = safe_total_distance(D, current_best_route)

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_route = current_best_route[:]
            stagnation = 0
        else:
            stagnation += 1

        best_history.append(best_cost)

        if log_every and (it % log_every == 0):
            print(f"[GA] iter {it:4d} | best = {best_cost:.4f} | mut={current_mut_rate:.2f}")

    return best_route, best_cost, best_history
