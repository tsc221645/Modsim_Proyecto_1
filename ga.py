import numpy as np
import random
from distance import total_distance


def tournament_selection(pop, fitness, k=3):

    selected = random.sample(list(zip(pop, fitness)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0][:]


def order_crossover(p1, p2):
    
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n

    child[a:b] = p1[a:b]

    existing = set(child[a:b])
    p2_filtered = [gene for gene in p2 if gene not in existing]

    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = p2_filtered[idx]
            idx += 1

    return child

def swap_mutation(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]


def reverse_mutation(route):
    a, b = sorted(random.sample(range(len(route)), 2))
    route[a:b] = reversed(route[a:b])


def genetic_algorithm_tsp(
    D,
    N=200,
    maxIter=800,
    crossover_rate=0.7,
    mutation_rate=0.4,
    elite_k=3,
    seed=None,
):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = D.shape[0]

    pop = [list(np.random.permutation(n)) for _ in range(N)]

    fitness = [total_distance(D, p) for p in pop]

    best_idx = int(np.argmin(fitness))
    best_route = pop[best_idx][:]
    best_cost = fitness[best_idx]
    best_history = [best_cost]

    for it in range(maxIter):
        elites = sorted(zip(pop, fitness), key=lambda x: x[1])[:elite_k]
        new_pop = [e[0][:] for e in elites]
        remaining = N - elite_k
        total_rate = crossover_rate + mutation_rate
        if total_rate == 0:
            num_cross = 0
            num_mut = remaining
        else:
            num_cross = int(round(remaining * crossover_rate / total_rate))
            num_mut = remaining - num_cross

        for _ in range(num_cross):
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            child = order_crossover(p1, p2)
            new_pop.append(child)

        for _ in range(num_mut):
            indiv = tournament_selection(pop, fitness)
            mutated = indiv[:]
            if random.random() < 0.5:
                swap_mutation(mutated)
            else:
                reverse_mutation(mutated)
            new_pop.append(mutated)

        if len(new_pop) < N:
            while len(new_pop) < N:
                new_pop.append(list(np.random.permutation(n)))
        elif len(new_pop) > N:
            new_pop = new_pop[:N]

        pop = new_pop
        fitness = [total_distance(D, p) for p in pop]

        current_best_idx = int(np.argmin(fitness))
        current_best_cost = fitness[current_best_idx]
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_route = pop[current_best_idx][:]

        best_history.append(best_cost)

    return best_route, best_cost, best_history
