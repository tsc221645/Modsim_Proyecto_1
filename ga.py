import numpy as np
import random
from distance import compute_tour_length

def tournament_selection(pop, fitness, k=3):
    selected = random.sample(range(len(pop)), k)
    best = min(selected, key=lambda idx: fitness[idx])
    return pop[best]

def order_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b+1] = parent1[a:b+1]
    ptr = 0
    for i in range(size):
        if parent2[i] not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = parent2[i]
    return child

def swap_mutation(tour):
    a, b = sorted(random.sample(range(len(tour)), 2))
    tour[a], tour[b] = tour[b], tour[a]

def reverse_mutation(tour):
    a, b = sorted(random.sample(range(len(tour)), 2))
    tour[a:b+1] = reversed(tour[a:b+1])

def genetic_algorithm_tsp(D, N=200, maxIter=800,
                          crossover_rate=0.7, mutation_rate=0.4,
                          elite_k=3):
    n = len(D)
    pop = [list(np.random.permutation(n)) for _ in range(N)]
    fitness = [compute_tour_length(D, ind) for ind in pop]
    best_fitness_per_gen = []

    for _ in range(maxIter):
        new_pop = []

        # elitismo
        elites = sorted(zip(pop, fitness), key=lambda x: x[1])[:elite_k]
        new_pop.extend([e[0][:] for e in elites])

        # cruce
        for _ in range(int(crossover_rate * N)):
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            child = order_crossover(p1, p2)
            new_pop.append(child)

        # mutación
        for _ in range(int(mutation_rate * N)):
            indiv = tournament_selection(pop, fitness)
            mutated = indiv[:]
            if random.random() < 0.5:
                swap_mutation(mutated)
            else:
                reverse_mutation(mutated)
            new_pop.append(mutated)

        while len(new_pop) < N:
            new_pop.append(list(np.random.permutation(n)))

        pop = new_pop
        fitness = [compute_tour_length(D, ind) for ind in pop]
        best_fitness_per_gen.append(min(fitness))

    best_index = np.argmin(fitness)
    return pop[best_index], best_fitness_per_gen
