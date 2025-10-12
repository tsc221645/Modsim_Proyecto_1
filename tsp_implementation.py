# implementing tsp case
import numpy as np
import random
import matplotlib.pyplot as plt


def read_tsp_file(filepath):
    coords = []
    with open(filepath, 'r') as f:
        start = False
        for line in f:
            if "NODE_COORD_SECTION" in line:
                start = True
                continue
            if "EOF" in line or line.strip() == "":
                break
            if start:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))
    return np.array(coords)

def distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = np.linalg.norm(coords[i] - coords[j])
    return D

def compute_tour_length(D, tour):
    return sum(D[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

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

# --- Algoritmo Genético Optimizado ---
def genetic_algorithm_tsp_optimizado(D, N=200, maxIter=800, crossover_rate=0.7, mutation_rate=0.4):
    n = len(D)
    pop = [list(np.random.permutation(n)) for _ in range(N)]
    fitness = [compute_tour_length(D, ind) for ind in pop]
    best_fitness_per_gen = []
    top_k = 3  # elitismo

    for gen in range(maxIter):
        new_pop = []

        # Elitismo
        elites = sorted(zip(pop, fitness), key=lambda x: x[1])[:top_k]
        for elite in elites:
            new_pop.append(elite[0][:])

        # Cruces
        num_cross = int(crossover_rate * N)
        for _ in range(num_cross):
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            child = order_crossover(p1, p2)
            new_pop.append(child)

        # Mutaciones
        num_mut = int(mutation_rate * N)
        for _ in range(num_mut):
            indiv = tournament_selection(pop, fitness)
            mutated = indiv[:]
            if random.random() < 0.5:
                swap_mutation(mutated)
            else:
                reverse_mutation(mutated)
            new_pop.append(mutated)

        # Rellenar si falta
        while len(new_pop) < N:
            new_pop.append(list(np.random.permutation(n)))

        # Evaluación
        pop = new_pop
        fitness = [compute_tour_length(D, ind) for ind in pop]
        current_best = min(zip(pop, fitness), key=lambda x: x[1])
        best_fitness_per_gen.append(current_best[1])

    best_index = np.argmin(fitness)
    best_solution = pop[best_index]
    return best_solution, best_fitness_per_gen


coords = read_tsp_file("inventado100.tsp")
D = distance_matrix(coords)
best_tour, history = genetic_algorithm_tsp_optimizado(D)
distancia = compute_tour_length(D, best_tour)
print(f"Distancia total del mejor recorrido: {distancia:.2f}")



def plot_tour(coords, tour, title="Mejor recorrido encontrado"):
    coords_tour = coords[tour + [tour[0]]]  # cerrar el ciclo
    plt.figure(figsize=(10, 6))
    plt.plot(coords_tour[:, 0], coords_tour[:, 1], 'o-', markersize=5)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=8)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def plot_fitness_evolution(fitness_history):
    plt.figure(figsize=(8, 4))
    plt.plot(fitness_history, label="Mejor distancia por generación", color="green")
    plt.xlabel("Generación")
    plt.ylabel("Distancia")
    plt.title("Evolución del fitness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_tour(coords, best_tour, title=f"Mejor recorrido encontrado (D = {distancia:.2f})")
plot_fitness_evolution(history)

print("\nRecorrido final (best_tour):")
print(best_tour)
