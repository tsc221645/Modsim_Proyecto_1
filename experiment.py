import time
from tsp_io import read_tsp_file
from distance import distance_matrix, compute_tour_length
from ga import genetic_algorithm_tsp
from lp_solver import solve_tsp_lp
from viz import plot_tour, plot_fitness

def run_experiment(filepath, scenario_name="TSP Scenario",
                   N=50, maxIter=50, crossover_rate=0.7, mutation_rate=0.4,
                   run_lp=False):
    # Leer archivo
    coords = read_tsp_file(filepath)
    if coords.size == 0:
        print(f"Error: no se pudieron cargar coordenadas de {filepath}")
        return
    print(f"Coords cargadas ({scenario_name}): {coords.shape}")

    # Matriz de distancias
    D = distance_matrix(coords)

    #  Algoritmo Genético 
    print("\nEjecutando GA...")
    start = time.time()
    best_tour_ga, history = genetic_algorithm_tsp(D, N=N, maxIter=maxIter,
                                                  crossover_rate=crossover_rate,
                                                  mutation_rate=mutation_rate)
    t_ga = time.time() - start
    dist_ga = compute_tour_length(D, best_tour_ga)
    print(f"GA completado: distancia={dist_ga:.2f}, tiempo={t_ga:.2f}s")

    # Visualización GA
    plot_tour(coords, best_tour_ga, title=f"GA - {scenario_name}")
    plot_fitness(history)

    print("\nMejor recorrido encontrado (orden de ciudades):")
    print(best_tour_ga)


    #  Solver LP
    if run_lp:
        print("\nEjecutando LP...")
        start = time.time()
        best_tour_lp, dist_lp = solve_tsp_lp(D)
        t_lp = time.time() - start
        print(f"LP completado: distancia={dist_lp:.2f}, tiempo={t_lp:.2f}s")
        plot_tour(coords, best_tour_lp, title=f"LP - {scenario_name}")
    else:
        print("\nLP desactivado (activar run_lp=True para usarlo)")

# run_experiment("berlin52.tsp", scenario_name="Berlin52",
#                N=50, maxIter=50, crossover_rate=0.7, mutation_rate=0.4,
#                run_lp=False)

# Para eil101
run_experiment("eil101.tsp", scenario_name="EIL101", N=50, maxIter=100)

# Para gr229
run_experiment("gr229.tsp", scenario_name="GR229", N=50, maxIter=100)

# Para inventado
run_experiment("inventado100.tsp", scenario_name="GR229", N=50, maxIter=100)

