import time
import csv
from tsp_io import read_tsp_file
from distance import distance_matrix, total_distance
from ga import genetic_algorithm_tsp
from lp_solver import solve_tsp_lp
from viz import plot_tour, plot_fitness


def run_experiment(
    tsp_filename,
    scenario_name,
    N=200,
    maxIter=800,
    crossover_rate=0.7,
    mutation_rate=0.4,
    elite_k=3,
    seed=None,
    run_lp=True,
    save_csv=True
):

    print(f"\n=== Escenario: {scenario_name} ===")

    coords = read_tsp_file(tsp_filename)
    D = distance_matrix(coords)
    n = D.shape[0]

    print("Ejecutando GA...")
    start = time.time()
    best_route, best_cost, best_history = genetic_algorithm_tsp(
        D,
        N=N,
        maxIter=maxIter,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_k=elite_k,
        seed=seed
    )
    t_ga = time.time() - start
    print(f"[GA] n={n}  Mejor longitud: {best_cost:.4f}  Tiempo: {t_ga:.2f}s")

    plot_tour(coords, best_route, title=f"GA - {scenario_name}")
    plot_fitness(best_history, title=f"GA Fitness - {scenario_name}")

    lp_cost = None
    lp_route = []
    lp_time = 0
    lp_vars = "-"
    lp_cons = "-"
    lp_status = "Not Run"

    if run_lp:
        print("Ejecutando LP...")
        start = time.time()
        lp_route, lp_cost, lp_vars, lp_cons = solve_tsp_lp(D,120)
        lp_time = time.time() - start

        if lp_cost is not None:
            lp_status = "Optimal"
            print(f"[LP] Status: Óptimo  Longitud: {lp_cost:.4f}  Tiempo: {lp_time:.2f}s")
            plot_tour(coords, lp_route, title=f"LP - {scenario_name}")
            error = 100 * (best_cost - lp_cost) / lp_cost
            print(f"[Comparación] Error GA vs LP: {error:.2f}%")
        else:
            lp_status = "Timeout/NotSolved"
            print(f"[LP] Status: {lp_status} (sin solución en el límite de tiempo)")
            print(f"[LP] Tiempo total: {lp_time:.2f}s")

    if save_csv:
        with open("results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                scenario_name,
                n,
                N,
                maxIter,
                crossover_rate,
                mutation_rate,
                elite_k,
                round(best_cost, 6),
                round(t_ga, 3),
                round(lp_cost, 6) if lp_cost is not None else "-",
                round(lp_time, 3),
                lp_vars,
                lp_cons,
                lp_status
            ])

    print("----------------------------------------------------------")


if __name__ == "__main__":

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Escenario", "n", "N", "maxIter", "crossover_rate", "mutation_rate",
            "elite_k", "GA_len", "GA_time", "LP_len", "LP_time", "LP_vars", "LP_cons", "LP_status"
        ])

    run_experiment("eil101.tsp", "EIL101", N=200, maxIter=800, seed=42)
    run_experiment("gr229.tsp", "GR229", N=250, maxIter=1000, seed=123)
    run_experiment("inventado100.tsp", "Inventado_100", N=150, maxIter=500, seed=99)
