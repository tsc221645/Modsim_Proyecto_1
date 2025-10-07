import matplotlib.pyplot as plt
import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_tour(coords, route, title="TSP Route"):
    ensure_dir("results/plots")

    x = [coords[i][0] for i in route] + [coords[route[0]][0]]
    y = [coords[i][1] for i in route] + [coords[route[0]][1]]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, "o-", markersize=4)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()

    filename = f"results/plots/{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Gráfico guardado: {filename}")


def plot_fitness(history, title="Fitness Evolution"):
    ensure_dir("results/plots")

    plt.figure(figsize=(6, 4))
    plt.plot(history, color="blue")
    plt.title(title)
    plt.xlabel("Iteración")
    plt.ylabel("Distancia mínima")
    plt.grid(True)
    plt.tight_layout()

    filename = f"results/plots/{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Gráfico guardado: {filename}")
