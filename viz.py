import matplotlib.pyplot as plt
import numpy as np

def plot_tour(coords, tour, title="Tour"):
    coords_tour = coords[tour + [tour[0]]]
    plt.figure(figsize=(8, 6))
    plt.plot(coords_tour[:,0], coords_tour[:,1], 'o-', markersize=5)
    for i,(x,y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=6)
    plt.title(title)
    plt.axis("equal")
    plt.show()

def plot_fitness(history):
    plt.plot(history, color="green")
    plt.title("Evolución del fitness")
    plt.xlabel("Generación")
    plt.ylabel("Distancia")
    plt.grid(True)
    plt.show()
