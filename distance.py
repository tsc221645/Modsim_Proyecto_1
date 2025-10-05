import numpy as np

def distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = np.linalg.norm(coords[i] - coords[j])
    return D

def compute_tour_length(D, tour):
    return sum(D[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
