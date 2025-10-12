import numpy as np


def distance_matrix(coords):

    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                xi, yi = coords[i]
                xj, yj = coords[j]
                D[i, j] = np.hypot(xi - xj, yi - yj)
    return D


def total_distance(D, route):
    n = len(route)
    dist = 0.0
    for i in range(n):
        dist += D[route[i], route[(i + 1) % n]] 
    return dist
