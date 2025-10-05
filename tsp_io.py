import numpy as np

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

def generate_random_coords(n=150, seed=42):
    np.random.seed(seed)
    return np.random.rand(n, 2) * 100
