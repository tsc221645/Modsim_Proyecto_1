import numpy as np

def generar_escenario_inventado(n_ciudades=100, nombre_archivo="inventado100.tsp"):
    t = np.linspace(0, 4*np.pi, n_ciudades)   # parámetro para espiral
    r = np.linspace(10, 100, n_ciudades)      # radio creciente
    x = r * np.cos(t) + 500                    # desplazamiento para coordenadas positivas
    y = r * np.sin(t) + 500

    coords = np.column_stack((x, y))

    with open(nombre_archivo, "w") as f:
        f.write(f"NAME: {nombre_archivo}\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"DIMENSION: {n_ciudades}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (xi, yi) in enumerate(coords, start=1):
            f.write(f"{i} {xi:.2f} {yi:.2f}\n")
        f.write("EOF\n")
    print(f"Escenario inventado guardado en {nombre_archivo} con {n_ciudades} ciudades.")

# Generar ejemplo con 100 ciudades
generar_escenario_inventado(100, "inventado100.tsp")
