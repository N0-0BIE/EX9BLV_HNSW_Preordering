import numpy as np

############################
###        TwoNN         ###
############################
def euclidian_distance(x, y):
    if len(x) != len(y):
        raise ValueError("Dimensions not matching!")

    sum = 0
    for i in range(0, len(x)):
        sum += (y[i] - x[i]) ** 2
    return np.sqrt(sum)

# TwoNN értékek meghatározása az adathalmazban minden pontra
def calculate_twonn(data):
    # 2NN számítás minden pontra
    k = 2
    data = np.asarray([np.asarray(v) for v in data], dtype=np.float32)
    scores = data @ data.T

    # Assign the same ID to identical vectors, then exclude them from each row.
    _, vector_ids = np.unique(data, axis=0, return_inverse=True)
    for row, vector_id in enumerate(vector_ids):
        scores[row, vector_ids == vector_id] = -np.inf

    topk = np.argpartition(
        -scores,
        kth=k-1,
        axis=1,
    )[:, :k]

    # Optional: sort those k results by actual score
    rows = np.arange(len(data))[:, None]
    order = np.argsort(-scores[rows, topk], axis=1)

    twonn_matrix = np.take_along_axis(topk, order, axis=1)

    # TwoNN számítása egyes vektorokra
    mu = np.full(len(data), np.nan, dtype=np.float64)

    for i, neighbor_indices in enumerate(twonn_matrix):
        #Távolságok számítása
        distances = np.array([euclidian_distance(data[i], data[j]) for j in neighbor_indices])

        # TwoNN érték kiszámítása
        mu[i] = distances[1]/distances[0]

    return mu