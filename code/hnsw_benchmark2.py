import numpy as np
import hnswlib
from time import time


############################
###         KNN          ###
############################
class KNN:
    def __init__(self, data):
        self._data = data

    def euclidian_distance(x, y):
        if len(x) != len(y):
            raise ValueError("Dimensions not matching!")
        
        sum = 0
        for i in range(0, len(x)):
            sum += (y[i] - x[i]) ** 2
        return np.sqrt(sum)

    def get_KNN(self, q, k, distance = euclidian_distance):
        distances = [distance(q, v) for v in self._data]
        knn_index = np.argsort(distances) # rendezés index szerint
        knn_index = knn_index[np.asarray(distances)[knn_index] > 0] # 0 távolságú indexek eldobása
        knn_index = knn_index[:k] # első k elem
        return np.array(knn_index)


############################
###    HNSW benchmark    ###
############################
class HNSW_benchmark:
    def __init__(self, data, M = 16, ef_construction = 128):
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("Bad dataset given. Count or dimension = 0")
        
        # Alap adatok
        self._data = [np.asarray(v) for v in data]
        self._M = M
        self._ef_construction = ef_construction
        # HNSW init
        self._hnsw = hnswlib.Index(space        = 'ip',
                                   dim          = self._data.shape[1])
        self._hnsw.init_index(max_elements      = self._data.shape[0],
                              M                 = self._M,
                              ef_construction   = self._ef_construction)
        self._hnsw.add_items(data               = self._data)

    def reconstruct_HNSW(self):
        self._hnsw.init_index(max_elements      = len(self._data),
                              M                 = self._M,
                              ef_construction   = self._ef_construction)
        self._hnsw.add_items(data               = self._data)

    def calculate_knn(self, qs, k=10):
        scores = qs @ (self._data).T

        topk = np.argpartition(
            -scores,
            kth=k - 1,
            axis=1,
        )[:, :k]

        # Optional: sort those k results by actual score
        rows = np.arange(len(qs))[:, None]
        order = np.argsort(-scores[rows, topk], axis=1)

        knn_matrix = np.take_along_axis(topk, order, axis=1)
        return knn_matrix # len(qs) x k nagyságú mátrix

    def run(self, qs, k = 10, ef_search = 40):
        # Várt eredmény (recall = 1) kiszámítása
        R_E_list = HNSW_benchmark.calculate_knn(qs, k)

        # HNSW futtatása
        self._hnsw.set_ef(ef_search)
        R_A_list, _ = self._hnsw.knn_query(qs, k=k)

        # Recall számítás
        recalls = []
        for R_A, R_E in zip(R_A_list, R_E_list):
            recalls.append(len(set(R_A) & set(R_E)) / len(R_E)) # R_E union R_A / ||R_E||

        return recalls


############################
###         LID          ###
############################
# LID@k értékek meghatározása az adathalmazban minden pontra
def calculate_lid(data, k=100):
    vectors = np.asarray(list(data), dtype=np.float32)
    n = len(vectors)

    if n <= k:
        raise ValueError(f"Dataset must contain more than k={k} vectors.")

    knn = KNN(vectors)
    lid_values = np.full(n, np.nan, dtype=np.float64)

    # LID számítása egyes vektorokra
    for i, vector in enumerate(vectors):
        # KNN számítás
        neighbor_indices = knn.get_KNN(vector, k)
        #Távolságok számítása
        distances = np.array([KNN.euclidian_distance(vector, vectors[j]) for j in neighbor_indices])

        distances = distances[distances > 0]

        if len(distances) < k:
            print(f"Point: {i}")
            print(neighbor_indices[:10])
            raise ValueError(f"Some elements deleted.")

        r_k = distances[-1]
        sum_log = np.sum(np.log2(distances / r_k)) # ln lenne helyes, de mindegy a rangsorolás szempontjából, mivel mindegyik log fv. szig. mon. növ.
        lid_values[i] = -((sum_log / k) ** -1)

    return lid_values