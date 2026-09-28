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
        if len(data) == 0 or len(data[0]) == 0:
            raise ValueError("Bad dataset given. Count or dimension = 0")
        
        # alap adatok
        self._data = [np.array(v) for v in data]
        self._M = M
        self._ef_construction = ef_construction
        # KNN init
        self._knn = KNN(data)
        # HNSW init
        self._hnsw = hnswlib.Index(space        = 'l2',
                                   dim          = len(self._data[0]))
        self._hnsw.init_index(max_elements      = len(self._data),
                              M                 = self._M,
                              ef_construction   = self._ef_construction)
        self._hnsw.add_items(data               = self._data)

    def reconstruct_HNSW(self):
        self._hnsw.init_index(max_elements      = len(self._data),
                              M                 = self._M,
                              ef_construction   = self._ef_construction)
        self._hnsw.add_items(data               = self._data)

    def run(self, q, k = 10, ef_search = 40):
        # Várt eredmény (recall = 1) kiszámítása
        R_E = self._knn.get_KNN(q, k)

        # HNSW keresési beállítások
        self._hnsw.set_ef(ef_search)

        # HNSW futtatása és recall számítása
        R_A = self._hnsw.knn_query([q], k, filter=(lambda x: KNN.euclidian_distance(self._data[x], q) > 0))
        return len(set(R_E) & set(R_A[0][0])) / k # R_E union R_A / ||R_E||

    def run_multiple(self, qs, k = 10, ef_search = 40, stats = False):
        if stats:
            start = time()
        recalls = [self.run(q, k=k, ef_search=ef_search) for q in qs]
        if stats:
            elapsed = time() - start
            start = time()
            print(f"Minimum recall: {np.min(recalls):.4f}")
            print(f"Average recall: {np.mean(recalls):.4f}")
            print(f"Maximum recall: {np.max(recalls):.4f}")
            print(f"Total time: {elapsed:.3f} seconds")
            print(f"Average time per query: {elapsed / len(qs) * 1000:.3f} ms")
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