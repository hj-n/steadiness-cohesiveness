import numba
from sklearn.neighbors import NearestNeighbors
from collections import deque

import numpy as np
import math

@numba.njit(parallel=True)
def snn(knn_info, length, k):
    snn_matrix = np.zeros((length, length))

    for i in numba.prange(length):
        for j in numba.prange(i, length):
            c = 0
            for m in numba.prange(k):
                for n in numba.prange(k):
                    if knn_info[i, m] == knn_info[j, n]:
                        c += (k + 1 - m) * (k + 1 - n)
            snn_matrix[i, j] = c
            snn_matrix[j, i] = c
    return snn_matrix


'''
Compute KNN with precomputed distance matrix
'''
def knn_info(dist_matrix, k):
    neighbors_instance = NearestNeighbors(n_neighbors=k, metric="precomputed")
    neighbors_instance.fit(dist_matrix)
    knn_info = neighbors_instance.kneighbors(return_distance=False)
    return knn_info

'''
random cluster extraction along knn bfs with snn probability
'''

def snn_based_cluster_extraction(knn_info, snn_matrix, seed_idx, walk_num):
    
    cluster_member = set()
    cluster_member.add(seed_idx)
    current_queue = deque([seed_idx])
    
    visit_num = 0
    while visit_num < walk_num:
        i = current_queue.popleft()
        knns = knn_info[i]
        for j in knns:
            probability = 1 - snn_matrix[i, j]
            dice = np.random.rand()
            if (dice > probability):
                current_queue.append(j)
                cluster_member.add(j)
                visit_num += 1
        if not current_queue:
            break

    return np.array(list(cluster_member))

'''
random cluster extraction along knn bfs
'''

def naive_cluster_extraction(knn_info, seed_idx, walk_num):
    
    cluster_member = set()
    cluster_member.add(seed_idx)
    current_queue = deque([seed_idx])
    
    visit_num = 0
    while visit_num < walk_num:
        i = current_queue.popleft()
        knns = knn_info[i]
        for j in knns:
            current_queue.append(j)
            cluster_member.add(j)
            visit_num += 1
        if not current_queue:
            break

    return np.array(list(cluster_member))

