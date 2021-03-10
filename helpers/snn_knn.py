from numba import cuda
from sklearn.neighbors import NearestNeighbors

import numpy as np
import math

## GPU Acceleration for snn matrix Construction

@cuda.jit
def snn_kernel(knn_info, param_arr, snn_matrix):
    ## Input: raw_knn (knn info)
    ## Output: snn_matrix (snn info)
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    length = param_arr[0]
    k      = param_arr[1]
    if i >= length or j >= length:
        return
    if i == j:
        snn_matrix[i, j] = 0
        return
    c = 0
    for m in range(k):
        for n in range(k):
            if knn_info[i, m] == knn_info[j,n]:
                c += (k + 1 - m) * (k + 1 - n)

    snn_matrix[i, j] = c

def snn_gpu(knn_info, length, k):
    ## INPUT
    knn_info_global_mem = cuda.to_device(knn_info)
    param_arr_global_mem = cuda.to_device(np.array([length, k]))
    ## OUTPUT
    snn_matrix_global_mem = cuda.device_array((length, length))

    TPB = 16
    tpb = (TPB, TPB)
    bpg = ((math.ceil(length / TPB), math.ceil(length / TPB)))

    snn_kernel[bpg, tpb](knn_info_global_mem, param_arr_global_mem, snn_matrix_global_mem)

    snn_matrix = snn_matrix_global_mem.copy_to_host()

    return snn_matrix

def knn_info(dist_matrix, k):
    neighbors_instance = NearestNeighbors(n_neighbors=k, metric="precomputed")
    neighbors_instance.fit(dist_matrix)
    knn_info = neighbors_instance.kneighbors(return_distance=False)
    return knn_info