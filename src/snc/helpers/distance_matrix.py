from numba import cuda
import numpy as np
import math
import time

'''
Distance Matrix Calculation Accelerated by GPGPU
'''

@cuda.jit
def dist_matrix_kernel(vectors, sizes, dist_matrix):
    ## Input 1: the list of vectors
    ## Input 3: the size of each vector
    ## Output : distance matrix
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    vsize = sizes[0]
    isize = sizes[1]

    if i >= isize or j >= isize:
        return
    if i == j:
        dist_matrix[i, j] = 0
        return
    else:
        dist = 0
        for idx in range(vsize):
            dist += (vectors[i, idx] - vectors[j, idx]) ** 2
        dist_matrix[i, j] = math.sqrt(dist)
            
## GPU Accelerated Distance matrix calculation
def dist_matrix_gpu(vectors):
    vector_size = len(vectors[0])
    item_num = len(vectors)

    ## INPUT
    vectors_global_mem = cuda.to_device(vectors)
    sizes_global_mem = cuda.to_device(np.array([vector_size, item_num]))

    ## OUTPUT
    dist_matrix_global_mem = cuda.device_array((item_num, item_num))
    
    ## Run Kernel
    TPB = 16
    tpb = (TPB, TPB)   
    bpg = (math.ceil(item_num / TPB), math.ceil(item_num / TPB))

    dist_matrix_kernel[bpg, tpb](vectors_global_mem, sizes_global_mem, dist_matrix_global_mem)
    
    ## GET Result
    dist_matrix = dist_matrix_global_mem.copy_to_host()

    return dist_matrix

## CPU Distance matrix calculation (for comparison & test)
def dist_matrix_cpu(vectors):
    vector_size = len(vectors[0])
    item_num = len(vectors)

    dist_matrix = np.zeros((item_num, item_num))

    for i in range(item_num):
        for j in range(i):
            dist_matrix[i, j] = np.linalg.norm(vectors[i] - vectors[j])
            dist_matrix[j, i] = dist_matrix[i, j]

    return dist_matrix

'''
k-nearest neighbor implementation with 
the further acceleration acheived by preprocessed distance matrix
'''

