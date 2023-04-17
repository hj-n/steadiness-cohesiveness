import numba
import numpy as np

'''
Distance Matrix Calculation Accelerated by Numba
'''

##  Distance matrix calculation
@numba.njit(parallel=True)
def dist_matrix(vectors):
    distance_matrix = np.zeros((len(vectors), len(vectors)))
    for i in numba.prange(len(vectors)):
        for j in numba.prange(i + 1, len(vectors)):
            # minutely faster than np.linalg.norm
            distance_matrix[i][j] = np.sqrt(np.sum((vectors[i] - vectors[j]) ** 2))
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix

'''
k-nearest neighbor implementation with 
the further acceleration acheived by preprocessed distance matrix
'''

