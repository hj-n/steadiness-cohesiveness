from abc import ABC, abstractmethod
from . import snn_knn as sk

import numpy as np
import hdbscan

def preprocessing(strategy : str, parameter, raw_dist_matrix, emb_dist_matrix):
    cstrat = {
        "snn" : SNNCS(parameter, raw_dist_matrix, emb_dist_matrix)
    }["snn"]
    cstrat.preprocessing()
    return cstrat

def extract_cluster(strategy : str):
    pass

class ClusterStrategy(ABC):

    '''
    Saving distance matrix info and setting parameter
    '''
    def __init__(self, raw_dist_matrix, emb_dist_matrix):
        self.raw_dist_matrix = raw_dist_matrix
        self.emb_dist_matrix = emb_dist_matrix
        self.length = len(raw_dist_matrix)

    @abstractmethod
    def preprocessing(self):
        pass

    @abstractmethod
    def extract_cluster(self):
        pass
    

    '''
    Get the indices of the points which to be clustered as input
    and return the clustereing result
    '''
    @abstractmethod
    def clustering(self):
        pass

class SNNCS(ClusterStrategy):

    def __init__(self, parameter, raw_dist_matrix, emb_dist_matrix):
        super().__init__(raw_dist_matrix, emb_dist_matrix)
        self.k = parameter["k"]
        self.a = 1    # parameter for similairty => distance = 1 / 1 + similarity

    def preprocessing(self):
        # Compute knn infos
        self.raw_knn_info = sk.knn_info(self.raw_dist_matrix, self.k)
        self.emb_knn_info = sk.knn_info(self.emb_dist_matrix, self.k)
        
        # Compute snn matrix
        self.raw_snn_matrix = sk.snn_gpu(self.raw_knn_info, self.length, self.k)
        self.emb_snn_matrix = sk.snn_gpu(self.emb_knn_info, self.length, self.k)
        self.raw_snn_max    = np.max(self.raw_snn_matrix)
        self.emb_snn_max    = np.max(self.emb_snn_matrix)
        

    def extract_cluster(self, mode, walk_num):
        # Seed selection
        if mode == "steadiness":   ## extract from the embedded (projected) space
            knn_info   = self.emb_knn_info
            snn_matrix = self.emb_snn_matrix
            snn_max    = self.emb_snn_max
        if mode == "cohesiveness": ## extract from the original space
            knn_info   = self.raw_knn_info
            snn_matrix = self.raw_snn_matrix
            snn_max    = self.raw_snn_max
        seed_idx = np.random.randint(self.length)
        return sk.snn_based_cluster_extraction(knn_info, snn_matrix, snn_max, seed_idx, walk_num)



    ## HDBSCAN with precomputed distance based on snn
    def clustering(self, mode, indices):
        if mode == "steadiness":
            snn_matrix = self.raw_snn_matrix
            snn_max    = self.raw_snn_max
        if mode == "cohesiveness":
            snn_matrix = self.emb_snn_matrix
            snn_max    = self.emb_snn_max
        # print((snn_matrix[indices].T)[indices].shape)
        cluster_snn_matrix = (snn_matrix[indices].T)[indices] / snn_max

        cluster_snn_dist_matrix = 1 / (cluster_snn_matrix + self.a)
        np.fill_diagonal(cluster_snn_dist_matrix, 0)

        print(cluster_snn_dist_matrix)
        clusterer = hdbscan.HDBSCAN(metric="precomputed")
        clusterer.fit(cluster_snn_dist_matrix)
        print(clusterer.labels_)
        
        return clusterer.labels_

        # print(cluster_snn_matrix)
        # print(cluster_snn_dist_matrix)

        # for (x,y) in [(2,7), (9, 3), (91, 98), (22, 33), (50, 50)]:
        #     print(snn_matrix[indices[x],indices[y]])
        #     print(cluster_snn_matrix[x, y])

        # print(cluster_snn_matrix )
        # for 
        # print(len(snn_matrix[indices, indices]))
        
        pass
        
        