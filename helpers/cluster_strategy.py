from abc import ABC, abstractmethod
from . import snn_knn as sk

import numpy as np
import hdbscan

def install_strategy(strategy : str, parameter, raw_dist_matrix, emb_dist_matrix):
    cstrat = {
        "snn" : SNNCS(parameter, raw_dist_matrix, emb_dist_matrix)
    }["snn"]
    return cstrat


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
    
    '''
    Extract the clusters from the given incidices
    mode : steadiness / cohesiveness 
    '''
    @abstractmethod
    def extract_cluster(self, mode, walk_num):
        pass
    

    '''
    Get the indices of the points which to be clustered as input
    and return the clustereing result
    mode : steadiness / cohesiveness 
    '''
    @abstractmethod
    def clustering(self, mode, indices):
        pass
    
    '''
    Compute the distance between two clusters in raw / emb space
    return two distance (raw_dist, emb_dist)
    mode: steadiness / cohesiveness
    '''
    @abstractmethod
    def compute_distance(self, mode, cluster_a, cluster_b):
        pass

class SNNCS(ClusterStrategy):

    def __init__(self, parameter, raw_dist_matrix, emb_dist_matrix):
        super().__init__(raw_dist_matrix, emb_dist_matrix)
        self.k = parameter["k"]
        self.a = 0.1    # parameter for similairty => distance = 1 / 1 + similarity

    def preprocessing(self):
        # Compute knn infos
        self.raw_knn_info = sk.knn_info(self.raw_dist_matrix, self.k)
        self.emb_knn_info = sk.knn_info(self.emb_dist_matrix, self.k)
        
        # Compute snn matrix
        self.raw_snn_matrix = sk.snn_gpu(self.raw_knn_info, self.length, self.k)
        self.emb_snn_matrix = sk.snn_gpu(self.emb_knn_info, self.length, self.k)
        raw_snn_max    = np.max(self.raw_snn_matrix)
        emb_snn_max    = np.max(self.emb_snn_matrix)

        # Normalize snn matrix
        self.raw_snn_matrix /= raw_snn_max
        self.emb_snn_matrix /= emb_snn_max

        # Compute dist matrix 
        # self.raw_snn_dist_matrix =  (1 / (self.raw_snn_matrix + self.a) - 0.5) * 2
        # self.emb_snn_dist_matrix =  (1 / (self.emb_snn_matrix + self.a) - 0.5) * 2

        self.raw_snn_dist_matrix = (1 / (self.raw_snn_matrix + self.a)) - 1
        self.emb_snn_dist_matrix = (1 / (self.emb_snn_matrix + self.a)) - 1

        dissimilarity_matrix = self.raw_snn_dist_matrix - self.emb_snn_dist_matrix
        dissimilarity_max = np.max(dissimilarity_matrix)
        dissimilarity_min = np.min(dissimilarity_matrix)

        max_compress = dissimilarity_max if dissimilarity_max > 0 else 0
        min_compress = dissimilarity_min if dissimilarity_min > 0 else 0
        max_stretch  = - dissimilarity_min if dissimilarity_min < 0 else 0
        min_stretch  = - dissimilarity_max if dissimilarity_max < 0 else 0
        return max_compress, min_compress, max_stretch, min_stretch
        

    def extract_cluster(self, mode, walk_num):
        # Seed selection
        if mode == "steadiness":   ## extract from the embedded (projected) space
            knn_info   = self.emb_knn_info
            snn_matrix = self.emb_snn_matrix
        if mode == "cohesiveness": ## extract from the original space
            knn_info   = self.raw_knn_info
            snn_matrix = self.raw_snn_matrix
        seed_idx = np.random.randint(self.length)
        extracted_cluster = []
        while len(extracted_cluster) == 0:
            cluster_candidate = sk.snn_based_cluster_extraction(knn_info, snn_matrix, seed_idx, walk_num)
            if cluster_candidate.size > 1:
                extracted_cluster = cluster_candidate
        return extracted_cluster



    ## HDBSCAN with precomputed distance based on snn
    def clustering(self, mode, indices):
        if mode == "steadiness":
            snn_dist_matrix = self.raw_snn_dist_matrix
        if mode == "cohesiveness":
            snn_dist_matrix = self.emb_snn_dist_matrix

        cluster_snn_dist_matrix = (snn_dist_matrix[indices].T)[indices] 

        np.fill_diagonal(cluster_snn_dist_matrix, 0)

        clusterer = hdbscan.HDBSCAN(metric="precomputed", allow_single_cluster=True)
        clusterer.fit(cluster_snn_dist_matrix)
        
        return clusterer.labels_


    def compute_distance(self, mode, cluster_a, cluster_b):

        pair_num = cluster_a.size * cluster_b.size
        if(cluster_a.size == 1):
            cluster_a = cluster_a[0]
        if(cluster_b.size == 1):
            cluster_b = cluster_b[0]
        raw_similarity = np.sum((self.raw_snn_matrix[cluster_a].T)[cluster_b]) / pair_num
        emb_similarity = np.sum((self.emb_snn_matrix[cluster_a].T)[cluster_b]) / pair_num

        # raw_dist = (1 / (raw_similarity + self.a) - 0.5) * 2
        # emb_dist = (1 / (emb_similarity + self.a) - 0.5) * 2

        raw_dist = 1 / (raw_similarity + self.a) - 1
        emb_dist = 1 / (emb_similarity + self.a) - 1

        return raw_dist, emb_dist
    
    def __get_centroid(self, cluster, snn_matrix):
        if cluster.size == 1:
            return cluster[0]
        if cluster.size == 2:
            return cluster[np.random.randint(2)]
        ## if cluster size is bigger than 2
        cluster_snn_matrix = (snn_matrix[cluster].T)[cluster]
        cluster_snn_sum = np.sum(cluster_snn_matrix, axis = 1)
        centroid = np.argmax(cluster_snn_sum)

        return centroid

        
        