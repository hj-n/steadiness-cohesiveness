from abc import ABC, abstractmethod
from . import snn_knn as sk
from sklearn.cluster import KMeans
from pyclustering.cluster.xmeans import xmeans

import numpy as np
import hdbscan
from . import distance_matrix as dm


'''
Helper functions for generating basic infos
e.g., knn info, distance matrix...
'''

def get_euclidean_infos(raw, emb, dist_parameter, dist_function, length, k):
    raw_dist_matrix = dm.dist_matrix_gpu(raw)
    emb_dist_matrix = dm.dist_matrix_gpu(emb)
    
    raw_dist_max = np.max(raw_dist_matrix)
    emb_dist_max = np.max(emb_dist_matrix)

    raw_dist_matrix /= raw_dist_max
    emb_dist_matrix /= emb_dist_max

    raw_knn_info = sk.knn_info(raw_dist_matrix, k) 
    emb_knn_info = sk.knn_info(emb_dist_matrix, k) 

    return {
        "raw_dist_matrix" : raw_dist_matrix,
        "emb_dist_matrix" : emb_dist_matrix, 
        "raw_dist_max"    : raw_dist_max,
        "emb_dist_max"    : emb_dist_max, 
        "raw_knn"         : raw_knn_info,
        "emb_knn"         : emb_knn_info
    }

def get_predefined_infos(raw, emb, dist_parameter, dist_function, length, k):
    raw_dist_matrix = np.zeros((length, length))
    emb_dist_matrix = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            raw_dist_matrix[i, j] = dist_function(raw[i], raw[j], dist_parameter)
            emb_dist_matrix[i, j] = dist_function(emb[i], emb[i], dist_parameter)
    
    raw_dist_max = np.max(raw_dist_matrix)
    emb_dist_max = np.max(emb_dist_matrix)

    raw_dist_matrix /= raw_dist_max
    emb_dist_matrix /= emb_dist_max

    raw_knn_info = sk.knn_info(raw_dist_matrix, k) 
    emb_knn_info = sk.knn_info(emb_dist_matrix, k) 

    return {
        "raw_dist_matrix" : raw_dist_matrix,
        "emb_dist_matrix" : emb_dist_matrix, 
        "raw_dist_max"    : raw_dist_max,
        "emb_dist_max"    : emb_dist_max, 
        "raw_knn"         : raw_knn_info,
        "emb_knn"         : emb_knn_info
    }




def get_snn_infos(raw, emb, dist_parameter, dist_function, length, k):

    infos = get_euclidean_infos(raw, emb, dist_parameter, dist_function, length, k)
    
    # Compute snn matrix
    raw_snn_matrix = sk.snn_gpu(infos["raw_knn"], length, k)
    emb_snn_matrix = sk.snn_gpu(infos["emb_knn"], length, k)
    raw_snn_max    = np.max(raw_snn_matrix)
    emb_snn_max    = np.max(emb_snn_matrix)

    # normalize snn matrix
    raw_snn_matrix /= raw_snn_max
    emb_snn_matrix /= emb_snn_max

    # compute dist matrix
    raw_snn_dist_matrix = 1 / (raw_snn_matrix + dist_parameter["alpha"])
    emb_snn_dist_matrix = 1 / (emb_snn_matrix + dist_parameter["alpha"])

    # update infos
    infos["raw_dist_matrix"] = raw_snn_dist_matrix
    infos["emb_dist_matrix"] = emb_snn_dist_matrix
    infos["raw_snn_matrix"]  = raw_snn_matrix
    infos["emb_snn_matrix"]  = emb_snn_matrix

    return infos


'''
Helper functions to extract a cluster
used to extract_cluster 
'''

def get_a_cluster_snn(infos, mode, seed_idx, walk_num):
    if mode == "steadiness":   ## extract from the embedded (projected) space
        knn_info   = infos["emb_knn"]
        snn_matrix = infos["emb_snn_matrix"]
    if mode == "cohesiveness": ## extract from the original space
        knn_info   = infos["raw_knn"]
        snn_matrix = infos["raw_snn_matrix"]
    
    return sk.snn_based_cluster_extraction(knn_info, snn_matrix, seed_idx, walk_num)

def get_a_cluster_naive(infos, mode, seed_idx, walk_num):
    if mode == "steadiness":
        knn_info = infos["emb_knn"]
    if mode == "cohesiveness":
        knn_info = infos["raw_knn"]
    
    return sk.naive_cluster_extraction(knn_info, seed_idx, walk_num)


'''
Helper functions for cluster the given indices
'''

def get_clustering_dbscan(dist_matrix, data, indices, dist_parameter):
    cluster_dist_matrix = (dist_matrix[indices].T)[indices]
    np.fill_diagonal(cluster_dist_matrix, 0)

    clusterer = hdbscan.HDBSCAN(metric="precomputed", allow_single_cluster=True)
    clusterer.fit(cluster_dist_matrix)

    return clusterer.labels_

def get_clustering_xmeans(dist_matrix, data, indices, dist_parameter):
    clusterer = xmeans(data[indices])
    clusterer.process()

    clusters = np.zeros(len(indices), dtype=np.int32)
    labels = clusterer.get_clusters()

    for cnum, cluster in enumerate(labels):
        for idx in cluster:
            clusters[idx] = cnum

    return clusters.tolist()

def get_clustering_kmeans(dist_matrix, data, indices, dist_parameter):
    clusterer = KMeans(n_clusters=dist_parameter["K"])
    clusterer.fit(data[indices])

    return clusterer.labels_


'''
Helper functions to compute the distances between clusters
'''

def get_snn_cluster_distance(cluster_a, cluster_b, raw, emb, infos, dist_parameter):
    pair_num = cluster_a.size * cluster_b.size
    if cluster_a.size == 1:
        cluster_a = cluster_a[0]
    if cluster_b.size == 1:
        cluster_b = cluster_b[0]
    raw_sim = np.sum(((infos["raw_snn_matrix"][cluster_a]).T)[cluster_b]) / pair_num
    emb_sim = np.sum(((infos["emb_snn_matrix"][cluster_a]).T)[cluster_b]) / pair_num

    raw_dist = 1 / (raw_sim + dist_parameter["alpha"])
    emb_dist = 1 / (emb_sim + dist_parameter["alpha"])

    return raw_dist, emb_dist

def get_euc_cluster_distance(cluster_a, cluster_b, raw, emb, infos, dist_parameter):
    a_raw_centroid, a_emb_centroid = euc_get_centroid(cluster_a, raw, emb)
    b_raw_centroid, b_emb_centroid = euc_get_centroid(cluster_b, raw, emb)

    raw_dist = np.linalg.norm(a_raw_centroid - b_raw_centroid) / infos["raw_dist_max"]
    emb_dist = np.linalg.norm(a_emb_centroid - b_emb_centroid) / infos["emb_dist_max"]

    return raw_dist, emb_dist

def euc_get_centroid(cluster, raw, emb):
    if (cluster.size == 1):
        raw_centroid = raw[cluster[0]]
        emb_centroid = emb[cluster[0]]
    else:
        raw_centroid = np.sum(raw[cluster], axis=0) / len(cluster)
        emb_centroid = np.sum(emb[cluster], axis=0) / len(cluster)
    return raw_centroid, emb_centroid

def get_predefined_cluster_distance(cluster_a, cluster_b, raw, emb, infos, dist_parameter):
    pair_num = cluster_a.size * cluster_b.size
    if cluster_a.size == 1:
        cluster_a = cluster_a[0]
    if cluster_b.size == 1:
        cluster_b = cluster_b[0]
    raw_dist = np.sum((infos["raw_dist_matrix"][cluster_a].T)[cluster_b]) / pair_num
    emb_dist = np.sum((infos["emb_dist_matrix"][cluster_a].T)[cluster_b]) / pair_num

    return raw_dist, emb_dist

'''
INSTALLING Hyperparameter functions
'''

def install_hparam(dist_strategy, dist_parameter, dist_function, cluster_strategy, raw, emb):
    get_infos = None
    get_a_cluster = None
    get_clusterinng = None 
    get_cluster_distance = None

    ## Prepare proper functions for distance strategy
    if dist_strategy == "snn": 
        get_infos = get_snn_infos
        get_a_cluster = get_a_cluster_snn
        get_cluster_distance = get_snn_cluster_distance
    elif dist_strategy == "euclidean":
        get_infos = get_euclidean_infos
        get_a_cluster = get_a_cluster_naive
        get_cluster_distance = get_euc_cluster_distance
    elif dist_strategy == "predefined":
        get_infos = get_predefined_infos
        get_a_cluster = get_a_cluster_naive
        get_cluster_distance = get_predefined_cluster_distance
    else:
        raise Exception("Wrong strategy choice!! check dist_strategy ('" + dist_strategy + "')")

    ## Prepare proper functions for cluster strategy
    if cluster_strategy == "dbscan":
        get_clustering = get_clustering_dbscan
    elif cluster_strategy == "x-means":
        get_clustering = get_clustering_xmeans
    else:
        if cluster_strategy[-5:] == "means":
            cluster_strategy_splitted = cluster_strategy.split("-")
            K_val = int(cluster_strategy_splitted[0])
            dist_parameter["K"] = K_val
            get_clustering = get_clustering_kmeans
        else:
            raise Exception("Wrong strategy choice!! check cluster_strategy ('" + cluster_strategy + "')")

    return HparamFunctions(
        raw, emb, dist_parameter, dist_function,
        get_infos, get_a_cluster, get_clustering, get_cluster_distance
    )
    

class HparamFunctions():
    '''
    Saving raw, emb info and setting parameter
    '''
    def __init__(self, raw, emb, dist_parameter, dist_function, get_infos, get_a_cluster, get_clustering, get_cluster_distance):
        self.raw = raw
        self.emb = emb
        self.length = len(self.raw)
        self.dist_parameter = dist_parameter
        self.dist_function  = dist_function
        self.k = dist_parameter["k"]

        ## Inject functions
        self.get_infos = get_infos
        self.get_a_cluster = get_a_cluster
        self.get_clustering = get_clustering
        self.get_cluster_distance = get_cluster_distance

    def preprocessing(self):
        self.infos = self.get_infos(self.raw, self.emb, self.dist_parameter, self.dist_function, self.length, self.k)
        dissim_matrix = self.infos["raw_dist_matrix"] - self.infos["emb_dist_matrix"]

        dissim_max = np.max(dissim_matrix)
        dissim_min = np.min(dissim_matrix)

        max_compress = dissim_max if dissim_max > 0 else 0
        min_compress = dissim_min if dissim_min > 0 else 0
        max_stretch  = - dissim_min if dissim_min < 0 else 0
        min_stretch  = - dissim_max if dissim_max < 0 else 0
        return max_compress, min_compress, max_stretch, min_stretch

    '''
    Extract the clusters from the given incidices
    mode : steadiness / cohesiveness 
    '''
    def extract_cluster(self, mode, walk_num):
        seed_idx = np.random.randint(self.length)
        extracted_cluster = []
        while len(extracted_cluster) == 0:
            cluster_candidate = self.get_a_cluster(self.infos, mode, seed_idx, walk_num)
            if cluster_candidate.size > 1:
                extracted_cluster = cluster_candidate
        return extracted_cluster
    

    '''
    Get the indices of the points which to be clustered as input
    and return the clustereing result
    mode : steadiness / cohesiveness 
    '''
    def clustering(self, mode, indices):
        if mode == "steadiness":
            dist_matrix = self.infos["raw_dist_matrix"]
            data = self.raw
        if mode == "cohesiveness":
            dist_matrix = self.infos["emb_dist_matrix"]
            data = self.emb
        
        return self.get_clustering(dist_matrix, data, indices, self.dist_parameter)
    
    '''
    Compute the distance between two clusters in raw / emb space
    return two distance (raw_dist, emb_dist)
    mode: steadiness / cohesiveness
    '''
    def compute_distance(self, mode, cluster_a, cluster_b):
        return self.get_cluster_distance(cluster_a, cluster_b, self.raw, self.emb, self.infos, self.dist_parameter)


