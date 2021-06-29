import numpy as np
import random
import json
import hdbscan
import numba
import sys

from sklearn.neighbors import KDTree
from pyclustering.cluster.xmeans import xmeans
from helpers import distance_matrix as dm
from helpers import cluster_strategy as cs
from concurrent.futures import ThreadPoolExecutor

class SNC:
    def __init__(
                 self,
                 raw,                      # raw data
                 emb,                      # emb data
                 iteration=200,           # iteration number
                 walk_num_ratio=0.4,             # random walk number,
                 cluster_strategy="snn",     # set the strategy for extracting cluster / clustering (snn, hdb, knn...)
                 cluster_parameter={},  # clustering paramters for current clustering method / cluster extraction
                ):
        self.raw = np.array(raw, dtype=np.float64)
        self.emb = np.array(emb, dtype=np.float64)
        self.N   = len(raw)    # number of points
        self.iter = iteration
        self.walk_num = int(self.N * walk_num_ratio)
        self.cluster_strategy = cluster_strategy
        self.cluster_parameter = cluster_parameter

        ## target score
        self.cohev_score = None
        self.stead_score = None

        ## Distortion log 
        self.stead_log = [] 
        for __ in range(self.N):
            new_dict = { }
            self.stead_log.append(new_dict)
        self.cohev_log = []
        for __ in range(self.N):
            new_dict = { }
            self.cohev_log.append(new_dict)


    def fit(self, record_vis_info=False):
        self.max_compress = None
        self.min_compress = None
        self.max_stretch  = None
        self.min_stretch  = None
        
        self.record = record_vis_info

        self.cstrat = cs.install_strategy(self.cluster_strategy, self.cluster_parameter, 
                                          self.raw, self.emb)
        
        self.max_compress, self.min_compress, self.max_stretch, self.min_stretch = self.cstrat.preprocessing()

        

    def steadiness(self):
        self.stead_score = self.__measure("steadiness", self.max_compress, self.min_compress)
        return self.stead_score

    
    def cohesiveness(self):
        self.cohev_score = self.__measure("cohesiveness", self.max_stretch, self.min_stretch)
        return self.cohev_score   

    def record_result(self):
        if self.record:
            for datum_log in self.stead_log:
                for key_idx in datum_log:
                    datum_log[key_idx] = datum_log[key_idx][0] / datum_log[key_idx][1]
            for datum_log in self.cohev_log:
                for key_idx in datum_log:
                    datum_log[key_idx] = datum_log[key_idx][0] / datum_log[key_idx][1]
            return self.stead_log, self.cohev_log
        else:
            return None, None

    def __measure(self, mode, max_val, min_val):
        distortion_sum = 0
        weight_sum = 0
        for _ in range(self.iter):
            partial_distortion_sum, partial_weight_sum = self.__measure_single_iter(mode, max_val, min_val)
            distortion_sum += partial_distortion_sum
            weight_sum += partial_weight_sum
        score = 1 - distortion_sum / weight_sum
        return score

    def __measure_single_iter(self, mode, max_val, min_val):
        cluster_indices = self.cstrat.extract_cluster(mode, self.walk_num)        
        clustering_result = self.cstrat.clustering(mode, cluster_indices)
        separated_clusters = self.__separate_cluster_labels(cluster_indices, clustering_result)
        partial_distortion_sum = 0
        partial_weight_sum = 0
        for i in range(len(separated_clusters)):
            for j in range(i):
                raw_dist, emb_dist = self.cstrat.compute_distance(mode, np.array(separated_clusters[i]), 
                                                            np.array(separated_clusters[j]))
                distance = None
                if mode == "steadiness":
                    distance = raw_dist - emb_dist
                else:
                    distance = emb_dist - raw_dist 
                if(distance <= 0):
                    continue

                distortion = (distance - min_val) / (max_val - min_val)
                weight = len(separated_clusters[i]) * len(separated_clusters[j])
                partial_distortion_sum += distortion * weight
                partial_weight_sum += weight

                if self.record == True:
                    self.__record_log(mode, distortion, weight, separated_clusters[i], separated_clusters[j])



        return partial_distortion_sum, partial_weight_sum




    def __separate_cluster_labels(self, cluster_indices, clustering_result):
        cluster_num = np.max(clustering_result) + 1
        clusters = []
        for _ in range(cluster_num):
            clusters.append([])
        for idx, cluster_idx in enumerate(clustering_result):
            if cluster_idx >= 0:
                clusters[cluster_idx].append(cluster_indices[idx])
            else:
                clusters.append([cluster_indices[idx]])
        return clusters

    def __record_log(self, mode, distortion, weight, cluster_a, cluster_b):
        log = None
        if mode == "steadiness":
            log = self.stead_log
        else:   ## if cohesiveness
            log = self.cohev_log
        
        for i in cluster_a:
            for j in cluster_b: 
                if j not in log[i]:
                    log[i][j] = [distortion * weight, 1]
                else:
                    log[i][j] = [log[i][j][0] + distortion * weight, log[i][j][1] + 1]
                
                if i not in log[j]:
                    log[j][i] = [distortion * weight, 1]
                else:
                    log[j][i] = [log[j][i][0] + distortion * weight, log[j][i][1] + 1]
   

        