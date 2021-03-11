import numpy as np
import random
import json
import hdbscan
import numba
import sys

from sklearn.neighbors import KDTree
from pyclustering.cluster.xmeans import xmeans
from sncvis import SNCVis
from helpers import distance_matrix as dm
from helpers import cluster_strategy as cs
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

class SNC:
    def __init__(
                 self,
                 raw,                      # raw data
                 emb,                      # emb data
                 iteration=100,           # iteration number
                 walk_num_ratio=1,             # random walk number,
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
        self.missing_log = [] 
        for __ in range(self.N):
            new_dict = { "value": [], "idx": [],}
            self.missing_log.append(new_dict)
        self.false_log = []
        
        for __ in range(self.N):
            new_dict = {"value": [], "direction": [], "idx": [] }
            self.false_log.append(new_dict)


    def fit(self):

        ## distance matrix
        self.raw_dist_matrix = dm.dist_matrix_gpu(self.raw)
        self.emb_dist_matrix = dm.dist_matrix_gpu(self.emb)

        ## normalizing
        self.raw_dist_max = np.max(self.raw_dist_matrix)
        self.emb_dist_max = np.max(self.emb_dist_matrix)
        
        self.raw_dist_matrix /= self.raw_dist_max
        self.emb_dist_matrix /= self.emb_dist_max 


        self.max_compress = None
        self.min_compress = None
        self.max_stretch  = None
        self.min_stretch  = None

        self.cstrat = cs.install_strategy(self.cluster_strategy, self.cluster_parameter, 
                                          self.raw_dist_matrix, self.emb_dist_matrix)
        
        self.max_compress, self.min_compress, self.max_stretch, self.min_stretch = self.cstrat.preprocessing()
        

    def steadiness(self):
        self.stead_score = self.__measure("steadiness", self.max_compress, self.min_compress)
        return self.stead_score

    
    def cohesiveness(self):
        self.cohev_score = self.__measure("cohesiveness", self.max_stretch, self.min_stretch)
        return self.cohev_score   

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
                distance = raw_dist - emb_dist
                if(distance <= 0):
                    continue
                distortion = (distance - min_val) / (max_val - min_val)
                weight = len(separated_clusters[i]) * len(separated_clusters[j])
                partial_distortion_sum += distortion * weight
                partial_weight_sum += weight

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



    # Legacy code reference for the visualization
    # def __compute_distortion(self, groups, is_false):
    #     group_num = len(groups)
    #     group_x = [] # ND centroid of groups
    #     group_y = [] # 2D centroid of groups
    #     for i in range(group_num):
    #         group_x.append(np.average(self.raw[groups[i]], axis=0))
    #         group_y.append(np.average(self.emb[groups[i]], axis=0))


    #     distortion_weight_list = []

    #     for i in range(group_num):
    #         for j in range(i):
    #             distortion = None
    #             max_mu = None
    #             min_mu = None
    #             if is_false:
    #                 mu_group = np.linalg.norm(group_x[i] - group_x[j]) / self.dist_max_x - np.linalg.norm(group_y[i] - group_y[j]) / self.dist_max_y
    #                 distortion = (mu_group - self.min_mu_compress) / (self.max_mu_compress - self.min_mu_compress) if mu_group > 0 else 0               # discard if mu_group < 0 (not compressed)
    #                 # max_mu, min_mu = self.max_mu_compress, self.min_mu_compress
    #             else:
    #                 mu_group = - np.linalg.norm(group_x[i] - group_x[j]) / self.dist_max_x + np.linalg.norm(group_y[i] - group_y[j]) / self.dist_max_y
    #                 distortion = (mu_group - self.min_mu_stretch) / (self.max_mu_stretch - self.min_mu_stretch) if mu_group > 0 else 0                  # discard if mu_group < 0 (not stretched)
    #                 # max_mu, min_mu = self.max_mu_stretch, self.min_mu_stretch
    #             weight = len(groups[i]) * len(groups[j])
    #             distortion_weight_list.append((distortion, weight))

    #             weighted_distortion = distortion * weight
    #             if weighted_distortion > 0:

    #                 ## ADDING Distortion info to each points (to aggregate in the future)
    #                 for idx in groups[i]:
    #                     if is_false:
    #                         point_y = self.emb[idx]
    #                         direction = group_y[j] - point_y
    #                         # direction = direction / np.linalg.norm(direction)
    #                         # self.false_log[idx]["direction"].append(-direction)
    #                         self.false_log[idx]["value"].append(distortion * weight)
    #                         self.false_log[idx]["idx"].append(groups[j])
    #                     else:
    #                         self.missing_log[idx]["value"].append(distortion * weight)
    #                         self.missing_log[idx]["idx"].append(groups[j])

                    

    #                 for idx in groups[j]:
    #                     if is_false:
    #                         point_y = self.emb[idx]
    #                         direction = group_y[i] - point_y
    #                         # direction = direction / np.linalg.norm(direction)
    #                         # self.false_log[idx]["direction"].append(-direction)
    #                         self.false_log[idx]["value"].append(distortion * weight)
    #                         self.false_log[idx]["idx"].append(groups[i])
    #                     else:
    #                         self.missing_log[idx]["value"].append(distortion * weight)
    #                         self.missing_log[idx]["idx"].append(groups[i])


    #     return distortion_weight_list


        