import numpy as np
import json


from .helpers import hparam_functions as hp
from .helpers import visualization as vis

class SNC:
    def __init__(
                 self,
                 raw,                      # raw data
                 emb,                      # emb data
                 iteration=150,            # iteration number
                 walk_num_ratio=0.3,       # random walk number,
                 dist_strategy="snn",      # determines the way to compute distance 
                 dist_parameter={          # parameters used to compute distance
                     "alpha": 0.1, "k": "sqrt"
                 },        
                 dist_function=None,       # inject predefined distance function
                 cluster_strategy="dbscan", # determines the way to consider clusters
                 snn_knn_matrix=None,  # inject predefined similarity matrix (dist_strategy should be "inject_snn")
                ):
        self.raw  = np.array(raw, dtype=np.float64)
        self.emb  = np.array(emb, dtype=np.float64)
        self.N    = len(raw)    # number of points
        self.iter = iteration
        self.walk_num = int(self.N * walk_num_ratio)

        self.dist_strategy    = dist_strategy
        if ("k" not in dist_parameter) or (dist_parameter["k"] == "sqrt"):
          dist_parameter["k"] = int(np.sqrt(self.N))

        self.dist_parameter   = dist_parameter
        self.dist_function    = dist_function
        self.cluster_strategy = cluster_strategy
        self.snn_knn_matrix   = snn_knn_matrix

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

        ## Check whether the log is recorded
        self.finished_stead = False
        self.finished_cohev = False
        


    def fit(self, record_vis_info=False):
        self.max_compress = None
        self.min_compress = None
        self.max_stretch  = None
        self.min_stretch  = None
        
        self.record = record_vis_info

        self.cstrat = hp.install_hparam(
            self.dist_strategy, self.dist_parameter, self.dist_function,
            self.cluster_strategy, self.snn_knn_matrix,
            self.raw, self.emb
        )
        
        self.max_compress, self.min_compress, self.max_stretch, self.min_stretch = self.cstrat.preprocessing()

        

    def steadiness(self):
        self.stead_score = self.__measure("steadiness", self.max_compress, self.min_compress)
        self.finished_stead = True
        return self.stead_score

    
    def cohesiveness(self):
        self.cohev_score = self.__measure("cohesiveness", self.max_stretch, self.min_stretch)
        self.finished_cohev = True
        return self.cohev_score   

    def vis_info(self, file_path=None, label=None, k=10):
        ## Exception handling
        if not self.record:
            raise Exception("The record_vis_info flag currently has 'False' value.")
        if not self.finished_stead:
            raise Exception("Please compute steadiness before extracting visualization infos")
        if not self.finished_cohev:
            raise Exception("Please compute cohesiveness before extracting visualization infos")
      



        for datum_log in self.stead_log:
            for key_idx in datum_log:
                datum_log[key_idx] = datum_log[key_idx][0] / datum_log[key_idx][1]
        for datum_log in self.cohev_log:
            for key_idx in datum_log:
                datum_log[key_idx] = datum_log[key_idx][0] / datum_log[key_idx][1]

        points, missing_log, edge_vis_infos, vertices_vis_infos = vis.generate_visualization_data( 
            self.stead_log, self.cohev_log, 
            self.stead_score, self.cohev_score, label, 
            self.raw, self.emb, k
        )

        if file_path == None:
            return points, missing_log, edge_vis_infos, vertices_vis_infos
        else:
            if file_path[-1] == "/":
                file_path += "info.json"
            elif not (file_path[-5:] == ".json"):
                file_path += ".json"
            with open(file_path, "w") as file:
                json.dump({
                    "points": points,
                    "missing_info": missing_log,
                    "edge_info": edge_vis_infos,
                    "vertices_info": vertices_vis_infos
                }, file)

    


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
   

        