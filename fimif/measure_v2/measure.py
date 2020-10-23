import numpy as np
import random
import json

from sklearn.neighbors import KDTree
from pyclustering.cluster.xmeans import xmeans
from fimifpath import *


class Fimif:
    def __init__(
                 self,
                 raw,     # raw data
                 emb,     # emb data
                 iteration=1000, # iteration number
                 k=6,     # for constructing knn graph
                 walk_num=200, # random walk number
                 max_cluster_num=20, # max cluster num for x-means clustering
                 beta=1 # beta for F_beta score calculation
                ):
        self.raw = raw
        self.emb = emb
        self.N   = len(raw)    # number of points
        self.iter = iteration
        self.k   = k
        self.walk_num = walk_num
        self.max_cluster_num = max_cluster_num
        self.beta = beta


        ## variables for FimifPath
        self.fimifpath_list = []
        for i in range(self.N):
            self.fimifpath_list.append(FimifPath(self.emb[i][0], self.emb[i][1]))  ## one fimifPath class object per point 

        ## intermediate variables
        self.raw_neighbors = None
        self.emb_neighbors = None 
        self.dist_max_x = None    # raw space
        self.dist_max_y = None    # emb space
        self.max_mu_stretch = None
        self.min_mu_stretch = None
        self.max_mu_compress = None
        self.min_mu_compress = None 

        ## target score
        self.score_missing = None
        self.score_false = None
        self.score = None     

        self.__initial_knn_graph_setup()
        self.__initial_dist_setup()
        self.__measure()


    def __measure(self):
        x = True
        false_distortion_weight_list = []
        missing_distortion_weight_list = []
        for mode in [True, False]:
            for i in range(self.iter):
                ## for progress checking
                if i % 100 = 0:
                    print(str(i) + "-th iteration completed")

                random_cluster = self.__random_cluster_selection(mode)
                clusters = self.__find_groups(random_cluster, mode)
                current_list = self.__compute_distortion(clusters, mode)
                if mode:
                    false_distortion_weight_list += current_list
                else:
                    missing_distortion_weight_list += current_list

        false_weight_sum = 0
        false_distortion_sum = 0
        for (distortion, weight) in false_distortion_weight_list:
            false_distortion_sum += distortion * weight
            false_weight_sum += weight
        self.score_false = 1 - false_distortion_sum / false_weight_sum 
        missing_weight_sum = 0
        missing_distortion_sum = 0
        for (distortion, weight) in missing_distortion_weight_list:
            missing_distortion_sum += distortion *  weight
            missing_weight_sum += weight
        self.score_missing = 1 - missing_distortion_sum / missing_weight_sum 


        self.score = (1 + self.beta * self.beta) * ((self.score_false * self.score_missing) / (self.beta * self.beta * self.score_false + self.score_missing))
        print("False Score (Precision):", self.score_false)
        print("Missing Score  (Recall):",self.score_missing)
        print("F_beta Score:", self.score)
        



    def __random_cluster_selection(self, is_false):
        seed = random.randint(0, self.N - 1)
        random_cluster = set()
        random_cluster.add(seed)

        neighbors = self.emb_neighbors if is_false else self.raw_neighbors

        visiting = seed  # current visiting point
        for _ in range(self.walk_num):
            visiting = neighbors[visiting][random.randint(0, self.k - 1)]
            random_cluster.add(visiting)
        
        return random_cluster

    def __find_groups(self, random_cluster, is_false):
        random_cluster_list = list(random_cluster)
        xmean_instance = None
        if is_false:
            xmean_instance = xmeans(self.raw[random_cluster_list], kmax=self.max_cluster_num, tolerance=0.01)
        else:
            xmean_instance = xmeans(self.emb[random_cluster_list], kmax=self.max_cluster_num, tolerance=0.01)
        
        xmean_instance.process()
        clusters_idx = xmean_instance.get_clusters()
        clusters = []
        for cluster in clusters_idx:
            clusters.append([random_cluster_list[idx] for idx in cluster])

        return clusters

    def __compute_distortion(self, groups, is_false):
        group_num = len(groups)
        group_x = [] # ND centroid of groups
        group_y = [] # 2D centroid of groups
        for i in range(group_num):
            group_x.append(np.average(self.raw[groups[i]], axis=0))
            group_y.append(np.average(self.emb[groups[i]], axis=0))


        distortion_weight_list = []

        for i in range(group_num):
            for j in range(i):
                distortion = None
                max_mu = None
                min_mu = None
                if is_false:
                    mu_group = np.linalg.norm(group_x[i] - group_x[j]) / self.dist_max_x - np.linalg.norm(group_y[i] - group_y[j]) / self.dist_max_y
                    distortion = (mu_group - self.min_mu_compress) / (self.max_mu_compress - self.min_mu_compress) if mu_group > 0 else 0               # discard if mu_group < 0 (not compressed)
                    max_mu, min_mu = self.max_mu_compress, self.min_mu_compress
                else:
                    mu_group = - np.linalg.norm(group_x[i] - group_x[j]) / self.dist_max_x + np.linalg.norm(group_y[i] - group_y[j]) / self.dist_max_y
                    distortion = (mu_group - self.min_mu_stretch) / (self.max_mu_stretch - self.min_mu_stretch) if mu_group > 0 else 0                  # discard if mu_group < 0 (not stretched)
                    max_mu, min_mu = self.max_mu_stretch, self.min_mu_stretch
                weight = len(groups[i]) * len(groups[j])
                distortion_weight_list.append((distortion, weight))

                ## The constants which should be send to fimifPath 
                ## G_i (mine)  : len(groups[i])   // 굳이 다른 점들 좌표 다 안 받고 평균값 조정해가면서 가능함 개굿
                ## G_j (yours) : len(groups[j])
                ## max_mu, min_mu : depends on is_false
                ## ND dist : np.linalg.norm(group_x[i] - group_x[j])
                ## centroid : group_y[i] (2D centroid)
                ## false or missing ? : is_false
                ND_dist = np.linalg.norm(group_x[i] - group_x[j])
                for point_idx in groups[i]:
                    self.fimifpath_list[point_idx].add_group(len(groups[i]), len(groups[j]), max_mu, min_mu, ND_dist, group_y[i], group_y[j], is_false)
                    


        return distortion_weight_list

    def __initial_dist_setup(self):
        X = np.zeros((self.N, self.N))
        Y = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i):
                X[i][j] = np.linalg.norm(self.raw[i] - self.raw[j])
                X[j][i] = X[i][j]
                Y[i][j] = np.linalg.norm(self.emb[i] - self.emb[j])
                Y[j][i] = Y[i][j]
        self.dist_max_x = np.max(X)
        self.dist_max_y = np.max(Y)
        for path in self.fimifpath_list:
            path.add_max_dists(self.dist_max_x, self.dist_max_y)
        X = X / self.dist_max_x
        Y = Y / self.dist_max_y ## normalize
        D = X - Y 
        D_max = np.max(D)
        D_min = np.min(D)
        self.max_mu_compress = D_max
        self.min_mu_compress = 0 if D_min < 0 else D_min
        self.max_mu_stretch = -D_min
        self.min_mu_stretch = 0 if D_max > 0 else -D_max
        
    def optimize_path(self):
        
        path_list = []
        for path in self.fimifpath_list[:1000]:
            path.optimize(200)
            path_list.append(path.get_trace())
        
        
        return path_list
        
        

    

    def __initial_knn_graph_setup(self):
        ## STUB should eliminate time measurement afterward
        self.__knn_emb()
        self.__knn_raw()

    def __knn_raw(self):
        raw_tree = KDTree(self.raw)
        neighbors = raw_tree.query(self.raw, self.k + 1, return_distance=False)
        self.raw_neighbors = neighbors[:, 1:]

    def __knn_emb(self):
        emb_tree = KDTree(self.emb)
        neighbors = emb_tree.query(self.emb, self.k + 1, return_distance=False)
        self.emb_neighbors = neighbors[:, 1:]

    

def test_file(file_name):
    file = open("./json/" + file_name + ".json", "r") 
    data = json.load(file)


    raw = np.array([datum["raw"] for datum in data])
    emb = np.array([datum["emb"] for datum in data])

    print("TEST for", file_name, "data")

    fimif = Fimif(raw, emb, iteration=1000)
    path_list = fimif.optimize_path()
    with open("./../../web/src/json/" + file_name + "_path.json", "w", encoding="utf-8") as json_file:
            json.dump(path_list, json_file, ensure_ascii=False, indent=4)
    



test_file("sphere_tsne")
# test_file("spheres_pca")
# test_file("spheres_topoae")
# test_file("spheres_tsne")
# test_file("spheres_umap")
# test_file("spheres_umato")

