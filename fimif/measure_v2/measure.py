import numpy as np
import random
import json

from sklearn.neighbors import KDTree
from pyclustering.cluster.xmeans import xmeans


class Fimif:
    def __init__(
                 self,
                 raw,     # raw data
                 emb,     # emb data
                 iteration=1000, # iteration number
                 k=6,     # for constructing knn graph
                 walk_num=500, # random walk number
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

        ## intermediate variables
        self.raw_neighbors = None
        self.emb_neighbors = None 

        ## target score
        self.score_missing = None
        self.score_false = None
        self.score = None     

        self.__initial_knn_graph_setup()
        self.__measure()


    def __measure(self):
        for i in range(self.iter):
            random_cluster = self.__random_cluster_selection(True)
            clusters = self.__find_groups(random_cluster, True)
            


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

    

file = open("./json/sphere_tsne.json", "r") 
data = json.load(file)

raw = np.array([datum["raw"] for datum in data])
emb = np.array([datum["emb"] for datum in data])

Fimif(raw, emb, iteration=1)

