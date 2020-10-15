import numpy as np
import random


class Fimif:
    def __init__(
                 self,
                 raw,     # raw data
                 emb,     # emb data
                 iteration=1000, # iteration number
                 k=4,     # for constructing knn graph
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
            random_cluster = __random_cluster_selection()

    def __random_cluster_selection(self):
        seed = random.randint(0, N - 1)
        random_cluster = set()
        random_cluster.add(seed)

        visiting = seed  # current visiting point
        for _ in range(self.walk_num):
            visiting = self.raw_neighbors[visiting][random.randint(0, self.k - 1)]
            random_cluster.add(visiting)
        
        return random_cluster

        


    

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

    

    