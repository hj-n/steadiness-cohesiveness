import numpy as np
import time
from convex_hull import *
from enum import Enum
from sklearn.neighbors import KDTree


## Fimif Measure class 
class FimifMeasure:
    def __init__(
                 self, 
                 data,             # dataset with raw, emb, (label)
                 boundary,         # boundary calculation : "hyperball", "convexhull", "convexhull_approx"
                 point_selection,  # point selection method : select point within "entire" or "class"
                 k = 4             # default value for k
                ):
        self.data = data
        self.emb = np.array([ el["emb"] for el in data ])
        self.raw = np.array([ el["raw"] for el in data ])
        self.boundary = boundary
        self.point_selection = point_selection
        self.k = k
        self.raw_neighbors = None
        self.emb_neighbors = None
        self.__initial_setup()

    def __initial_setup(self):
        start = time.time()
        self.__knn_raw()
        end = time.time()
        print("KNN RAW construction: ", end - start)
        start = time.time()
        self.__knn_emb()
        end = time.time()
        print("KNN EMB construction: ", end - start)
        

    def __knn_raw(self):
        raw_tree = KDTree(self.raw)
        neighbors = raw_tree.query(self.raw, self.k + 1, return_distance=False)
        self.raw_neighbors = neighbors[:, 1:]
        print(self.raw_neighbors)

    def __knn_emb(self):
        emb_tree = KDTree(self.emb)
        neighbors = emb_tree.query(self.emb, self.k + 1, return_distance=False)
        self.emb_neighbors = neighbors[:, 1:]
        print(self.emb_neighbors)

