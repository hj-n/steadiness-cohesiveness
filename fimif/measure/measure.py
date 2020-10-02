import numpy as np
import time
import random
from convex_hull import *
from enum import Enum
from sklearn.neighbors import KDTree

## Helper
from smallest_enclosing_circle import *



## Fimif Measure class 
class FimifMeasure:
    def __init__(
                 self, 
                 data,                         # dataset with raw, emb, (label)
                 boundary = "hyperball",       # boundary calculation : "hyperball", "convexhull", "convexhull_approx"
                 cluster_selection = "entire", # cluster selection method : select point within "entire" or "class" or "knn_based"
                 cluster_shape = "circle",     # shape of generated 2D cluster : "circle" or "convexhull"
                 k = 4,                        # k value for knn
                 cluster_seed_num = 20,        # number of seed points to form a random cluster 
                 iter = 200,                   # number of iterations (200 is enough, usually...)
                 class_num = None              # If not given, calculated
                ):
        self.data = data
        self.boundary = boundary
        self.cluster_selection = cluster_selection
        self.cluster_shape = cluster_shape
        self.k = k
        self.cluster_seed_num = cluster_seed_num
        self.iter = iter
        self.class_num = class_num

        self.size = len(data)
        self.dim  = len(data[0]["raw"])
        self.emb  = np.array([ el["emb"] for el in data ])
        self.raw  = np.array([ el["raw"] for el in data ])
        if "label" in self.data[0]:
            self.class_label = np.array([ el["label"] for el in data ])

        self.log = []
        self.avg_proportion = None

        self.raw_neighbors = None
        self.emb_neighbors = None
        self.__initial_setup()

    def __initial_setup(self):
        ## STUB should eliminate time measurement afterward

        # KNN construction for both raw and emb
        start = time.time()
        # ANCHOR should uncomment it afterward
        # self.__knn_raw()
        end = time.time()
        # print("KNN RAW construction: ", end - start)
        start = time.time()
        self.__knn_emb()
        end = time.time()
        # print("KNN EMB construction: ", end - start)

    def __knn_raw(self):
        raw_tree = KDTree(self.raw)
        neighbors = raw_tree.query(self.raw, self.k + 1, return_distance=False)
        self.raw_neighbors = neighbors[:, 1:]

    def __knn_emb(self):
        emb_tree = KDTree(self.emb)
        neighbors = emb_tree.query(self.emb, self.k + 1, return_distance=False)
        self.emb_neighbors = neighbors[:, 1:]

    # Evaluate the avg missing family proportion
    def evaluate(self):

        # start = time.time()

        # sum = 0
        # error = 0
        # for _ in range(self.iter):
        #     cluster = self.__random_cluster()
        #     back_projection = self.__backward_projected_result(cluster)
        #     cluster_set  = set(cluster)
        #     backward_set = set(back_projection)
        #     missing_families = backward_set - cluster_set
        #     if(len(backward_set) == 0):
        #         error += 1
        #         continue
        #     proportion = len(missing_families) / len(backward_set)
        #     sum += proportion
        #     self.log.append(proportion)
        

        # self.avg_proportion = sum / (self.iter - error)
        # print("Avg val:", self.avg_proportion)
        
        # end = time.time()
        # print("Elpased time for measurement:", end - start, "seconds")

        cluster = self.__random_cluster()
        back_projection = self.__backward_projected_result(cluster)
        cluster_set  = set(cluster)
        backward_set = set(back_projection)
        missing_families = backward_set - cluster_set
       
       


    def __random_cluster(self):
        
        def seeds_from_entire():
            return random.sample(range(self.size), self.cluster_seed_num)
        def seeds_from_class():
            if "label" not in self.data[0]:
                raise "Cannot use class-based random selection in current dataset"
            if self.class_num == None:
                classes = set(self.class_label)
                self.class_num = len(classes)

            current_class = random.randint(0, self.class_num - 1)

            current_class_idx_array = []
            for (idx, label) in enumerate(self.class_label):
                if label == current_class:
                    current_class_idx_array.append(idx)
            return random.sample(current_class_idx_array, self.cluster_seed_num)

        def seeds_from_knn():
            # TODO
            return []
        seed_selector = {
            "entire"   : seeds_from_entire,
            "class"    : seeds_from_class,
            "knn_based": seeds_from_knn,
        }

        def circle_cluster(seeds):
            points = [(arr[0], arr[1]) for arr in self.emb[seeds]]
            c = make_circle(points)
            cluster = []
            for idx, point in enumerate(self.emb):
                dist = math.sqrt((point[0] - c[0]) ** 2 + (point[1] - c[1]) ** 2)
                if (dist <= c[2]):
                    cluster.append(idx)
            return cluster
        def convexhull_cluster(seeds):
            # TODO
            return []
        cluster_constructor = {
            "circle"        : circle_cluster,
            "convexhull"    : convexhull_cluster,
        }

        return cluster_constructor[self.cluster_shape](seed_selector[self.cluster_selection]())

    def __backward_projected_result(self, cluster):

        def hyperball_backward():
            raw_points = self.raw[cluster]
            center_approx = raw_points.mean(axis=0)
            radius = -1
            for point in raw_points:
                dist = np.linalg.norm(point - center_approx)
                if radius < dist:
                    radius = dist
            result = []
            for idx, point in enumerate(self.raw):
                dist = np.linalg.norm(point - center_approx)
                if radius >= dist:
                    result.append(idx)

            return result
        def convexhull_backward():
            # TODO
            raw_points = self.raw[cluster].tolist()
            hull = ConvexHullWithScipy(raw_points)
            result = []
            for idx, bool_val in enumerate(hull.is_in_hull(self.raw)):
                if bool_val:
                    result.append(idx)
            return result
        def convexhull_approx_backward():
            # TODO
            return []
        
        
        boundary_selection = {
            "hyperball"        : hyperball_backward,
            "convexhull"       : convexhull_backward,
            "convexhull_approx": convexhull_approx_backward
        }
        return boundary_selection[self.boundary]()
        

    

