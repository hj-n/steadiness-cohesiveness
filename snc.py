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

class SNC:
    def __init__(
                 self,
                 raw,                      # raw data
                 emb,                      # emb data
                 iteration=1000,           # iteration number
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

        ## Calculate Important values for the measuring
        dissimlarity_matrix = self.raw_dist_matrix - self.emb_dist_matrix
        dissimilarity_max = np.max(dissimlarity_matrix)
        dissimilarity_min = np.min(dissimlarity_matrix)

        self.max_compress = dissimilarity_max if dissimilarity_max > 0 else 0
        self.min_compress = dissimilarity_min if dissimilarity_min > 0 else 0
        self.max_stretch  = - dissimilarity_min if dissimilarity_min < 0 else 0
        self.min_stretch  = - dissimilarity_max if dissimilarity_max < 0 else 0

        self.cstrat = cs.preprocessing(self.cluster_strategy, self.cluster_parameter, 
                                       self.raw_dist_matrix, self.emb_dist_matrix)

    def steadiness(self):
        # TODO
        cluster_indices = self.cstrat.extract_cluster("steadiness", self.walk_num)
        
        self.cstrat.clustering("steadiness", cluster_indices)
        

        return cluster_indices
        # pass
    
    def cohesiveness(self):
        # TODO
        cluster_indices = self.cstrat.extract_cluster("cohesiveness", self.walk_num)
        
        self.cstrat.clustering("cohesiveness", cluster_indices)
        pass





    def result(self):
        return {
            "steadiness" : self.stead_score,
            "cohesiveness" : self.cohev_score,
        }

    def __measure(self):
        x = True
        false_distortion_weight_list = []
        missing_distortion_weight_list = []
        for mode in [True, False]:
            for i in range(self.iter):
                ## for progress checking
                if i % 100 == 0:
                    print(str(i) + "-th iteration completed")
                    pass

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
        self.stead_score = 1 - false_distortion_sum / false_weight_sum 
        missing_weight_sum = 0
        missing_distortion_sum = 0
        for (distortion, weight) in missing_distortion_weight_list:
            missing_distortion_sum += distortion *  weight
            missing_weight_sum += weight
        self.cohev_score = 1 - missing_distortion_sum / missing_weight_sum 


        



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
        
        clusters = []
        random_cluster_list = list(random_cluster)
        
        ## parameters for hdbscan
        min_cluster_size = self.clustering_parameter["min_cluster_size"] if "min_cluster_size" in self.clustering_parameter else 5
        min_samples      = self.clustering_parameter["min_samples"] if "min_samples" in self.clustering_parameter else 1

        if is_false: 
            if self.clustering == "hdbscan":
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)#, '''allow_single_cluster=True''')
                clusterer.fit(self.raw[random_cluster_list])
                clusters_idx = {}
                for (i,label) in enumerate(clusterer.labels_):
                    if label in clusters_idx:
                        clusters_idx[label].append(i)
                    else:
                        clusters_idx[label] = [i]

                for key in clusters_idx:
                    if key == -1:
                        for single_cluster in clusters_idx[key]:
                            clusters.append([random_cluster_list[single_cluster]])
                    else:
                        cluster = clusters_idx[key]
                        clusters.append([random_cluster_list[idx] for idx in cluster])
           
            if self.clustering == "xmeans":
                max_cluster_num = self.clustering_paramter["max_cluster_num"] if "max_cluster_num" in self.clustering_parameter else 20   ## defualt value:20
                xmean_instance = xmeans(self.raw[random_cluster_list], kmax=max_cluster_num, tolerance=0.01)
                xmean_instance.process()
                clusters_idx = xmean_instance.get_clusters()
                for cluster in clusters_idx:
                    clusters.append([random_cluster_list[idx] for idx in cluster])

        else:   ## embedding clustering  default:hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            clusterer.fit(self.emb[random_cluster_list])
            clusters_idx = {}
            for (i,label) in enumerate(clusterer.labels_):
                if label in clusters_idx:
                    clusters_idx[label].append(i)
                else:
                    clusters_idx[label] = [i]

            for key in clusters_idx:
                if key == -1:
                    for single_cluster in clusters_idx[key]:
                        clusters.append([random_cluster_list[single_cluster]])
                else:
                    cluster = clusters_idx[key]
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
                    # max_mu, min_mu = self.max_mu_compress, self.min_mu_compress
                else:
                    mu_group = - np.linalg.norm(group_x[i] - group_x[j]) / self.dist_max_x + np.linalg.norm(group_y[i] - group_y[j]) / self.dist_max_y
                    distortion = (mu_group - self.min_mu_stretch) / (self.max_mu_stretch - self.min_mu_stretch) if mu_group > 0 else 0                  # discard if mu_group < 0 (not stretched)
                    # max_mu, min_mu = self.max_mu_stretch, self.min_mu_stretch
                weight = len(groups[i]) * len(groups[j])
                distortion_weight_list.append((distortion, weight))

                weighted_distortion = distortion * weight
                if weighted_distortion > 0:

                    ## ADDING Distortion info to each points (to aggregate in the future)
                    for idx in groups[i]:
                        if is_false:
                            point_y = self.emb[idx]
                            direction = group_y[j] - point_y
                            # direction = direction / np.linalg.norm(direction)
                            # self.false_log[idx]["direction"].append(-direction)
                            self.false_log[idx]["value"].append(distortion * weight)
                            self.false_log[idx]["idx"].append(groups[j])
                        else:
                            self.missing_log[idx]["value"].append(distortion * weight)
                            self.missing_log[idx]["idx"].append(groups[j])

                    

                    for idx in groups[j]:
                        if is_false:
                            point_y = self.emb[idx]
                            direction = group_y[i] - point_y
                            # direction = direction / np.linalg.norm(direction)
                            # self.false_log[idx]["direction"].append(-direction)
                            self.false_log[idx]["value"].append(distortion * weight)
                            self.false_log[idx]["idx"].append(groups[i])
                        else:
                            self.missing_log[idx]["value"].append(distortion * weight)
                            self.missing_log[idx]["idx"].append(groups[i])
                    


        return distortion_weight_list

    
    def __initial_dist_setup(self):
        result = dist_setup_helper(self.N, self.raw, self.emb)
        self.dist_max_x = result[0]
        self.dist_max_y = result[1]

        self.max_mu_compress = result[2]
        self.min_mu_compress = result[3]
        self.max_mu_stretch = result[4]
        self.min_mu_stretch = result[5]
        
    def optimize_path(self):
        
        path_list = []
        for path in self.fimifpath_list[:1000]:
            path.optimize(1000)
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
        print("K-NN graph for raw data finished!!")

    def __knn_emb(self):
        emb_tree = KDTree(self.emb)
        neighbors = emb_tree.query(self.emb, self.k + 1, return_distance=False)
        self.emb_neighbors = neighbors[:, 1:]
        print("K-NN graph for emb data finished!!")



'''


@numba.njit(
    locals = {
        "X": numba.types.float64[:, ::1],
        "Y": numba.types.float64[:, ::1],
    },
    parallel=True,
    fastmath=True
)
def dist_setup_helper(N, raw, emb):
    X = np.zeros((N, N), dtype=np.float64)
    Y = np.zeros((N, N), dtype=np.float64)
    raw = raw.astype(np.float64)
    emb = emb.astype(np.float64)
    for i in numba.prange(N):
        for j in numba.prange(i):
            X[i][j] = np.dot(raw[i] - raw[j], raw[i] - raw[j])
            X[i][j] = X[i][j] ** 0.5
            X[j][i] = X[i][j]
            Y[i][j] += (emb[i][0] - emb[j][0]) ** 2
            Y[i][j] += (emb[i][1] - emb[j][1]) ** 2
            Y[i][j] = Y[i][j] ** 0.5
            Y[j][i] = Y[i][j]
    dist_max_x = np.max(X)
    dist_max_y = np.max(Y)

    X = X / dist_max_x
    Y = Y / dist_max_y ## normalize
    D = X - Y 
    D_max = np.max(D)
    D_min = np.min(D)
    max_mu_compress = D_max
    min_mu_compress = 0 if D_min < 0 else D_min
    max_mu_stretch = -D_min
    min_mu_stretch = 0 if D_max > 0 else -D_max

    return dist_max_x, dist_max_y, max_mu_compress, min_mu_compress, max_mu_stretch, min_mu_stretch






def test_file(file_name, k, n, p, walk_ratio):
    file = open("./json/" + file_name + ".json", "r") 
    data = json.load(file)


    raw = np.array([np.array(datum["raw"]).astype(np.float64) for datum in data])
    emb = np.array([np.array(datum["emb"]).astype(np.float64) for datum in data])

    print("TEST for", file_name, "data with K=", k)
    fimif = Fimif(raw, emb, iteration=1000, walk_num_ratio=walk_ratio, k=k)

    

    SNCVis = SNCVis(fimif)
    result = fimif.result()
    print("steadiness:", result["steadiness"])
    print("cohesivness:", result["cohesiveness"])
    print("f1:",result["f1"])

    result_aggregate.append([k, n, p, result["steadiness"], result["cohesiveness"], result["f1"]])

    # for knn
    emb_tree = KDTree(emb)
    neighbors = emb_tree.query(emb, 10, return_distance=False)
    emb_neighbors = neighbors[:, 1:].tolist()
    

    with open("./map_json/" + file_name + "_false.json", "w") as outfile:
        json.dump(SNCVis.false_log_aggregated, outfile)
    
    with open("./map_json/" + file_name + "_missing.json", "w") as outfile:
        json.dump(SNCVis.missing_log_aggregated, outfile)

    with open("./map_json/" + file_name + "_knn.json", "w") as outfile:
        json.dump(emb_neighbors, outfile)






# test_file("mnist_sampled_2_umap")
# test_file("mnist_sampled_2_pca")


# for i in range(0, 15):
#     test_file("multiclass_swissroll_half_" + str(i) + "_none")


## Mammoth umap
# for n in [3, 5, 10, 15, 20, 50, 100,200]:
#     for d in [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]:
#         key_summary = str(n) + "_" + str(d)
#         test_file("mammoth_" + key_summary)

## Mammoth t-sne
# for n in [3, 5, 10, 15, 20, 50, 100, 200]:
#     test_file("mammoth_" + str(n) + "_0.1_umap", n)
# for n in [3, 5, 10, 15, 20, 50, 100, 200]:
#     print(result_aggregate[n])


args = sys.argv
data = args[1]
k    = int(args[2])
n    = int(args[3])

result_aggregate = []
if data == "spheres":
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        test_file("spheres_" + str(n) + "_" + str(int(p * 100)) + "_umap", k, n, p, 0.4)    

    print("SPHERES UMAP, k=", k, ", n=", n)
    for i in range(len(result_aggregate)):
        print(result_aggregate[i])  
elif data == "mammoth_50k":
    for p in [0.0, 0.5, 1, 1.5, 2, 2.5, 3]:
        test_file("mammoth_50k_15_" + str(int(p * 100)) + "_umap", 20, 15, p, 0.4)
elif data == "mammoth":
    for p in [0, 10, 25, 50,  80, 99]:
        test_file("mammoth_" + str(n) + "_" + str(p) + "_umap", k, n, p, 0.4)    
    print("MAMMOTH UMAP, k=", k, ", n=", nn)
    for i in range(len(result_aggregate)):
        print(result_aggregate[i])
elif data == "fmnist":
    test_file("fmnist_sampled_2_pca", 5, 0, 0, 0.4)
elif data == "kmnist":
    test_file("kmnist_sampled_2_pca", 5, 0, 0, 0.4)
elif data == "mnist":
    for ratio in [10]:
        test_file("mnist_sampled_" + str(ratio) + "_pca", 5, 0, 0, 0.4)
        test_file("mnist_sampled_" + str(ratio) + "_tsne", 5, 0, 0, 0.4)
        test_file("mnist_sampled_" + str(ratio) + "_umap", 5, 0, 0, 0.4)
        test_file("mnist_sampled_" + str(ratio) + "_isomap", 5, 0, 0, 0.4)
    
elif data == "spheres_all":
    test_file("spheres_pca", 5, 0, 0, 2000)
    test_file("spheres_atsne", 5, 0, 0, 2000)
    test_file("spheres_topoae", 5, 0, 0, 2000)
    test_file("spheres_tsne", 5, 0, 0, 2000)
    test_file("spheres_umap", 5, 0, 0, 2000)
    test_file("spheres_umato", 5, 0, 0, 2000)
elif data == "cubic":
    test_file("open_cubic_pca", 5, 0, 0, 0.4)

    
else:
    print("No such dataset!!")



result_aggregate = []
for k in [5, 10, 15, 20, 25, 30, 35, 40]:
    for n in [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            test_file("spheres_" + str(n) + "_" + str(int(p * 100)) + "_umap", k, n, p, 2000)    

print("SPHERES UMAP")
for i in range(len(result_aggregate)):
    print(result_aggregate[i])  
'''    
'''
result_aggregate = []
## Final measure for mammoth-umap
for k in [5, 10, 15, 20, 25, 30, 35, 40]:
    for n in [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            test_file("mammoth_" + str(n) + "_" + str(int(p * 100)) + "_umap", k, n, p, 4000)    

print("MAMMOTH UMAP")
for i in range(len(result_aggregate)):
    print(result_aggregate[i])
    
## reinitialize result_aggregate
result_aggregate = []


## Mammoth t-sne
# for n in [20, 50, 100, 200]:
#     test_file("mammoth_" + str(n) + "_0.25_umap", n)
# for n in [20, 50, 100, 200]:
#     print(result_aggregate[n])
# for n in [5, 10]:
#     test_file("mammoth_" + str(n) + "_tsne", n)
# for n in [5, 10]:
#     print(result_aggregate[n])
## MNIST TSNE

for i in [1, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]:
    test_file("mnist_test_" + str(i) + "_tsne")


## Spheres umap
# for n in [20, 40, 60, 80, 120, 160, 200]:
#     for d in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.99]:
#         key_summary = str(n) + "_" + str(d)
#         test_file("spheres_" + key_summary)

# for d in [0.2, 0.4]:
#     for n in [3, 10, 20, 30, 40, 50, 100, 150, 200, 400, 600, 800, 1000]:
#         key_summary = str(n) + "_" + str(d)
#         test_file("spheres_sampled_" + key_summary)
# test_file("spheres_200_0.99")

# test_file("sphere_tsne")
# test_file("sphere_umap")
# test_file("sphere_pca")
# test_file("swiss_roll_tsne")
# test_file("swiss_roll_umap")
# test_file("swiss_roll_pca")
# test_file("mnist_sampled_tsne")
# test_file("mnist_sampled_umap")
# test_file("mnist_sampled_pca")
# test_file("mnist_test_2_euclidean_tsne", 0.00002)
# test_file("mnist_test_3_euclidean_tsne", 0.00003)
# test_file("mnist_test_4_euclidean_tsne", 0.00004)
# test_file("mnist_test_5_euclidean_tsne", 0.00005)
# test_file("mnist_test_6_euclidean_tsne", 0.00006)
# test_file("mnist_test_7_euclidean_tsne", 0.00007)
# test_file("mnist_test_8_euclidean_tsne", 0.00008)
# test_file("mnist_test_9_euclidean_tsne", 0.00009)
# test_file("spheres_pca")
# test_file("spheres_topoae")
# test_file("spheres_tsne")
# test_file("spheres_umap")
# test_file("spheres_umato")

'''