import json
import numpy as np
from sklearn.neighbors import KDTree
import math


def generate_visualization_data(stead_log, cohev_log, stead_score, cohev_score, label, raw, emb, k=10):

    raw_tree = KDTree(np.array(raw))
    raw_neighbors = raw_tree.query(raw, k + 1, return_distance=False)
    raw_neighbors = raw_neighbors[:, 1:]
    emb_tree = KDTree(np.array(emb))
    emb_neighbors = emb_tree.query(emb, k + 1, return_distance=False)
    emb_neighbors = emb_neighbors[:, 1:]

    ## compute trustworthiness and cohesiveness
    conti_trust = knn_based_measure(raw, emb, raw_neighbors, emb_neighbors, k)

    ## Get edge data

    edges = []
    for i, neighbors in enumerate(emb_neighbors):
        for j in neighbors:
            edges.append((i, j)) 

    edges_stead_info = get_edges_info(edges, stead_log)
    edges_cohev_info = get_edges_info(edges, cohev_log)


    edge_keys = set(edges_stead_info.keys()).union(set(edges_cohev_info.keys()))

    edge_vis_infos = []
    for key in list(edge_keys):
        nodes = key.split("_")

        edge_vis_infos.append({
            "start": nodes[0],
            "end": nodes[1],
            "missing_val": edges_cohev_info[key]  if key in edges_cohev_info else 0,
            "false_val": edges_stead_info[key]  if key in edges_stead_info else 0
        })


    ## get missing info from cohev_log
    max_missing_val = 0
    for datum in cohev_log:
        for key in datum:
            max_missing_val = max_missing_val if datum[key] < max_missing_val else datum[key]
    
    missing_log = []
    for datum in cohev_log:
        info_new_dict = {}
        for key in datum:
            info_new_dict[int(key)] = (datum[key] / max_missing_val) 
        missing_log.append(info_new_dict)
        


    ## data aggregation

    points = []
    for i, coor in enumerate(emb):
        info_dict = {}
        info_dict["coor"] = coor.tolist()
        info_dict["cont"] = conti_trust[i][0]
        info_dict["trust"] = conti_trust[i][1]
        info_dict["label"] = label[i]
        points.append(info_dict)

    return points, missing_log, edge_vis_infos
    
  

# for checkviz visualization
# show trustworthiness / continuity score per point
def knn_based_measure(raw, emb, raw_neighbors, emb_neighbors, k):

    score = []
    k_sum = (k * (k+1)) / 2
    for i, _ in enumerate(raw_neighbors):
        raw_n_set = set(raw_neighbors[i].tolist())
        emb_n_set = set(emb_neighbors[i].tolist())
        cont_num = k_sum
        for (ii, idx) in enumerate(raw_neighbors[i].tolist()):
            if idx not in emb_n_set:
                cont_num += (ii + 1 - k)
        trust_num = k_sum
        for (ii, idx) in enumerate(emb_neighbors[i].tolist()):
            if idx not in raw_n_set:
                trust_num += (ii + 1 - k)
        ## push [cont, trust]
        score.append([cont_num / k_sum, trust_num / k_sum])
    return score
                

def get_edges_info(edges, log):
    edges_info = {}
    for (start, end) in edges:
        start_keys = set(log[start].keys())
        end_keys = set(log[end].keys())
        common_keys = list(start_keys.intersection(end_keys))
        if len(common_keys) == 0:
            continue
        acc = 0
        for key in common_keys:
            acc += (log[start][key] + log[end][key]) / 2 
        acc /= len(common_keys)
        edges_info[str(start) + "_" + str(end)] = acc
    # values = edges_info.values()
    # # max_value = max(values)
    # # for key in edges_info.keys():
    # #     edges_info[key] /= max_value

    return edges_info

# def get_points_info(data):
#     # for checkviz visualization
#     score = knn_based_measure(data, 5)
    
#     if "label" in data[0]:
#         return [{"coor": datum["emb"], "label": datum["label"], "cont": score[i][0], "trust": score[i][1]} for (i,datum)in enumerate(data)]
#     else:
#         return [{"coor": datum["emb"], "cont": score[i][0], "trust": score[i][1]} for (i,datum)in enumerate(data)] 



    


# def visualization(file_name, save_missing_edges):

#     shepard_diagram(file_name)

#     file = open("./map_json/" + file_name + ".json", "r") 
#     data = json.load(file)
#     file = open("./map_json/" + file_name + "_false.json", "r")
#     false_data = json.load(file)
#     file = open("./map_json/" + file_name + "_missing.json", "r")
#     missing_data = json.load(file)
#     file = open("./map_json/" + file_name + "_knn.json", "r")
#     knn_data = json.load(file)

#     edges = []
#     for i, neighbors in enumerate(knn_data):
#         for j in neighbors:
#             edges.append((i, j))
#     edges_missing_info = get_edges_info(edges, missing_data)
#     edges_false_info = get_edges_info(edges, false_data)

#     edge_keys = set(edges_false_info.keys()).union(set(edges_missing_info.keys()))

#     edge_vis_infos = []
#     for key in list(edge_keys):
#         nodes = key.split("_")

#         edge_vis_infos.append({
#             "start": nodes[0],
#             "end": nodes[1],
#             "missing_val": edges_missing_info[key] if key in edges_missing_info else 0,
#             "false_val": edges_false_info[key] if key in edges_false_info else 0
#         })

#     point_vis_infos = get_points_info(data)

    

#     # missing_edges = get_missing_edges(missing_data, point_vis_infos)
#     # print(len(missing_edges))
#     max_missing_val = 0
#     for missing_datum in missing_data:
#         for key in missing_datum:
#             max_missing_val = max_missing_val if missing_datum[key] < max_missing_val else missing_datum[key]
    
#     for missing_datum in missing_data:
#         for key in missing_datum:
#             missing_datum[key] /= max_missing_val
    
#     with open("./vis/" + file_name + "_edges.json", "w") as outfile:
#         json.dump(edge_vis_infos, outfile)
#     with open("./vis/" + file_name + "_points.json", "w") as outfile:
#         json.dump(point_vis_infos, outfile)
#     with open("./vis/" + file_name + "_missing_points.json", "w") as outfile:
#         json.dump(missing_data, outfile)
#     # if save_missing_edges:
#     #     with open("./vis/" + file_name + "_missing_edges.json", "w") as outfile:
#     #         json.dump(missing_edges, outfile)


# def shepard_diagram(file_name):
#     file = open("./map_json/" + file_name + ".json", "r") 
#     data = json.load(file)

#     raw = []
#     emb = []
#     for datum in data:
#         raw.append(datum["raw"])
#         emb.append(datum["emb"])

#     result = {}
#     raw_max = -1
#     emb_max = -1
#     for i in range(len(raw)):
#         for j in range(i):
#             raw_dist = np.linalg.norm(np.array(raw[i]) - np.array(raw[j]))
#             raw_max = raw_max if raw_max > raw_dist else raw_dist
#             emb_dist = np.linalg.norm(np.array(emb[i]) - np.array(emb[j]))
#             emb_max = emb_max if emb_max > emb_dist else emb_dist
#             result[str(i) + "_" + str(j)] = [emb_dist, raw_dist]

#     shepard = {}
#     for i in range(20):
#         for j in range(20):
#             shepard[str(i) +"_" + str(j)] = 0
            

#     for i in range(len(raw)):
#         for j in range(i):
#             result[str(i) + "_" + str(j)] = [result[str(i) + "_" + str(j)][0] / emb_max, result[str(i) + "_" + str(j)][1] / raw_max]
#             shepard_emb_idx = math.floor(result[str(i) + "_" + str(j)][0] * 20)
#             shepard_raw_idx = math.floor(result[str(i) + "_" + str(j)][1] * 20)
#             if shepard_emb_idx >= 20:
#                 shepard_emb_idx -= 1
#             if shepard_raw_idx >= 20:
#                 shepard_raw_idx -= 1
#             shepard[str(shepard_emb_idx) + "_" + str(shepard_raw_idx)] += 1
    
#     with open("./vis/" + file_name + "_shepard.json", "w") as outfile:
#         json.dump(shepard, outfile)




# # visualization("kmnist_sampled_2_pca", True)
# # visualization("fmnist_sampled_2_pca", True)
# # visualization("open_cubic_pca", True)
# # visualization("spheres_pca", True)
# for ratio in [10]:
#     visualization("mnist_sampled_" + str(ratio) + "_pca", True)
#     visualization("mnist_sampled_" + str(ratio) + "_tsne", True)
#     visualization("mnist_sampled_" + str(ratio) + "_umap", True)
#     visualization("mnist_sampled_" + str(ratio) + "_isomap", True)
# # visualization("mnist_sampled_2_pca", True)
# # visualization("mnist_sampled_2_tsne", True)
# # visualization("mnist_sampled_2_umap", True)
# # visualization("mnist_sampled_2_isomap", True)
# # visualization("spheres_topoae", True)
# # visualization("spheres_tsne", True)
# # visualization("spheres_umato", True)
# # visualization("spheres_umap", True)
#         # test_file("mnist_sampled_" + str(ratio) + "_tsne", 5, 0, 0, 0.4)