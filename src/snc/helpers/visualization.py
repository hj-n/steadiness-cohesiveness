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

    ## score ratio
    stead_score_ratio = (1 - stead_score) * 2 if (1 - stead_score) * 2 > 1 else 1
    cohev_score_ratio = (1 - cohev_score) * 2 if (1 - cohev_score) * 2 > 1 else 1



    edge_vis_infos = []
    for key in list(edge_keys):
        nodes = key.split("_")

        edge_vis_infos.append({
            "start": nodes[0],
            "end": nodes[1],
            "missing_val": edges_cohev_info[key] * stead_score_ratio if key in edges_cohev_info else 0,
            "false_val": edges_stead_info[key] * cohev_score_ratio if key in edges_stead_info else 0
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
        info_dict["label"] = label[i] if label != None else 0
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
        # for key in common_keys:
        #     acc += (log[start][key] + log[end][key]) / 2 
        for key in start_keys:
            acc += log[start][key]
        for key in end_keys:
            acc += log[end][key]
        edges_info[str(start) + "_" + str(end)] = acc
    values = edges_info.values()
    max_value = max(values)
    for key in edges_info.keys():
        edges_info[key] /= max_value

    return edges_info
