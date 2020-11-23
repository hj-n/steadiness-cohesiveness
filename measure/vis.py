import json
import numpy as np
from sklearn.neighbors import KDTree


def get_edges_info(edges,data):
    edges_info = {}
    for (start, end) in edges:
        start_keys = set(data[start].keys())
        end_keys = set(data[end].keys())
        common_keys = list(start_keys.intersection(end_keys))
        if len(common_keys) == 0:
            continue
        acc = 0
        for key in common_keys:
            acc += (data[start][key] + data[end][key]) / 2 
        
        edges_info[str(start) + "_" + str(end)] = acc
    values = edges_info.values()
    max_value = max(values)
    for key in edges_info.keys():
        edges_info[key] /= max_value

    return edges_info

def get_points_info(data):
    # for checkviz visualization
    score = knn_based_measure(data, 30)
    
    if "label" in data[0]:
        return [{"coor": datum["emb"], "label": datum["label"], "cont": score[i][0], "trust": score[i][1]} for (i,datum)in enumerate(data)]
    else:
        return [{"coor": datum["emb"], "cont": score[i][0], "trust": score[i][1]} for (i,datum)in enumerate(data)] 


# for checkviz visualization
# show trustworthiness / continuity score per point
def knn_based_measure(data, k):
    raw = []
    emb = []
    for datum in data:
        raw.append(datum["raw"])
        emb.append(datum["emb"])
    raw_tree = KDTree(np.array(raw))
    raw_neighbors = raw_tree.query(raw, k + 1, return_distance=False)
    raw_neighbors = raw_neighbors[:, 1:]
    emb_tree = KDTree(np.array(emb))
    emb_neighbors = emb_tree.query(emb, k + 1, return_distance=False)
    emb_neighbors = emb_neighbors[:, 1:]
    score = []
    k_sum = (k * (k+1)) / 2
    for i, _ in enumerate(raw_neighbors):
        raw_n_set = set(raw_neighbors[i].tolist())
        emb_n_set = set(emb_neighbors[i].tolist())
        cont_num = k_sum
        for (i, idx) in enumerate(raw_neighbors[i].tolist()):
            if idx not in emb_n_set:
                cont_num += (i + 1 - k)
        trust_num = k_sum
        for (i, idx) in enumerate(emb_neighbors[i].tolist()):
            if idx not in raw_n_set:
                trust_num += (i + 1 - k)
        ## push [cont, trust]
        score.append([cont_num / k_sum, trust_num / k_sum])
    return score
                
    


def visualization(file_name, save_missing_edges):
    file = open("./map_json/" + file_name + ".json", "r") 
    data = json.load(file)
    file = open("./map_json/" + file_name + "_false.json", "r")
    false_data = json.load(file)
    file = open("./map_json/" + file_name + "_missing.json", "r")
    missing_data = json.load(file)
    file = open("./map_json/" + file_name + "_knn.json", "r")
    knn_data = json.load(file)

    edges = []
    for i, neighbors in enumerate(knn_data):
        for j in neighbors:
            edges.append((i, j))
    edges_missing_info = get_edges_info(edges, missing_data)
    edges_false_info = get_edges_info(edges, false_data)

    edge_keys = set(edges_false_info.keys()).union(set(edges_missing_info.keys()))

    edge_vis_infos = []
    for key in list(edge_keys):
        nodes = key.split("_")

        edge_vis_infos.append({
            "start": nodes[0],
            "end": nodes[1],
            "missing_val": edges_missing_info[key] if key in edges_missing_info else 0,
            "false_val": edges_false_info[key] if key in edges_false_info else 0
        })

    point_vis_infos = get_points_info(data)

    

    # missing_edges = get_missing_edges(missing_data, point_vis_infos)
    # print(len(missing_edges))
    max_missing_val = 0
    for missing_datum in missing_data:
        for key in missing_datum:
            max_missing_val = max_missing_val if missing_datum[key] < max_missing_val else missing_datum[key]
    
    for missing_datum in missing_data:
        for key in missing_datum:
            missing_datum[key] /= max_missing_val
    
    with open("./vis/" + file_name + "_edges.json", "w") as outfile:
        json.dump(edge_vis_infos, outfile)
    with open("./vis/" + file_name + "_points.json", "w") as outfile:
        json.dump(point_vis_infos, outfile)
    with open("./vis/" + file_name + "_missing_points.json", "w") as outfile:
        json.dump(missing_data, outfile)
    # if save_missing_edges:
    #     with open("./vis/" + file_name + "_missing_edges.json", "w") as outfile:
    #         json.dump(missing_edges, outfile)


# visualization("kmnist_sampled_2_pca", True)
# visualization("fmnist_sampled_2_pca", True)

# visualization("spheres_pca", True)
visualization("mnist_sampled_2_pca", True)
visualization("mnist_sampled_2_tsne", True)
visualization("mnist_sampled_2_umap", True)
visualization("mnist_sampled_2_isomap", True)
# visualization("spheres_topoae", True)
# visualization("spheres_tsne", True)
# visualization("spheres_umato", True)
# visualization("spheres_umap", True)