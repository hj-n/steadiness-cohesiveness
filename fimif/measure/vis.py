import json


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
    if "label" in data[0]:
        return [{"coor": datum["emb"], "label": datum["label"]} for datum in data]
    else:
        return [{"coor": datum["emb"]} for datum in data]


def visualization(file_name):
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

    
    with open("./vis/" + file_name + "_edges.json", "w") as outfile:
        json.dump(edge_vis_infos, outfile)
    with open("./vis/" + file_name + "_points.json", "w") as outfile:
        json.dump(point_vis_infos, outfile)
    

visualization("mnist_sampled_10_pca")