## Helper functions for the test.ipynb

import json

def make_path(dataset_path, dataset, method, sample_rate, parameter_list):
    path = dataset_path + "/" + dataset + "/" + method + "/" + str(sample_rate) + "/"
    for param in parameter_list:
        path += str(param) + "/"
    return path


def get_data(path):
    emb_data   = None
    raw_data   = None
    label_data = None
    with open(path + "emb.json") as emb_json:
        emb_data = json.load(emb_json)
    with open(path + "raw.json") as raw_json:
        raw_data = json.load(raw_json)
    try:
        with open(path + "label.json") as label_json:
            label_data = json.load(label_json)
    except:
        print("No label data exits")
    return emb_data, raw_data, label_data