import time
import tadasets
import umap

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np
import pandas as pd
import csv
import os
from sklearn.decomposition import PCA
import copy
import random

import json

PATH_TO_WEB = "../../web/src/json/"
PATH_TO_MEASURE = "./../measure/json/"
PATH_TO_MEASURE_MAP = "./../measure/map_json/"
PATH = PATH_TO_MEASURE

def sampling(original_list, ratio):
    return [datum for (i, datum) in enumerate(original_list) if i % ratio == 0]


for ratio in [10]:
    image, label = mnist_test()
    data = [np.array(datum).flatten() for datum in image]
    data = np.array(sampling(data, ratio))
    label = np.array(sampling(label, ratio))

    start = time.time()
    # emb_tsne = TsneEmbedding("mnist_sampled_" + str(ratio), data, label=label)
    emb_pca = UmapEmbedding("mnist_sampled_" + str(ratio), data, label=label)
    end   = time.time()
    hp.print_time_spent(start, end, emb_pca.get_info())
    emb_pca.print_file(path=PATH)
    emb_pca.print_file(path=PATH_TO_MEASURE_MAP)



### cubic data generation


# def fill_space(start, x_direction, y_direction, x_move, y_move, accumulator):
#     for x in range(x_move):
#         for y in range(y_move):
#             current = copy.deepcopy(start)
#             x_random = random.random() * 0.5 - 0.25
#             y_random = random.random() * 0.5 - 0.25
#             current[0] += x_direction[0] * (x + x_random)  + y_direction[0] * (y + y_random)
#             current[1] += x_direction[1] * (x + x_random) + y_direction[1] * (y + y_random)
#             current[2] += x_direction[2] * (x + x_random) + y_direction[2] * (y + y_random)
#             accumulator.append(current)

# accumulator = []
# fill_space([0, 0, 0], [1, 0, 0], [0, 0, 1], 10, 11, accumulator)
# fill_space([10, 0, 0], [0, 1, 0], [0, 0, 1], 10, 11, accumulator)
# fill_space([10, 10, 0], [-1, 0, 0], [0, 0, 1], 10, 11, accumulator)
# fill_space([0, 10, 0], [0, -1, 0], [0, 0, 1], 10, 11, accumulator)
# fill_space([1, 1, 0], [1, 0, 0], [0, 1, 0], 9, 9, accumulator)

# # print(accumulator)

# accumulator = np.array(accumulator)
# emb_pca = PcaEmbedding("open_cubic", accumulator)
# emb_pca.print_file(path=PATH)
# emb_pca.print_file(path=PATH_TO_MEASURE_MAP)

    





# image, label = fashion_mnist_test()
# data = [np.array(datum).flatten() for datum in image]
# data = np.array(sampling(data, 2))
# label = np.array(sampling(label, 2))
# start = time.time()
# emb_tsne = PcaEmbedding("fmnist_sampled_2", data, label=label)
# end   = time.time()
# hp.print_time_spent(start, end, emb_tsne.get_info())
# emb_tsne.print_file(path=PATH)
# emb_tsne.print_file(path=PATH_TO_MEASURE_MAP)


# def load_kmnist(path, dtype="kmnist", kind='test'):
#     images_path = os.path.join(path, f'{dtype}-{kind}-imgs.npz')
#     labels_path = os.path.join(path, f'{dtype}-{kind}-labels.npz')
#     images = np.load(images_path)
#     images = images.f.arr_0
#     images = images.reshape(images.shape[0], -1)
#     labels = np.load(labels_path)
#     labels = labels.f.arr_0
#     labels = labels.reshape(-1)
#     return images, labels

# images, labels = load_kmnist("./raw_data/kmnist_test")
# data = [np.array(datum).flatten() for datum in images]
# data = np.array(sampling(data, 2))
# labels = np.array(sampling(labels, 2))
# start = time.time()
# emb_tsne = PcaEmbedding("kmnist_sampled_2", data, label=labels)
# end   = time.time()
# hp.print_time_spent(start, end, emb_tsne.get_info())
# emb_tsne.print_file(path=PATH)


## for test
# pca = PCA(n_components=2)
# pca.fit_transform(data)
# print(pca.explained_variance_ratio_)

## Spheres data generation for final test data extraction (umap)
# spheres_data = list(csv.reader(open("./raw_data/spheres/raw.csv")))[1:]
# spheres_raw_data = np.array([datum[:-1] for datum in spheres_data])
# spheres_label = np.array([datum[-1] for datum in spheres_data])

# for n in [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
#     for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
#         final_data = [] 
#         key_summary = str(n) + "_" + str(int(p * 100))
#         umap_instance = umap.UMAP(n_neighbors=n, min_dist=p)
#         spheres_emb_data =umap_instance.fit_transform(spheres_raw_data)
#         for (i, _) in enumerate(spheres_emb_data):
#             datum = {}
#             datum["raw"] = spheres_raw_data[i].tolist()
#             datum["emb"] = spheres_emb_data[i].tolist()
#             datum["label"] = spheres_label[i]
#             final_data.append(datum)
#         print("UMAP for", "spheres", key_summary, "finished!!")
#         with open(PATH + "spheres_" + key_summary + "_umap.json", "w") as outfile:
#             json.dump(final_data, outfile)


# with open('./raw_data/mammoth/mammoth_3d_50k.json') as json_file:
#     mammoth_raw_data = sampling(np.array(json.load(json_file)), 5)
#     for p in [0.0, 0.5, 1, 1.5, 2, 2.5, 3]:
#         final_data = []
#         key_summary = "15_" + str(int(p * 100))
#         umap_instance = umap.UMAP(n_neighbors=15, min_dist=p)
#         mammoth_emb_data = umap_instance.fit_transform(mammoth_raw_data)
#         for (i, _) in enumerate(mammoth_emb_data):
#                 datum = {}
#                 datum["raw"] = mammoth_raw_data[i].tolist()
#                 datum["emb"] = mammoth_emb_data[i].tolist()
#                 final_data.append(datum)
#         print("UMAP for", "mammoth", key_summary, "finished!!")
#         with open(PATH + "mammoth_50k_" + key_summary + "_umap.json", "w") as outfile:
#             json.dump(final_data, outfile)


'''
## Mammoth data generation for final test data extraction (umap)

with open('./raw_data/mammoth/mammoth_umap.json') as json_file:
    json_data = json.load(json_file)
    mammoth_raw_data = np.array(json_data["3d"])
    mammoth_label = np.array(json_data["labels"])
    for n in [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            final_data = [] 
            key_summary = str(n) + "_" + str(int(p * 100))
            umap_instance = umap.UMAP(n_neighbors=n, min_dist=p)
            mammoth_emb_data = umap_instance.fit_transform(mammoth_raw_data)
            for (i, _) in enumerate(mammoth_emb_data):
                datum = {}
                datum["raw"] = mammoth_raw_data[i].tolist()
                datum["emb"] = mammoth_emb_data[i].tolist()
                datum["label"] = int(mammoth_label.tolist()[i])
                final_data.append(datum)
            print("UMAP for", "mammoth", key_summary, "finished!!")
            with open(PATH + "mammoth_" + key_summary + "_umap.json", "w") as outfile:
                json.dump(final_data, outfile)


'''

# Mammoth UMAP dataset
# file_path = "./raw_data/mammoth/mammoth_"
# with open(file_path + 'tsne.json') as tsne_file, open(file_path + '3d.json') as raw_file, open(file_path + 'umap.json') as umap_file:
#     umap_data  = json.load(umap_file)
#     labels = umap_data["labels"]
#     raw_data = umap_data["3d"]
#     for key in umap_data["projections"].keys():
#         # print(key)
#         nd = key.replace("n=","").replace("d=","").split(",")
#         # print(nd)
#         n = str(nd[0])
#         d = str(int(float(nd[1]) * 100))
#         key_str = n + "_" + d
#         file_name = "mammoth_" + key_str + "_umap.json"
#         print(file_name)
#         emb_data = umap_data["projections"][key]
#         final_data = []
#         for (i, _) in emb_data:
#             datum = {}
#             datum["raw"] = raw_data[i]
#             datum["emb"] = emb_data[i]
#             datum["label"] = labels[i]
#             final_data.append(datum)
#         with open(PATH + file_name, "w") as outfile:
#             json.dump(final_data, outfile) 


