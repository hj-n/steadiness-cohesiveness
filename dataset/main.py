import time
import tadasets
import umap

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np
import pandas as pd
import csv

import json

PATH_TO_WEB = "../../web/src/json/"
PATH_TO_MEASURE = "./../measure/json/"
PATH_TO_MEASURE_MAP = "./../measure/map_json/"
PATH = PATH_TO_MEASURE

def sampling(original_list):
    return [datum for (i, datum) in enumerate(original_list) if i % 2 == 0]

# image, label = mnist_test()
# data = [np.array(datum).flatten() for datum in image]
# data = np.array(sampling(data))
# label = np.array(sampling(label))

# start = time.time()
# # emb_tsne = UmapEmbedding("mnist_sampled_2", data, label=label)
# emb_pca = PcaEmbedding("mnist_sampled_2", data, label=label)
# end   = time.time()
# hp.print_time_spent(start, end, emb_pca.get_info())
# emb_pca.print_file(path=PATH)
# emb_pca.print_file(path=PATH_TO_MEASURE_MAP)



'''
image, label = mnist_test()
data = [np.array(datum).flatten() for datum in image]
data = np.array(sampling(data))
for p in [1, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]:
    start = time.time()
    emb_tsne = TsneEmbedding("mnist_test_" + str(p), data, label=label, metric="cosine", perplexity=p)
    end   = time.time()
    hp.print_time_spent(start, end, emb_tsne.get_info())
    emb_tsne.print_file(path=PATH)
'''

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
                datum["label"] = mammoth_label[i]
                final_data.append(datum)
            with open(PATH + "mammoth_" + key_summary + "_umap.json", "w") as outfile:
                json.dump(final_data, outfile)

            
## Mammoth t-SNE dataset
# file_path = "./raw_data/mammoth/mammoth_"
# with open(file_path + 'tsne.json') as tsne_file, open(file_path + '3d.json') as raw_file, open(file_path + 'umap.json') as umap_file:
#     tsne_data = json.load(tsne_file)["projections"]
#     raw_data  = json.load(raw_file)
#     labels = json.load(umap_file)["labels"]
#     for key in tsne_data.keys():
#         key_num = key[2:]
#         file_name = "mammoth_" + key_num + "_tsne.json"
#         emb_data = tsne_data[key]
#         final_data = []
#         for (i, _) in emb_data:
#             datum = {}
#             datum["raw"] = raw_data[i]
#             datum["emb"] = emb_data[i]
#             datum["label"] = labels[i]
#             final_data.append(datum)
#         with open(PATH + file_name, "w") as outfile:
#             json.dump(final_data, outfile) 


