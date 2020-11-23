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

import json

PATH_TO_WEB = "../../web/src/json/"
PATH_TO_MEASURE = "./../measure/json/"
PATH_TO_MEASURE_MAP = "./../measure/map_json/"
PATH = PATH_TO_MEASURE

def sampling(original_list):
    return [datum for (i, datum) in enumerate(original_list) if i % 2 == 0]

image, label = mnist_test()
data = [np.array(datum).flatten() for datum in image]
data = np.array(sampling(data))
label = np.array(sampling(label))

start = time.time()
# emb_tsne = IsomapEmbedding("mnist_sampled_2", data, label=label)
emb_tsne = PcaEmbedding("mnist_sampled_2", data, label=label)
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH)
emb_tsne.print_file(path=PATH_TO_MEASURE_MAP)
'''
######################
#### spheres data #### (different to tadaset sphere) 
raw_spheres_data = list(csv.reader(open("./raw_data/spheres/raw.csv")))[1:]

### AtSNE with spheres data ### 
atsne_spheres_data = list(csv.reader(open("./raw_data/spheres/atsne.csv")))[1:]
spheres_atsne = SpheresEmbedding("spheres", raw_spheres_data, atsne_spheres_data, "atsne")
spheres_atsne.print_file(path=PATH)
print("SPHERES dataset with AtSNE generated")

### PCA with spheres data ###
pca_spheres_data = list(csv.reader(open("./raw_data/spheres/pca.csv")))[1:]
spheres_pca = SpheresEmbedding("spheres", raw_spheres_data, pca_spheres_data, "pca")
spheres_pca.print_file(path=PATH)
print("SPHERES dataset with PCA generated")

### Topoae with spheres data ### 
topoae_spheres_data = list(csv.reader(open("./raw_data/spheres/topoae.csv")))[1:]
spheres_topoae = SpheresEmbedding("spheres", raw_spheres_data, topoae_spheres_data, "topoae")
spheres_topoae.print_file(path=PATH)
print("SPHERES dataset with topoae generated")

### TSNE with spheres data ###
tsne_spheres_data = list(csv.reader(open("./raw_data/spheres/tsne.csv")))[1:]
spheres_tsne = SpheresEmbedding("spheres", raw_spheres_data, tsne_spheres_data, "tsne")
spheres_tsne.print_file(path=PATH)
print("SPHERES dataset with tsne generated")

### UMAP with spheres data ###
umap_spheres_data = list(csv.reader(open("./raw_data/spheres/umap.csv")))[1:]
spheres_umap = SpheresEmbedding("spheres", raw_spheres_data, umap_spheres_data, "umap")
spheres_umap.print_file(path=PATH)
print("SPHERES dataset with umap generated")

### UMATO with spheres data ###
umato_spheres_data = list(csv.reader(open("./raw_data/spheres/umato.csv")))[1:]
spheres_umato = SpheresEmbedding("spheres", raw_spheres_data, umato_spheres_data, "umato")
spheres_umato.print_file(path=PATH)
print("SPHERES dataset with umato generated")

#####################
'''

# image, label = fashion_mnist_test()
# data = [np.array(datum).flatten() for datum in image]
# data = np.array(sampling(data))
# start = time.time()
# emb_tsne = PcaEmbedding("fmnist_sampled_2", data, label=label)
# end   = time.time()
# hp.print_time_spent(start, end, emb_tsne.get_info())
# emb_tsne.print_file(path=PATH)


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
# data = np.array(sampling(data))
# labels = np.array(sampling(labels))
# start = time.time()
# emb_tsne = PcaEmbedding("kmnist_sampled_2", data, label=labels)
# end   = time.time()
# hp.print_time_spent(start, end, emb_tsne.get_info())
# emb_tsne.print_file(path=PATH)


## for pca test
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


