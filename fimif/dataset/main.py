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
PATH = PATH_TO_MEASURE

def sampling(original_list):
    return [datum for (i, datum) in enumerate(original_list) if i % 3 == 0]


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

spheres_data = list(csv.reader(open("./raw_data/spheres/raw.csv")))[1:]
spheres_raw_data = np.array(sampling([datum[:-1] for datum in spheres_data]))
spheres_label = np.array(sampling([datum[-1] for datum in spheres_data]))


for p in [0.2, 0.4]:
    for n in [3, 10, 20, 30, 40, 50, 100, 150, 200, 400, 600, 800, 1000]:
        final_data = [] 
        key_summary = str(n) + "_" + str(p)
        umap_instance = umap.UMAP(n_neighbors=n, min_dist=p)
        spheres_emb_data =umap_instance.fit_transform(spheres_raw_data)
        for (i, _) in enumerate(spheres_emb_data):
            datum = {}
            datum["raw"] = spheres_raw_data[i].tolist()
            datum["emb"] = spheres_emb_data[i].tolist()
            datum["label"] = spheres_label[i]
            final_data.append(datum)
        print("UMAP for", "spheres", key_summary, "finished!!")
        with open(PATH + "spheres_sampled_" + key_summary + ".json", "w") as outfile:
            json.dump(final_data, outfile)
        

        

'''
with open('./raw_data/mammoth/mammoth_umap.json') as json_file:
    json_data = json.load(json_file)
    projections = json_data["projections"]
    raw_data = sampling(json_data["3d"])
    labels = sampling(json_data["labels"])
    for key in projections.keys():
        final_data = []
        key_summary = key.replace("n=","").replace("d=","").replace(",", "_")
        projection = sampling(projections[key])
        for (i, _) in projection:
            datum = {}
            datum["raw"] = raw_data[i]
            datum["emb"] = projection[i]
            datum["label"] = projection[i]
            final_data.append(datum)
        with open(PATH + "mammoth_" + key_summary + ".json", "w") as outfile:
            json.dump(final_data, outfile)
'''            
            

