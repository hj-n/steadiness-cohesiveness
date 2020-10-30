import time
import tadasets
import umap

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np
import pandas as pd
import csv

PATH_TO_WEB = "../../web/src/json/"
PATH_TO_MEASURE = "./../fimif_measure/json/"
PATH = PATH_TO_MEASURE

preswissroll = [i.strip().split() for i in open("./raw_data/multiclass_swiss_roll/preswissroll.dat").readlines()]
swissroll = [i.strip().split() for i in open("./raw_data/multiclass_swiss_roll/swissroll.dat").readlines()]

def mss_generator(move):
    multiclass_swissroll_data = []
    for i in range(1600):
        datum = {
            "raw": [float(idx) for idx in swissroll[i]],
            "emb": [float(idx) for idx in preswissroll[i]],
            "label": i // 400 + 1
        }
        direction = {
            0: [-1, -1], 1: [-1, 1], 2: [1, -1], 3: [1, 1]
        }[i // 400]
        datum["emb"][0] += move * direction[0] * 0.5
        datum["emb"][1] += move * direction[1] * 0.5
        multiclass_swissroll_data.append(datum)

    with open(PATH + "multiclass_swissroll_"+ str(move) + "_none.json", "w", encoding="utf-8") as json_file:
                json.dump(multiclass_swissroll_data, json_file, ensure_ascii=False, indent=4)

def mss_missing_generator(move):
    multiclass_swissroll_data = []
    for i in range(1600):
        datum = {
            "raw": [float(idx) for idx in swissroll[i]],
            "emb": [float(idx) for idx in preswissroll[i]],
            "label": i // 400 + 1
        }
        centroid = {
            0: [7.5, 7.5], 1: [7.5, 12.5], 2: [12.5, 7.5], 3: [12.5, 12.5]
        }[i // 400]
        if(datum["emb"][0] < centroid[0]):
            datum["emb"][0] -= move * 0.25
        else:
            datum["emb"][0] += move * 0.25
        if(datum["emb"][1] < centroid[1]):
            datum["emb"][1] -= move * 0.25
        else:
            datum["emb"][1] += move * 0.25

        multiclass_swissroll_data.append(datum)

    with open(PATH + "multiclass_swissroll_oneside_"+ str(move) + "_none.json", "w", encoding="utf-8") as json_file:
                json.dump(multiclass_swissroll_data, json_file, ensure_ascii=False, indent=4)
                

                

image, label = mnist_test()
data = [np.array(datum).flatten() for datum in image]
data = np.array(data)

for p in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    start = time.time()
    emb_tsne = TsneEmbedding("mnist_test_cosine_similarity", data, label=label, metric="cosine", perplexity=p)
    end   = time.time()
    hp.print_time_spent(start, end, emb_tsne.get_info())
    emb_tsne.print_file(path=PATH)

# mss_missing_generator(0)
# mss_missing_generator(1)
# mss_missing_generator(2)
# mss_missing_generator(3)
# mss_missing_generator(4)
# mss_missing_generator(5)
# mss_missing_generator(6)
# mss_missing_generator(7)
# mss_missing_generator(8)
# mss_missing_generator(9)
# mss_missing_generator(10)
# mss_missing_generator(11)
# mss_missing_generator(12)
# mss_missing_generator(13)


# mss_generator(-7)
# mss_generator(-6)
# mss_generator(-5)
# mss_generator(-4)
# mss_generator(-3)
# mss_generator(-2)
# mss_generator(-1)
# mss_generator(0)
# mss_generator(1)
# mss_generator(2)
# mss_generator(3)
# mss_generator(4)
# mss_generator(5)
# mss_generator(6)
# mss_generator(7)