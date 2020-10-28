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


multiclass_swissroll_data = []
for i in range(1600):
    datum = {
        "raw": [float(idx) for idx in swissroll[i]],
        "emb": [float(idx) for idx in preswissroll[i]],
        "label": i // 400 + 1
    }
    multiclass_swissroll_data.append(datum)

print(multiclass_swissroll_data)

with open(PATH + "multiclass_swissroll_none.json", "w", encoding="utf-8") as json_file:
            json.dump(multiclass_swissroll_data, json_file, ensure_ascii=False, indent=4)


