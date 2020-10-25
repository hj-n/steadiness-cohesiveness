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

image, label = mnist_test()
data = [np.array(datum).flatten() for datum in image]
data = np.array(data)

# print(data[3])
label = np.array(label)


start = 400
num = 150
data_sample = data[start: start + num]
label_sample = label[start: start + num]
for i in range(1, 10):
    data_sample = np.concatenate((data_sample,data[i * 1000 + start: i * 1000 + start + num]))
    label_sample = np.concatenate((label_sample, label[i * 1000 + start: i * 1000 + start + num]))
unique, counts = np.unique(label_sample, return_counts=True)
print(dict(zip(unique, counts)))

start = time.time()
emb_umap = UmapEmbedding("mnist_sampled", data_sample, label=label_sample)
end   = time.time()
hp.print_time_spent(start, end, emb_umap.get_info())
emb_umap.print_file(path=PATH)

start = time.time()
emb_pca = PcaEmbedding("mnist_sampled", data_sample, label=label_sample)
end   = time.time()
hp.print_time_spent(start, end, emb_pca.get_info())
emb_pca.print_file(path=PATH)

start = time.time()
emb_tsne = TsneEmbedding("mnist_sampled", data_sample, label=label_sample, metric="euclidean")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH)




##########

data = swiss_roll(n=1500, r=10)

start = time.time()
emb_umap = UmapEmbedding("swiss_roll", data)
end   = time.time()
hp.print_time_spent(start, end, emb_umap.get_info())
emb_umap.print_file(path=PATH)

start = time.time()
emb_pca = PcaEmbedding("swiss_roll", data)
end   = time.time()
hp.print_time_spent(start, end, emb_pca.get_info())
emb_pca.print_file(path=PATH)

start = time.time()
emb_tsne = TsneEmbedding("swiss_roll", data, metric="euclidean")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH)
#################################

#############################
data = sphere(n=1500, r=10)

start = time.time()
emb_umap = UmapEmbedding("sphere", data)
end   = time.time()
hp.print_time_spent(start, end, emb_umap.get_info())
emb_umap.print_file(path=PATH)

start = time.time()
emb_pca = PcaEmbedding("sphere", data)
end   = time.time()
hp.print_time_spent(start, end, emb_pca.get_info())
emb_pca.print_file(path=PATH)

start = time.time()
emb_tsne = TsneEmbedding("sphere", data, metric="euclidean")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH)
