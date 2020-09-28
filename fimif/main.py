import time
import tadasets

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np
import pandas as pd
import csv

PATH_TO_WEB = "../web/src/json/"

### spheres dataset (different to tadaset sphere dataset) ###
# atsne_spheres = pd.read_csv("https://raw.githubusercontent.com/hyungkwonko/umato/master/visualization/public/results/spheres/atsne.csv", "r")
# pca_spheres = pd.read_csv("https://raw.githubusercontent.com/hyungkwonko/umato/master/visualization/public/results/spheres/pca.csv", "r")
# topoae_spheres = pd.read_csv("https://raw.githubusercontent.com/hyungkwonko/umato/master/visualization/public/results/spheres/topoae.csv", "r")
# tsne_spheres = pd.read_csv("https://raw.githubusercontent.com/hyungkwonko/umato/master/visualization/public/results/spheres/tsne.csv", "r")
# umap_spheres = pd.read_csv("https://raw.githubusercontent.com/hyungkwonko/umato/master/visualization/public/results/spheres/umap.csv", "r")




# raw_spheres =  pd.read_csv("https://raw.githubusercontent.com/hyungkwonko/umato/master/data/spheres/spheres.csv", "r")
# umato_spheres = pd.read_csv("https://raw.githubusercontent.com/hyungkwonko/umato/master/visualization/public/results/spheres/umato.csv", "r")
# raw_spheres = np.array(raw_spheres)
# umato_spheres = np.array(umato_spheres)

raw_spheres = list(csv.reader(open("./data/spheres/raw.csv")))[1:]
umato_spheres = list(csv.reader(open("./data/spheres/umato.csv")))[1:]

spheres_umato = SpheresEmbedding("spheres", raw_spheres, umato_spheres, "umato")
spheres_umato.print_file(path=PATH_TO_WEB)


# ### TSNE with mnist & Cosine Similarity distance ###
# start = time.time()
# emb_tsne = TsneEmbedding("mnist_test_cosine_similarity", data, label=label, metric="cosine")
# end   = time.time()

# ### TSNE with mnist & Euclidean distance ###
# # start = time.time()
# # emb_tsne = TsneEmbedding("mnist_test_euclidean", data, label=label, metric="cosine")
# # end   = time.time()
# ######################
# hp.print_time_spent(start, end, emb_tsne.get_info())


# emb_tsne.print_file(path=PATH_TO_WEB)



