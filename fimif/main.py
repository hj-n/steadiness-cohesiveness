import time
import tadasets

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np
import pandas as pd
import csv

PATH_TO_WEB = "../web/src/json/"



######################
#### spheres data #### (different to tadaset sphere) 
raw_spheres_data = list(csv.reader(open("./data/spheres/raw.csv")))[1:]

### AtSNE with spheres data ### 
atsne_spheres_data = list(csv.reader(open("./data/spheres/atsne.csv")))[1:]
spheres_atsne = SpheresEmbedding("spheres", raw_spheres_data, atsne_spheres_data, "atsne")
spheres_atsne.print_file(path=PATH_TO_WEB)

### PCA with spheres data ###
raw_spheres_data = list(csv.reader(open("./data/spheres/raw.csv")))[1:]
pca_spheres_data = list(csv.reader(open("./data/spheres/pca.csv")))[1:]
spheres_pca = SpheresEmbedding("spheres", raw_spheres_data, pca_spheres_data, "pca")
spheres_pca.print_file(path=PATH_TO_WEB)

### Topoae with spheres data ### 
topoae_spheres_data = list(csv.reader(open("./data/spheres/topoae.csv")))[1:]
spheres_topoae = SpheresEmbedding("spheres", raw_spheres_data, topoae_spheres_data, "topoae")
spheres_topoae.print_file(path=PATH_TO_WEB)

### TSNE with spheres data ###
tsne_spheres_data = list(csv.reader(open("./data/spheres/tsne.csv")))[1:]
spheres_tsne = SpheresEmbedding("spheres", raw_spheres_data, tsne_spheres_data, "tsne")
spheres_tsne.print_file(path=PATH_TO_WEB)

### UMAP with spheres data ###
umap_spheres_data = list(csv.reader(open("./data/spheres/umap.csv")))[1:]
spheres_umap = SpheresEmbedding("spheres", raw_spheres_data, umap_spheres_data, "umap")
spheres_umap.print_file(path=PATH_TO_WEB)

### UMATO with spheres data ###
umato_spheres_data = list(csv.reader(open("./data/spheres/umato.csv")))[1:]
spheres_umato = SpheresEmbedding("spheres", raw_spheres_data, umato_spheres_data, "umato")
spheres_umato.print_file(path=PATH_TO_WEB)



