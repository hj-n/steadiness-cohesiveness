import time
import tadasets

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np
import csv


PATH_TO_WEB = "../../web/src/json/"

#################################
### TSNE with swiss_roll data ###
data = swiss_roll(n=4000, r=10)

start = time.time()
emb_tsne = TsneEmbedding("swiss_roll", data, label=label, metric="cosine")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH_TO_WEB)
#################################

#############################
### TSNE with sphere data ###
data = sphere(n=1000, r=10)

start = time.time()
emb_tsne = TsneEmbedding("swiss_roll", data, label=label, metric="cosine")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH_TO_WEB)
#############################

#######################
### mnist_test data ###
image, label = mnist_test()
data = [np.array(datum).flatten() for datum in image]
data = np.array(data)

### TSNE with mnist & Cosine Similarity distance ###
start = time.time()
emb_tsne = TsneEmbedding("mnist_test_cosine_similarity", data, label=label, metric="cosine")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH_TO_WEB)

### TSNE with mnist & Euclidean distance ###
start = time.time()
emb_tsne = TsneEmbedding("mnist_test_euclidean", data, label=label, metric="cosine")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH_TO_WEB)

######################


######################
#### spheres data #### (different to tadaset sphere) 
raw_spheres_data = list(csv.reader(open("./raw_data/spheres/raw.csv")))[1:]

### AtSNE with spheres data ### 
atsne_spheres_data = list(csv.reader(open("./raw_data/spheres/atsne.csv")))[1:]
spheres_atsne = SpheresEmbedding("spheres", raw_spheres_data, atsne_spheres_data, "atsne")
spheres_atsne.print_file(path=PATH_TO_WEB)

### PCA with spheres data ###
raw_spheres_data = list(csv.reader(open("./raw_data/spheres/raw.csv")))[1:]
pca_spheres_data = list(csv.reader(open("./raw_data/spheres/pca.csv")))[1:]
spheres_pca = SpheresEmbedding("spheres", raw_spheres_data, pca_spheres_data, "pca")
spheres_pca.print_file(path=PATH_TO_WEB)

### Topoae with spheres data ### 
topoae_spheres_data = list(csv.reader(open("./raw_data/spheres/topoae.csv")))[1:]
spheres_topoae = SpheresEmbedding("spheres", raw_spheres_data, topoae_spheres_data, "topoae")
spheres_topoae.print_file(path=PATH_TO_WEB)

### TSNE with spheres data ###
tsne_spheres_data = list(csv.reader(open("./raw_data/spheres/tsne.csv")))[1:]
spheres_tsne = SpheresEmbedding("spheres", raw_spheres_data, tsne_spheres_data, "tsne")
spheres_tsne.print_file(path=PATH_TO_WEB)

### UMAP with spheres data ###
umap_spheres_data = list(csv.reader(open("./raw_data/spheres/umap.csv")))[1:]
spheres_umap = SpheresEmbedding("spheres", raw_spheres_data, umap_spheres_data, "umap")
spheres_umap.print_file(path=PATH_TO_WEB)

### UMATO with spheres data ###
umato_spheres_data = list(csv.reader(open("./raw_data/spheres/umato.csv")))[1:]
spheres_umato = SpheresEmbedding("spheres", raw_spheres_data, umato_spheres_data, "umato")
spheres_umato.print_file(path=PATH_TO_WEB)



######################