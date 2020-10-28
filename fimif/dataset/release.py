import time
import tadasets

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np
import csv


PATH_TO_WEB = "./../../web/src/json/"
PATH_TO_MEASURE = "./../measure/json/"
PATH = PATH_TO_MEASURE

#################################
### TSNE with swiss_roll data ###
data = swiss_roll(n=1000, r=5)

start = time.time()
emb_tsne = TsneEmbedding("swiss_roll", data, metric="euclidean")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH)
#################################

#############################
### TSNE with sphere data ###
data = sphere(n=1000, r=10)

start = time.time()
emb_tsne = TsneEmbedding("sphere", data, metric="euclidean")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH)
#############################

'''
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
emb_tsne.print_file(path=PATH)

### TSNE with mnist & Euclidean distance ###
start = time.time()
emb_tsne = TsneEmbedding("mnist_test_euclidean", data, label=label, metric="euclidean")
end   = time.time()
hp.print_time_spent(start, end, emb_tsne.get_info())
emb_tsne.print_file(path=PATH)

######################
'''

# ######################
# #### spheres data #### (different to tadaset sphere) 
# raw_spheres_data = list(csv.reader(open("./raw_data/spheres/raw.csv")))[1:]

# ### AtSNE with spheres data ### 
# atsne_spheres_data = list(csv.reader(open("./raw_data/spheres/atsne.csv")))[1:]
# spheres_atsne = SpheresEmbedding("spheres", raw_spheres_data, atsne_spheres_data, "atsne")
# spheres_atsne.print_file(path=PATH)
# print("SPHERES dataset with AtSNE generated")

# ### PCA with spheres data ###
# pca_spheres_data = list(csv.reader(open("./raw_data/spheres/pca.csv")))[1:]
# spheres_pca = SpheresEmbedding("spheres", raw_spheres_data, pca_spheres_data, "pca")
# spheres_pca.print_file(path=PATH)
# print("SPHERES dataset with PCA generated")

# ### Topoae with spheres data ### 
# topoae_spheres_data = list(csv.reader(open("./raw_data/spheres/topoae.csv")))[1:]
# spheres_topoae = SpheresEmbedding("spheres", raw_spheres_data, topoae_spheres_data, "topoae")
# spheres_topoae.print_file(path=PATH)
# print("SPHERES dataset with topoae generated")

# ### TSNE with spheres data ###
# tsne_spheres_data = list(csv.reader(open("./raw_data/spheres/tsne.csv")))[1:]
# spheres_tsne = SpheresEmbedding("spheres", raw_spheres_data, tsne_spheres_data, "tsne")
# spheres_tsne.print_file(path=PATH)
# print("SPHERES dataset with tsne generated")

# ### UMAP with spheres data ###
# umap_spheres_data = list(csv.reader(open("./raw_data/spheres/umap.csv")))[1:]
# spheres_umap = SpheresEmbedding("spheres", raw_spheres_data, umap_spheres_data, "umap")
# spheres_umap.print_file(path=PATH)
# print("SPHERES dataset with umap generated")

# ### UMATO with spheres data ###
# umato_spheres_data = list(csv.reader(open("./raw_data/spheres/umato.csv")))[1:]
# spheres_umato = SpheresEmbedding("spheres", raw_spheres_data, umato_spheres_data, "umato")
# spheres_umato.print_file(path=PATH)
# print("SPHERES dataset with umato generated")

######################



'''


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


'''