import time
import tadasets

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np


PATH_TO_WEB = "../web/src/json/"
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


emb_tsne.print_file(path=PATH_TO_WEB)



