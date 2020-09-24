import time
import tadasets

from embedding import *
from dataset_gen import *
import helper as hp
import numpy as np


PATH_TO_WEB = "../web/src/json/"

### tadaset data ###
# data = swiss_roll(n=4000, r=10)
# data = sphere(n=1000, r=10)

### MNIST TEST data ###
image, label = mnist_test()
data = [np.array(datum).flatten() for datum in image]
data = np.array(data)

start = time.time()
emb_tsne = TsneEmbedding("mnist_test", data, label=label)
end   = time.time()

hp.print_time_spent(start, end, emb_tsne.get_info())


emb_tsne.print_file(path=PATH_TO_WEB)



