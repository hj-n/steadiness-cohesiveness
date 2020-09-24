
import tadasets

from data_generation import *

data = tadasets.sphere(n=1000, r=10)

emb_sphere_tsne = TsneEmbedding(data)

emb_sphere_tsne.print_file(file_name="./test.json")

