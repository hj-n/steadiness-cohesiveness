import time
import tadasets

from data_generation import *
import helper as hp


PATH_TO_WEB = "../web/src/json/"

data = tadasets.sphere(n=1000, r=10)


start = time.time()
emb_sphere_tsne = TsneEmbedding("sphere", data)
end   = time.time()

hp.print_time_spent(start, end, emb_sphere_tsne.get_info())


emb_sphere_tsne.print_file(path=PATH_TO_WEB)



