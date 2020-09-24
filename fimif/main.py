import time
import tadasets

from embedding import *
from dataset_gen import *
import helper as hp


PATH_TO_WEB = "../web/src/json/"

data = swiss_roll(n=4000, r=10)


start = time.time()
emb_swiss_roll_tsne = TsneEmbedding("swiss_roll", data)
end   = time.time()

hp.print_time_spent(start, end, emb_swiss_roll_tsne.get_info())


emb_swiss_roll_tsne.print_file(path=PATH_TO_WEB)



