import numpy as np
import json
from sklearn.manifold import TSNE


class Embedding:


    def __init__(self):
        self.data = []

    def print_file(self, file_name=None):
        file_name = "./" + self.__name__ if file_name == None else file_name
        with open(file_name, "w", encoding="utf-8") as json_file:
            json.dump(self.data, json_file, ensure_ascii=False, indent=4)

    

class TsneEmbedding(Embedding):
    def __init__(self, hd_data):
        Embedding.__init__(self)
        embedded = TSNE(n_components=2).fit_transform(hd_data)
        for idx, datum in enumerate(hd_data):
            datum_set = {}
            datum_set["raw"] = datum.tolist()
            datum_set["emb"] = embedded[idx].tolist()
            self.data.append(datum_set)
        
        self.size = hd_data.shape[0]



    
    
