import numpy as np
import json
from sklearn.manifold import TSNE


class Embedding:

    def __init__(self, data_name):
        self.data = []
        self.data_name = data_name
        self.method_name = None
        self.size = None

    def print_file(self, path=None):
        identifier = self.data_name + "_" + self.method_name + ".json"
        file_name = "./" + identifier if path == None else path + identifier
        with open(file_name, "w", encoding="utf-8") as json_file:
            json.dump(self.data, json_file, ensure_ascii=False, indent=4)


    
    def get_info(self):
        # parses info (indicated in class identifier) and return as string
        info = [self.data_name, self.method_name]
        return info

## Generates Spheres embedding with pre-computed raw / emb data
class SpheresEmbedding(Embedding):
    def __init__(self, data_name, raw_data, emb_data, method_name):
        Embedding.__init__(self, data_name)
        if (len(raw_data) != len(emb_data)):
            raise Exception("spheres data length unmatching error")
        for idx, raw_datum in enumerate(raw_data):
            datum_set = {}
            datum_set["raw"] = [ float(el) for el in raw_datum[:-1] ]
            datum_set["emb"] = [ float(el) for el in emb_data[idx][:-1] ]
            datum_set["label"] = raw_datum[-1]
            self.data.append(datum_set)
        self.method_name = method_name
        self.size = len(raw_data)



## Generates TSNE Embedding with given data
class TsneEmbedding(Embedding):
    def __init__(self, data_name, hd_data, label=None, metric='euclidean'):
        Embedding.__init__(self, data_name)
        embedded = TSNE(n_components=2, metric=metric).fit_transform(hd_data)
        for idx, datum in enumerate(hd_data):
            datum_set = {}
            datum_set["raw"] = datum.tolist()
            datum_set["emb"] = embedded[idx].tolist()
            if(label != None):
                datum_set["label"] = label[idx]
            self.data.append(datum_set)
        
        self.size = hd_data.shape[0]
        self.method_name = "tsne"



    
    
