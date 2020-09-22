import numpy as np
from sklearn.manifold import TSNE

# def tsne_2d(hd_data):
#     embedding = TSNE(n_components=2).fit_transform(hd_data)
#     return embedding

class TsneDataPair:
    def __init__(self, hd_data):
        embedded = TSNE(n_components=2).fit_transform(hd_data)
        
        self.data = []
        for idx, datum in enumerate(hd_data):
            datum_set = {}
            datum_set["raw"] = datum
            datum_set["emb"] = embedded[idx]
            self.data.append(datum_set)
        
        self.size = hd_data.shape[0]
    
    
