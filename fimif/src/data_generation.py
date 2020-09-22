import numpy as np
from sklearn.manifold import TSNE

# def tsne_2d(hd_data):
#     embedding = TSNE(n_components=2).fit_transform(hd_data)
#     return embedding

class TsneDataPair:
    def __init__(self, hd_data):
        self.raw = hd_data
        self.embedded = TSNE(n_components=2).fit_transform(self.raw)
        self.size = self.raw.shape[0]
    
