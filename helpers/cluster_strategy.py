from abc import ABC, abstractmethod
from . import snn_knn as sk

def preprocessing(strategy : str, parameter, raw_dist_matrix, emb_dist_matrix):
    cstrat = {
        "snn" : SNNCS(parameter, raw_dist_matrix, emb_dist_matrix)
    }["snn"]
    cstrat.preprocessing()
    return cstrat

def extract_cluster(strategy : str):
    pass

class ClusterStrategy(ABC):

    '''
    Saving distance matrix info and setting parameter
    '''
    def __init__(self, raw_dist_matrix, emb_dist_matrix):
        self.raw_dist_matrix = raw_dist_matrix
        self.emb_dist_matrix = emb_dist_matrix

    @abstractmethod
    def preprocessing(self):
        pass

    @abstractmethod
    def extract_cluster(self):
        pass
    

    '''
    Get the indices of the points which to be clustered as input
    and return the clustereing result
    '''
    @abstractmethod
    def clustering(self):
        pass

class SNNCS(ClusterStrategy):

    def __init__(self, parameter, raw_dist_matrix, emb_dist_matrix):
        super().__init__(raw_dist_matrix, emb_dist_matrix)
        self.k = parameter["k"]

    def preprocessing(self):
        # Compute knn infos
        raw_knn_info = sk.knn_info(self.raw_dist_matrix, self.k)
        emb_knn_info = sk.knn_info(self.emb_dist_matrix, self.k)
        
        # Compute snn matrix
        length = len(raw_knn_info)
        self.raw_snn_matrix = sk.snn_gpu(raw_knn_info, length, self.k)
        self.emb_snn_matrix = sk.snn_gpu(emb_knn_info, length, self.k)

        pass

    def extract_cluster(self):
        pass

    def clustering(self):
        pass
        
        