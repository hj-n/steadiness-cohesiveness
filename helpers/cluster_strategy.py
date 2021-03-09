from abc import ABC, abstractmethod

def preprocessing(strategy : str, raw_data, emb_data):
    cstrat = {
        "snn" : SNNCS(raw_data, emb_data)
    }
    return cstrat
    pass

def extract_cluster(strategy : str):
    pass

class ClusterStrategy(ABC):

    def __init__(self, raw_data, emb_data):
        self.raw = raw_data
        self.emb = emb_data

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

    def preprocessing(self):
        

    def extract_cluster(self):
        pass

    def clustering(self):
        pass
        
        