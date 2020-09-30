import abc
import numpy as np
from scipy.spatial import ConvexHull, Delaunay


# Convex Hull Interface
class ConvexHullABC:

    __metaclass__ = abc.ABCMeta

    def __init__(self, data):
        self.data = np.array(data)     # initial data set
        self.size = len(data)
        self.dim  = len(data[0])
        self.hull = None       # final convex hull

    @abc.abstractmethod
    def __compute_convex_hull(self):
        pass

    ## INPUT  arbitrary point
    ## OUTPUT True if convex hull contains point. False if not
    @abc.abstractmethod
    def is_in_hull(self, points):
        pass



## Quite Scalable for until... ( < 1 seconds in mac mini 2018)
## dimension: 5
## Convex hull candidate: 10000
## Convex hull test: 100000
class ConvexHullWithScipy(ConvexHullABC):
    def __init__(self, data):
        ConvexHullABC.__init__(self, data)
        self.__compute_convex_hull()
    def __compute_convex_hull(self):
        self.hull = ConvexHull(self.data)
        hull_vertices = [self.data[i] for i in self.hull.vertices]
        self.delaunay = Delaunay(hull_vertices)
 
    def is_in_hull(self, points):
        return self.delaunay.find_simplex(points) >= 0


class ConvexHullApprox(ConvexHullABC):
    def __init__(self, data):
        ConvexHullABC.__init__(self, data)

    def __compute_convex_hull(self):
        pass

    def is_in_hull(self, points):
        pass
