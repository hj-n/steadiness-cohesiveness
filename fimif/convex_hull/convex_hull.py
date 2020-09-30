import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import time


# Convex Hull Interface
class ConvexHullABC:

    __metaclass__ = abc.ABCMeta

    def __init__(self, data):
        self.data = np.array(data)     # initial data set
        self.size = len(data)
        self.dim  = len(data[0])
        self.hull = None       # final convex hull
        self.delaunay = None

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


## TEST code for convex hull implementation
def test_convex_hull(num, dim):
    points = np.random.rand(num, dim)

    # Convex hull construction
    start = time.time()
    convex_hull = ConvexHullWithScipy(points)
    end = time.time()
    print("COMPUTING CONVEX HULL:", end - start)

    if dim == 2:
        plt.plot(points[:,0], points[:,1], 'o')
        for simplex in convex_hull.hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        plt.triplot(convex_hull.delaunay.points[:,0], convex_hull.delaunay.points[:,1], convex_hull.delaunay.simplices)
    

    test_points = np.random.rand(100000, dim)

    start = time.time()
    inside = convex_hull.is_in_hull(test_points)
    end = time.time()
    print("CHECKING CONVEX HULL POINT CONTAINMENT:", end - start)

    if dim == 2:
        plt.plot(test_points[ inside,0],test_points[ inside,1],'.k')
        plt.plot(test_points[~inside,0],test_points[~inside,1],'.r')
    
    print("FINISHED")
    
    if dim == 2:
        plt.show()

test_convex_hull(10000, 5)
