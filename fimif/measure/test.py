import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import json


from convex_hull import *
from measure import *

## TEST code for convex hull implementation
def test_convex_hull_with_scipy(num, dim):
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

def test_fimif_measure_vanilla():
    file = open("./json/swiss_roll_tsne.json", "r") 
    data = json.load(file)
    measure = FimifMeasure(
                           data=data, 
                           boundary="hyperball", 
                           cluster_selection="entire", 
                           cluster_shape="circle",
                           k=4, 
                           cluster_seed_num=2
                          )
    measure.evaluate()


test_fimif_measure_vanilla()
