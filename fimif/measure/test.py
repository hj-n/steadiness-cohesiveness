import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import json
import tadasets

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

## TEST code for vanilla measure
def test_fimif_measure_vanilla():
    # print("ATSNE-SPHERES")
    # file = open("./json/spheres_atsne.json", "r") 
    # data = json.load(file)
    # measure = FimifMeasure(
    #                        data=data, 
    #                        boundary="hyperball", 
    #                        cluster_selection="class", 
    #                        cluster_shape="circle",
    #                        k=4, 
    #                        cluster_seed_num=4,
    #                        iter=1000,
    #                        class_num=11,
    #                       )
    # measure.evaluate()
    print("SWISS_ROLL_HYPERBALL")
    file = open("./json/swiss_roll_tsne.json", "r") 
    data = json.load(file)
    measure = FimifMeasure(
                           data=data, 
                           boundary="hyperball", 
                           cluster_selection="entire", 
                           cluster_shape="circle",
                           k=4, 
                           cluster_seed_num=4,
                           iter=1000,
                           class_num=11,
                          )
    measure.evaluate()
    print("SWISS_ROLL_CONVEXHULL")
    file = open("./json/swiss_roll_tsne.json", "r") 
    data = json.load(file)
    measure = FimifMeasure(
                           data=data, 
                           boundary="convexhull", 
                           cluster_selection="entire", 
                           cluster_shape="circle",
                           k=4, 
                           cluster_seed_num=4,
                           iter=1000,
                           class_num=11,
                          )
    measure.evaluate()

    

    avg_log = []
    sum = 0
    for idx, val in enumerate(measure.log):
        sum += val
        avg_log.append(sum / (idx + 1))
   
    plt.plot(range(len(measure.log)), measure.log, 'o')
    plt.plot(range(len(measure.log)), avg_log, 'v')
    plt.show()


def test_convex_hull_approx():
    data = tadasets.swiss_roll(n=300, r=10)
    convex_hull = ConvexHullApprox(data, 0.5, 0.01)
    
test_convex_hull_approx()

