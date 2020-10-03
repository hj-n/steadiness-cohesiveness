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

## TEST code for plotting the result of approx / scipy-based convex hull
def test_plot_approx_convex_hull():
    dim = 2
    data = np.random.rand(100, dim)
    convex_hull_approx = ConvexHullApprox(data, 0.5, 0.000001)
    convex_hull_scipy = ConvexHullWithScipy(data)

    scipy_vertices = set(convex_hull_scipy.hull.vertices)
    approx_vertices = set(convex_hull_approx.hull_vertices)

    scipy_vertices_coor = data[convex_hull_scipy.hull.vertices]
    approx_vertices_coor = data[convex_hull_approx.hull_vertices]

    plt.plot(data[:, 0], data[:, 1], 'o')

    random_data = 5 * np.random.rand(4000, dim) - 2.5
    outside_scipy = []
    outside_approx = []

    outside_scipy = np.invert(convex_hull_scipy.is_in_hull(random_data))
    outside_approx = np.invert(convex_hull_approx.is_in_hull(random_data))

    outside_scipy_idx = list(np.array(range(4000))[outside_scipy])
    outside_approx_idx = list(np.array(range(4000))[outside_approx])

    outside_scipy_array = random_data[outside_scipy]
    outside_approx_array = random_data[outside_approx]

    difference_set = set(outside_approx_idx) - set(outside_scipy_idx)

    print("====== TEST RESULT ======")
    print("Total random point tested:", 4000)
    print("REAL   outside points num:", len(outside_scipy_array))
    print("APPROX outside points num:", len(outside_approx_array))
    print("Difference               :", len(difference_set))
    print("Error rate               :", float(len(difference_set)) / 4000)
    print("=========================")

    plt.plot(outside_approx_array[:, 0], outside_approx_array[:, 1], 'd')
    plt.plot(outside_scipy_array[:, 0], outside_scipy_array[:, 1], 'v')

    plt.show()


    
test_plot_approx_convex_hull()



