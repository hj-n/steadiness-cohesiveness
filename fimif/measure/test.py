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
                           cluster_seed_num=2,
                           iter=200,
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
                           cluster_seed_num=2,
                           iter=200,
                           class_num=11,
                          )
    measure.evaluate()

    

    # avg_log = []
    # sum = 0
    # for idx, val in enumerate(measure.log):
    #     sum += val
    #     avg_log.append(sum / (idx + 1))
   
    # plt.plot(range(len(measure.log)), measure.log, 'o')
    # plt.plot(range(len(measure.log)), avg_log, 'v')
    # plt.show()



# test_fimif_measure_vanilla()
tested = np.array([[-5.92141457,  7.89902009,  1.59446894],
 [ 2.0491028,  -5.45365773,  8.12765619],
 [ 3.65892579, -1.6350357,   9.16181861],
 [-4.65937127,  8.21246851,  3.29326893],
 [ 8.98298156,  0.66479,     4.34328177],
 [-2.45807746,  3.58609736,  9.00542953],
 [-1.94579528, -6.67635215,  7.18611179],
 [-8.31764115, -5.51700749, -0.61601468],
 [-9.32226484,  3.61434082,  0.17865777],
 [-7.95167845, -5.87687262,  1.49438216],
 [ 8.42131903, -0.41938117,  5.37638402],
 [-1.77484238, -3.11024679,  9.33682491],
 [-7.77408759, -5.94485804,  2.05480535],
 [-5.85359132, -5.37929608,  6.06618846],
 [-8.93258746, -3.39448489,  2.94726206],
 [ 0.70822349, -5.83349358,  8.09127754],
 [-2.36169208,  2.96897129,  9.2524386 ],
 [-1.93691599, -3.50294288,  9.16393734],
 [-5.24001016, -2.35404499,  8.18539954],
 [ 2.17840345, -2.5000694,   9.43420433]])

tested = tested / 1000
cloud = np.random.rand(20, 3)
print(cloud.shape)
print(tested.shape)

for i in range(20):
    for j in range(3):
        cloud[i][j] = tested[i][j]

convex = ConvexHullWithScipy(tested)
print(convex.hull.vertices.shape)
convex = ConvexHullWithScipy(cloud)
print(convex.hull.vertices.shape)
