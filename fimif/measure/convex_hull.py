import abc
import numpy as np
import copy
from scipy.spatial import ConvexHull, Delaunay
from qpsolvers import solve_qp


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


## NOTE Convex Hull implementation with scipy
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
        self.hull_vertices = self.hull.points[np.unique(self.hull.simplices)]
        hull_vertices = [self.data[i] for i in self.hull.vertices]
        self.delaunay = Delaunay(hull_vertices)
 
    def is_in_hull(self, points):
        return self.delaunay.find_simplex(points) >= 0


## NOTE Approximate Convex Hull implementation
## Assumes n dimension
## Based on "Conputing the approximate convex hull in high dimensions". Sartuouzadeh et al.
class ConvexHullApprox(ConvexHullABC):
    def __init__(
                 self, 
                 data, 
                 vertices_limit_ratio,    # max vertices ratio
                 error                    # limit error for distance
                ):
        ConvexHullABC.__init__(self, data)
        self.vertices_limit_ratio = vertices_limit_ratio
        self.error = error
        self.__compute_convex_hull()


    def __compute_convex_hull(self):
        E = set()      # for convex hull (holds INDEX)
        S = set(range(len(self.data)))   # original points INDEX
        e_max_num = len(S) * self.vertices_limit_ratio

        current_error = float('inf')
        while(len(E) < e_max_num and current_error > self.error):
            print(current_error)
            # print(E)
            candidate, error, interior_points = self.__find_next_point(S, E)
            E_origin = E
            E.add(candidate)
            S = S - interior_points
            current_error = error
            
            


            if len(E) > 1:
                E_list = list(E)
                E_data = self.data[E_list]
                interior_in_E = set()

                E_copy = copy.deepcopy(E)
                
                for point_idx in E:
                    E_copy.remove(point_idx)
                    if -1e-9 < self.__distance_to_hull(self.data[point_idx], self.data[list(E_copy)]) < 1e-9:
                        interior_in_E.add(point_idx)
                    E_copy.add(point_idx)
                E -= interior_in_E
            


        self.hull_vertices = list(E)
            

    def __find_next_point(self, S, E):
        ## data extraction by idx
        
        I = S - E
        N = len(I)
        S = list(S)
        E = list(E)

        S_data = self.data[S]
        E_data = self.data[E]

        indicies = np.zeros(N)
        max_arr = np.zeros(N)
        matrix = np.zeros((N, N))

        for idx in range(N):
            # print(S_data[0].shape)
            # print(E_data.shape)
            # print(np.array([S_data[point_idx]]).shape)
            # # print(np.concatenate((E_data, S_data[point_idx]), axis=0).shape)
            # print(point_idx)
            dist = self.__distance_to_hull(S_data[0], 
                                           np.concatenate((E_data, np.array([S_data[idx]])), axis=0))
            matrix[0][idx] = dist 
            max_arr[idx] = dist 
        candidate_idx = np.argmin(max_arr)
        indicies = indicies.astype(int)
        indicies[candidate_idx] += 1

        while indicies[candidate_idx] < N:
            # print(indicies[candidate_idx])
            # print(S_data[indicies[candidate_idx]])
            # print()
            dist = self.__distance_to_hull(S_data[indicies[candidate_idx]], 
                                           np.concatenate((E_data, np.array([S_data[candidate_idx]])), axis=0))
            matrix[indicies[candidate_idx]][candidate_idx] = dist
            max_arr[candidate_idx] = max_arr[candidate_idx] if max_arr[candidate_idx] > dist else dist
            candidate_idx = np.argmin(max_arr)
            indicies[candidate_idx] += 1
        # print(len(max_arr))
        error = max_arr[candidate_idx]
        interior_points = set()

        # print(len(S))
        # print(len(E))

        I.remove(S[candidate_idx])
        I = list(I)

        for i in range(N):
            val = matrix[i][candidate_idx]
            if -1e-9 < val < 1e-9:
                if S[i] in I:
                    interior_points.add(S[i])
        # print("interior", len(interior_points))
        return S[candidate_idx], error, interior_points


    
    # INPUT  z: target point (d), S: points set (n X d)
    # OUTPUT distance (the result of quadratic programming)
    def __distance_to_hull(self, z, S):
        S = np.array(S)
        z = np.array(z)
        n, d = S.shape

        P = np.dot(S, S.T)
        P += np.eye(P.shape[0]) * 0.00000000001
        q = - np.dot(S, z).reshape((n,))
        A = np.ones(n)
        b = np.array([1.])
        lb = np.zeros(n)

        x = solve_qp(P=P, q=q, A=A, b=b, lb=lb)

        interior_point = (S.T * x).T.sum(axis=0)
        dist = np.linalg.norm(z - interior_point)
        return dist


        
    
    def is_in_hull(self, points):
        pass
