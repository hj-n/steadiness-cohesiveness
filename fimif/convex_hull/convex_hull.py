

class ConvexHull:
    def __init__(self, data):
        self.data = data     # initial data set
        self.size = len(data)
        self.dim  = len(data[0])
        self.hull = []       # final convex hull
        self.__compute_convex_hull(self)

    def __compute_convex_hull(self):
        ## Things to implement ##
        pass

    ## INPUT  arbitrary point
    ## OUTPUT True if convex hull contains point. False if not
    def contains_point(self, point):
        ## Things to implement ##
        return True