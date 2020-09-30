from convex_hull import *
from enum import Enum

## CONSTANT Boundary implementation enum ##
class Boundray(Enum):
    HB  = 1   # hyperball
    CHS = 2   # convex hull with scipy implementation
    CHA = 3   # convex hull with approximation

## CONSTANT random selection enum ##
class PointSelection(Enum):
    RDE = 1    # Random selection within Entire point set
    RDC = 2    # Random selection within a single class


## Fimif Measure class 
class FimifMeasure:
    def __init__(self, data, boundary, point_selection):
        self.data = data
        self.boundary = boundary
        self.point_selection = point_selection

    def 

