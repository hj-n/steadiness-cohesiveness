### For logging data for FimifMap Implementation

# from measure import *

import numpy as np

class SNCVis:

    def __init__(self, fimif):
        self.missing_log = fimif.missing_log
        self.false_log = fimif.false_log

        self.__false_aggregation()
        self.__missing_aggregation()

    def __false_aggregation(self):
        self.false_log_aggregated = []
        for log in self.false_log:
            sum_dict = {}
            num_dict = {}
            for i, point_indices in enumerate(log["idx"]):
                for point_idx in point_indices:
                    if point_idx in sum_dict:
                        sum_dict[str(point_idx)] += log["value"][i]
                        num_dict[str(point_idx)] += 1
                    else:
                        sum_dict[str(point_idx)] = log["value"][i]
                        num_dict[str(point_idx)] = 1
            
            for key in sum_dict:
                sum_dict[key] /= num_dict[key]
                sum_dict[key] = sum_dict[key].tolist()
            self.false_log_aggregated.append(sum_dict)



            # sum_list = np.array([.0,.0])
            # for i, __ in enumerate(log["value"]):
            #     sum_list += log["direction"][i] * log["value"][i] 
            # self.false_log_aggregated.append(sum_list.tolist())
        

    def __missing_aggregation(self):
        self.missing_log_aggregated = []
        for log in self.missing_log:
            sum_dict = {}
            num_dict = {}
            for i, point_indices in enumerate(log["idx"]):
                for point_idx in point_indices:
                    if point_idx in sum_dict:
                        sum_dict[str(point_idx)] += log["value"][i]
                        num_dict[str(point_idx)] += 1
                    else:
                        sum_dict[str(point_idx)] = log["value"][i]
                        num_dict[str(point_idx)] = 1
            
            for key in sum_dict:
                sum_dict[key] /= num_dict[key]
                sum_dict[key] = sum_dict[key].tolist()
            self.missing_log_aggregated.append(sum_dict)

