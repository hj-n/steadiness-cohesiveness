# class which finds the fimifpath for input point
# fimifpath consists of 

import numpy as np


class FimifPath:
    def __init__(self, y1, y2):
        self.false_groups = []
        self.missing_groups = []
        self.y1 = y1
        self.y2 = y2        ## initial 2D coordinates
        self.trace = [[y1, y2]]
        
    def get_trace(self):
        return self.trace
        
    def add_max_dists(self, dist_max_x, dist_max_y):
        self.dist_max_x = dist_max_x
        self.dist_max_y = dist_max_y
        
    def add_group(self, group_size_mine, group_size_yours, max_mu, min_mu, ND_dist, centroid_mine, centroid_yours, is_false):
        new_group = {
            "group_size_mine": group_size_mine,
            "group_size_yours": group_size_yours,
            "max_mu": max_mu,
            "min_mu": min_mu,
            "ND_dist": ND_dist / self.dist_max_x,    ## already nomalized
            "centroid_mine": centroid_mine,
            "centroid_yours": centroid_yours,
            "current_loss": None,
        }
        if is_false:
            self.false_groups.append(new_group)
        else:
            self.missing_groups.append(new_group)
    
    def F(self, x):
        return x * x
    
    def F_grad(self, x):
        return 2 * x

    def optimize(self, step=1000):
        
        lr = 0.00005

        false_ratio = 1   # monotonic decrease
        
        
        for i in range(step):
            
            ## forward pass
            false_loss_sum = 0
            false_weight_sum = 0
            
            false_y1_grad_sum = 0
            false_y2_grad_sum = 0 

            for fg in self.false_groups:
                ## forward path
                LD_dist = np.linalg.norm(fg["centroid_mine"] - fg["centroid_yours"])
                mu_group = fg["ND_dist"] - LD_dist / self.dist_max_y
                distortion = (mu_group - fg["min_mu"]) / (fg["max_mu"] - fg["min_mu"]) if mu_group > 0 else 0
                group_loss = fg["group_size_mine"] * fg["group_size_yours"] * self.F(distortion)
                false_weight_sum += fg["group_size_mine"] * fg["group_size_yours"]
                false_loss_sum += group_loss
                
                ## backpropagation
                if (mu_group <= 0):
                    continue
                gradient_constant = (fg["group_size_yours"] * self.F_grad(distortion)) / (fg["max_mu"] - fg["min_mu"])
                dist_grad_g1 = - (fg["centroid_mine"][0] - fg["centroid_yours"][0]) / LD_dist  
                g1_grad_y1 = 1 
                dist_grad_g2 = - (fg["centroid_mine"][1] - fg["centroid_yours"][1]) / LD_dist
                g2_grad_y2 = 1
                
                y1_temp_grad = gradient_constant * dist_grad_g1 * g1_grad_y1
                y2_temp_grad = gradient_constant * dist_grad_g2 * g2_grad_y2
                
                false_y1_grad_sum += y1_temp_grad
                false_y2_grad_sum += y2_temp_grad
                
            missing_loss_sum = 0
            missing_weight_sum = 0
            
            missing_y1_grad_sum = 0
            missing_y2_grad_sum = 0
                
            for fg in self.missing_groups:
                ## forward path
                LD_dist = np.linalg.norm(fg["centroid_mine"] - fg["centroid_yours"])
                mu_group = - fg["ND_dist"] + LD_dist / self.dist_max_y
                distortion = (mu_group - fg["min_mu"]) / (fg["max_mu"] - fg["min_mu"]) if mu_group > 0 else 0
                group_loss = fg["group_size_mine"] * fg["group_size_yours"] * self.F(distortion)
                missing_weight_sum += fg["group_size_mine"] * fg["group_size_yours"]
                missing_loss_sum += group_loss
                
                ## backpropagation
                if (mu_group <= 0):
                    continue
                gradient_constant = (fg["group_size_yours"] * self.F_grad(distortion)) / (fg["max_mu"] - fg["min_mu"])
                dist_grad_g1 = (fg["centroid_mine"][0] - fg["centroid_yours"][0]) / LD_dist  
                g1_grad_y1 = 1 
                dist_grad_g2 = (fg["centroid_mine"][1] - fg["centroid_yours"][1]) / LD_dist
                g2_grad_y2 = 1
                
                y1_temp_grad = gradient_constant * dist_grad_g1 * g1_grad_y1
                y2_temp_grad = gradient_constant * dist_grad_g2 * g2_grad_y2
                
                missing_y1_grad_sum += y1_temp_grad
                missing_y2_grad_sum += y2_temp_grad
            
            ## update step
            ## position update
            
            y1_grad = missing_y1_grad_sum * (1 - false_ratio) + false_y1_grad_sum * false_ratio
            y2_grad = missing_y2_grad_sum * (1  -false_ratio) + false_y2_grad_sum * false_ratio

            false_ratio = (1 - ((i + 1) / step))

            # y1_grad = y1_grad / (missing_weight_sum + false_weight_sum)
            # y2_grad = y2_grad / (missing_weight_sum + false_weight_sum)
            
            self.y1 -= lr * y1_grad
            self.y2 -= lr * y2_grad
            self.trace.append([self.y1, self.y2])
            ## centroid update
            for fg in self.false_groups:
                fg["centroid_mine"] = (fg["centroid_mine"] * fg["group_size_mine"] - self.trace[-2] + self.trace[-1]) / fg["group_size_mine"]
            for fg in self.missing_groups:
                fg["centroid_mine"] = (fg["centroid_mine"] * fg["group_size_mine"] - self.trace[-2] + self.trace[-1]) / fg["group_size_mine"]
                

        #     print(missing_loss_sum, false_loss_sum, missing_loss_sum + false_loss_sum)
        # print(self.trace)
            
        
        # centroid = centroid - self.trace[-1]