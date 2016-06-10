'''
Created on 10/06/2016
Utils for patient_predict module
@author: Andres Moreno B
'''

import numpy as np
def find_new_explore_c(exploreC, best_val):
    index_val = exploreC.index(best_val)
    distance_next = float('inf') if index_val == len(exploreC) - 1 else float(exploreC[index_val + 1] - exploreC[index_val])
    distance_prev = float('inf') if index_val == 0 else float(exploreC[index_val] - exploreC[index_val - 1])
    new_range = min(distance_next, distance_prev)
    min_val = 0.0 if best_val - new_range < 0 else float(best_val - new_range)
    max_val = float(best_val + new_range)
    exploreC = np.linspace(min_val, max_val, len(exploreC)).tolist()
    return exploreC
