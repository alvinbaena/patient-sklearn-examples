'''
Created on 10/06/2016
Utils for patient_predict module
@author: Andres Moreno B
'''

import numpy as np
def find_new_explore_c(paramArray, best_val):
    index_val = paramArray.index(best_val)
    distance_next = float('inf') if index_val == len(paramArray) - 1 else float(paramArray[index_val + 1] - paramArray[index_val])
    distance_prev = float('inf') if index_val == 0 else float(paramArray[index_val] - paramArray[index_val - 1])
    if index_val!=0 and index_val!=len(paramArray)-1:
        paramArray = np.linspace(paramArray[index_val - 1]+distance_prev/2, paramArray[index_val + 1]-distance_next/2, len(paramArray)).tolist()
            
        
   
    else:
        new_range=min(distance_next,distance_prev)
        min_val = 0.0001 if best_val - new_range <= 0 else float(best_val - new_range)
        max_val = float(best_val + new_range)
        if index_val==0:
            paramArray = np.linspace(min_val, paramArray[index_val + 1]-distance_next/2, len(paramArray)).tolist()
            
        else:
            paramArray = np.linspace(paramArray[index_val - 1]+distance_prev/2, max_val, len(paramArray)).tolist()
        
    
    print paramArray
    return paramArray