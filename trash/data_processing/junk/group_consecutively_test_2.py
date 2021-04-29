"""Creates a segment-keyword matrix, feature index, and document; leaves only those keyword that in at least 100 interviews
"""

import constants
import pandas as pd
import pdb
import numpy as np
import argparse
from tqdm.auto import tqdm
import pdb
from statistics import stdev
import more_itertools as mit

def group_segments(data,max_gap=2):
    result = [list(group) for group in mit.consecutive_groups(data)]
    #check if merge is possible
    if len(result)>1:
        new_result = []
        for i,group in enumerate(result):
            if i == 0:
                new_result.append(group)
            else:
                first_element = group[0]
                last_element_prev_group = result[i-1][-1]
                
                if (first_element - last_element_prev_group) <=max_gap:
                    new_result[-1].extend(group)
                else:
                    new_result.append(group)
        return new_result

    else:    
        return result


if __name__ == '__main__':
    
    case = [0, 1, 2]
    case_res_exp =  [[0,1,2]]
    case_res = group_segments(case)
    assert case_res_exp == case_res


    case = [0, 1, 3,4]
    case_res_exp =  [[0, 1, 3,4]]
    case_res = group_segments(case)
    assert case_res_exp == case_res



    case = [0, 1, 3,4,6,7]
    case_res_exp =  [[0, 1, 3,4,6,7]]
    case_res = group_segments(case)
    assert case_res_exp == case_res

    case = [0, 1, 3,4,7,19]
    case_res_exp =  [[0, 1, 3,4],[7],[19]]
    case_res = group_segments(case)
    assert case_res_exp == case_res
    