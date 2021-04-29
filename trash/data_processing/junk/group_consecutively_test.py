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



def group_into_consecutive_lists(data):
    # sort the data, for simplicity
    data = sorted(data)


    
    if len(data)==2:
        if data[1]-data[0]==1:
            return [data]
        else:
            return np.nan

    if len(data)>2:
        
        # create a list of the gaps between the consecutive values
        gaps = [y - x for x, y in zip(data[:-1], data[1:])]
        # have python calculate the standard deviation for the gaps
        sd = stdev(gaps)
        pdb.set_trace()
        if sd ==0:
            return [data]

        else:

            # create a list of lists, put the first value of the source data in the first
            lists = [[data[0]]]
            for x in data[1:]:
                # if the gap from the current item to the previous is more than 1 SD
                # Note: the previous item is the last item in the last list
                # Note: the '> 1' is the part you'd modify to make it stricter or more relaxed
                if (x - lists[-1][-1]) / sd > 1:
                    # then start a new list
                    lists.append([])
                # add the current item to the last list in the list
                lists[-1].append(x)

            
            result = [element for element in lists if len(element)>1]
            if len(result)>0:
                return result
            else:
                return np.nan
        

    else:
        return np.nan

if __name__ == '__main__':
    case_1 = [0, 2, 4]
    case_1_res_exp =  [[0],[2],[4]]
    case_1_res = group_into_consecutive_lists(case_1)
    pdb.set_trace()
    assert case_1_res_exp == case_1_res
    