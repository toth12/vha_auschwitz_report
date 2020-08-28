import pandas as pd 
import pdb
from itertools import islice
import numpy as np
from pyemma import msm
import msmtools
from msmtools.estimation import connected_sets
import networkx as nx
import pyemma
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

#from utils_test import print_mean_passage_time



def print_mean_passage_time(mm,topic_labels,source,limit = 10):
    
    source_index = topic_labels.index(source)
    topic_labels_active_set = {i:topic_labels[j] for i, j in enumerate(mm.active_set)}
    
    df_passage_times = pd.DataFrame(topic_labels_active_set.items(),columns=['index','topic_labels']).set_index('index')
    mean_ps = []
    for key in topic_labels_active_set:
        try:
            mfpt = pyemma.msm.tpt(mm, [mm._full2active[source_index]],[mm._full2active[mm.active_set[key]]]).mfpt
            mean_ps.append(mfpt)

        except:
            
            mean_ps.append(np.nan)
    df_passage_times['mfpt'] =mean_ps
    df_passage_times = df_passage_times.sort_values('mfpt',ascending=True)
    print ('hello')
    for row in df_passage_times[0:limit].iterrows():
        print (row[1]['topic_labels'])
        print (row[1]['mfpt'])