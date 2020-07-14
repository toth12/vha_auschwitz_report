import igraph
from igraph import*
import pandas as pd
import pdb
import numpy as np
import constants
import markov_clustering as mc
import networkx as nx
from pprint import pprint 
import pyintergraph
from networkx.algorithms import community


input_directory = constants.output_data_segment_keyword_matrix
data = np.loadtxt('transition_matrix.txt')
nodes = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index_100)
stats = pd.read_csv('stat.csv')
stats = stats[0:300]
stats  =stats.rename(columns={'topic_name':'KeywordLabel'})
nodes = nodes.rename(columns={'Unnamed: 0':'index_original'})

del stats['Unnamed: 0']
stats_with_indices =stats.merge(nodes)
stats_with_indices = stats_with_indices.sort_values("index_original")

stats_with_indices =stats_with_indices.reset_index()
indices = stats_with_indices.index_original.to_list()

new_data = np.take(data,indices,axis=0)
new_data = np.take(new_data,indices,axis=1)

pdb.set_trace()