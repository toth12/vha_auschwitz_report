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
from train_markov_model_on_labeled_segments import window,cg_transition_matrix,train_markov_chain,print_stationary_distributions,calculate_flux, calculate_mean_passage_time_between_states

input_files = ['transition_matrix.txt','transition_matrix0.txt','transition_matrix1.txt']

for i,element in enumerate(input_files):
    input_directory = constants.output_data_segment_keyword_matrix
    data = np.loadtxt(element)
    nodes = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index_100)

    topic_labels = pd.read_csv(constants.input_data+'topic_labels_index.csv')

    stats = pd.read_csv('stat.csv')
    stats = stats[0:100]
    stats  =stats.rename(columns={'topic_name':'KeywordLabel'})
    nodes = nodes.rename(columns={'Unnamed: 0':'index_original'})

    del stats['Unnamed: 0']
    stats_with_indices = stats.merge(nodes)
    stats_with_indices = stats_with_indices.sort_values("index_original")

    stats_with_indices =stats_with_indices.reset_index()


    original_indices = stats_with_indices.index_original.to_list()

    new_data = np.take(data,original_indices,axis=0)
    new_data = np.take(new_data,original_indices,axis=1)


    state_index =  stats_with_indices.KeywordLabel.to_list()
    new_data = new_data +1e-12
    new_data = (new_data /new_data.sum(axis=1,keepdims=1))

    mm = train_markov_chain(new_data)
    mean_p = calculate_mean_passage_time_between_states(mm,state_index)

    mean_p.to_csv('mean_p_+'+str(i)+'.csv')
    

    print (calculate_flux(mm,state_index,['friends'],['camp food sharing']))
    pdb.set_trace()

