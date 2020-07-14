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
from train_markov_model_on_labeled_segments import window,cg_transition_matrix,train_markov_chain,print_stationary_distributions,calculate_flux


input_directory = constants.output_data_segment_keyword_matrix
data = np.loadtxt('transition_matrix1.txt')
nodes = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index_100)
nodes = nodes.rename(columns={'Unnamed: 0':'index_original'})
topic_labels = pd.read_csv(constants.input_data+'topic_labels_index.csv')
nodes['topic']=np.nan

anchors = []
for i,element in enumerate(topic_labels.iterrows()):
    topic_list = element[1]['KeywordLabels_text'].split(',')
    final_topic_list = []
    for el in topic_list:
        if "|" in el:
            el = ','.join(el.split('|'))
            if len(nodes[nodes.KeywordLabel==el].index)>0:
                index = nodes[nodes.KeywordLabel==el].index[0]
                nodes['topic'][index]=element[1]['topic']
                final_topic_list.append(el)

            
        else:
            if len(nodes[nodes.KeywordLabel==el].index)>0:
                index = nodes[nodes.KeywordLabel==el].index[0]
                nodes['topic'][index]=element[1]['topic']
                final_topic_list.append(el)
nodes = nodes.dropna()
nodes = nodes.sort_values("index_original")


nodes = nodes.reset_index()

original_indices = nodes.index_original.to_list()

new_data = np.take(data,original_indices,axis=0)
new_data = np.take(new_data,original_indices,axis=1)

del nodes['index']

new_data = new_data +1e-12
new_data = (new_data /new_data.sum(axis=1,keepdims=1))

assert np.allclose(new_data.sum(1),1)
topics = nodes.topic.unique().tolist()
binary_map = np.zeros(shape=(new_data.shape[0],len(topic_labels)))
for element in nodes.iterrows():
    label_index = element[0]
    topic_index = topics.index(element[1].topic)
    binary_map[label_index][topic_index]=1
transition_m=cg_transition_matrix(new_data,binary_map)
m=train_markov_chain(transition_m)
stat=print_stationary_distributions(m,topics)
print(calculate_flux(m,topics,['food'],['socialrelations']))
pdb.set_trace()
