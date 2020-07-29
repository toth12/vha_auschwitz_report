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
#new_data = (new_data /new_data.sum(1))

q_df = pd.DataFrame(new_data,columns=stats_with_indices.KeywordLabel.to_list(), index=stats_with_indices.KeywordLabel.to_list())


def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(q_df)
#pprint(edges_wts)


G=nx.DiGraph() 

# nodes correspond to states
G.add_nodes_from( stats_with_indices.KeywordLabel.to_list())
indices = stats_with_indices.KeywordLabel.to_list()

# edges represent transition probabilities
'''for k, v in edges_wts.items():
    tmp_origin, tmp_destination = k[0], k[1]
    G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
'''



labels={}
edge_labels={}
count = 0
for i, origin_state in enumerate(new_data):
    for j, destination_state in enumerate(origin_state):
        rate = new_data[i][j]

        
        if rate > 0:
            count = count+1
            try:
                
                G.add_edge(indices[i],indices[j],
                           weight=rate)
            except:
                pdb.set_trace()
print ('ol')
matrix = nx.to_scipy_sparse_matrix(G)

for inflation in [i / 10 for i in range(15, 26)]:
    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)
    Q = mc.modularity(matrix=result, clusters=clusters)
    print("inflation:", inflation, "modularity:", Q)
#communities_generator = community.girvan_newman(G)
#top_level_communities = next(communities_generator)

result = mc.run_mcl(matrix)           # run MCL with default parameters
clusters = mc.get_clusters(result)

pdb.set_trace()
GG=pyintergraph.nx2igraph(G,labelname="node_label")
clusters = nx.clustering(G,weight='weight')


    