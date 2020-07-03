import igraph
from igraph import*
import pandas as pd
import pdb
import numpy as np
import constants

data = np.loadtxt(constants.output_data_features+'segment_keyword_matrix_from_preprocessed_data.txt')
your_matrix = data[:-1].T@data[:-1]
del data
vcount = max(your_matrix.shape)
sources, targets = your_matrix.nonzero()

nodes = pd.read_csv(constants.output_data_features+"feature_index_from_preprocessed_data.csv").KeywordLabel.tolist()
edgelist = list(zip(sources.tolist(), targets.tolist()))
g = igraph.Graph(vcount, edgelist,directed=False)
#g = g.simplify(combine_edges='sum')
g.vs['label'] = nodes
print (g.clusters().membership)
pdb.set_trace()
g.es['weight'] = your_matrix[your_matrix.nonzero()]
communities = g.community_edge_betweenness(10,directed=True)
clusters = communities.as_clustering()   

pdb.set_trace()

nodes=pd.read_csv('node_list.csv')
a = pd.DataFrame(data, index=nodes.node.to_list(), columns=nodes.node.to_list())
# Get the values as np.array, it's more convenenient.
A = a.values

# Create graph, A.astype(bool).tolist() or (A / A).tolist() can also be used.
g = igraph.Graph.Weighted_Adjacency(A.tolist())

# Add edge weights and node labels.
#g.es['weight'] = A[A.nonzero()]
g.vs['label'] = nodes.node.to_list() 
#or a.index/a.columns
g.es['weight'] = your_matrix[your_matrix.nonzero()]