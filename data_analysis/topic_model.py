import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
import pandas as pd
import pdb
from corextopic import vis_topic as vt
import constants
import codecs
import json
import prince
import matplotlib.pyplot as plt


input_directory = constants.output_data_features
data = np.loadtxt(input_directory+'segment_keyword_matrix_merged_birkenau.txt', dtype=int)
features_df = pd.read_csv(input_directory+'keyword_index_merged_segments_birkenau.csv')
segment_df = pd.read_csv(input_directory+'segment_index_merged_birkenau.csv')
features = features_df['KeywordLabel'].values.tolist()
node_filters = constants.output_data_filtered_nodes + "node_filter_1_output.json"




# Sparse matrices are also supported
X = ss.csr_matrix(data)
# Word labels for each column can be provided to the model
# Document labels for each row can be provided

#anchors=['camp adaptation methods']
# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=15)  # Define the number of latent (hidden) topics to use.


topic_model.fit(X, docs=segment_df.updated_id.tolist(),words=features)

topics = topic_model.get_topics()
for topic_n,topic in enumerate(topics):
    words,mis = zip(*topic)
    topic_str = str(topic_n+1)+': '+','.join(words)
    print(topic_str)

top_docs = topic_model.get_top_docs()
for topic_n, topic_docs in enumerate(top_docs):
    docs,probs = zip(*topic_docs)
    docs = [str(element)for element in docs] 
    topic_str = str(topic_n+1)+': '+','.join(docs)
    print(topic_str)


vt.vis_rep(topic_model, column_label=features, prefix='topic-model-example')
plt.figure(figsize=(10,5))
plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16);
plt.savefig('topics.png')

pdb.set_trace()
# Train the first layer
topic_model = ct.Corex(n_hidden=18)
topic_model.fit(X)

# Train successive layers
tm_layer2 = ct.Corex(n_hidden=3)
tm_layer2.fit(topic_model.labels)

tm_layer3 = ct.Corex(n_hidden=1)
tm_layer3.fit(tm_layer2.labels)

vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=features, max_edges=300, prefix='topic-model-hierarchical')
