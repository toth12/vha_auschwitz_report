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

input_directory = constants.output_data_features
data = np.loadtxt(input_directory+'segment_keyword_matrix_merged.txt', dtype=int)
features_df = pd.DataFrame.from_csv(input_directory+'keyword_index_merged_segments.csv')
segment_df = pd.DataFrame.from_csv(input_directory+'segment_index_merged.csv')
features = features_df['KeywordLabel'].values.tolist()
node_filters = constants.output_data_filtered_nodes + "node_filter_1_output.json"

with codecs.open(node_filters) as json_file:
        new_features = json.load(json_file)
covering_term = "social_relations"
anchors = []
for element in new_features:
    if covering_term in element.keys():
        for term in element[covering_term]:
            try:
                anchors.append(features_df[features_df.KeywordID==term]['KeywordLabel'].values[0])
            except:
                pass

mca = prince.MCA(n_components=2,  n_iter=3,copy=True,check_input=True,engine='auto',random_state=42)
pdb.set_trace()

# Sparse matrices are also supported
X = ss.csr_matrix(data)
# Word labels for each column can be provided to the model
# Document labels for each row can be provided

#anchors=['camp adaptation methods']
# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=20)  # Define the number of latent (hidden) topics to use.
anchors.append("camp menstruation")

topic_model.fit(X, docs=segment_df.updated_id.tolist(),words=features,anchors=["camp adaptation methods","camp adaptation methods","camp adaptation methods","camp adaptation methods","camp adaptation methods"],anchor_strength=10)
#topic_model.fit(X, words=features)
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

pdb.set_trace()
# Train the first layer
topic_model = ct.Corex(n_hidden=18)
topic_model.fit(X)

# Train successive layers
tm_layer2 = ct.Corex(n_hidden=5)
tm_layer2.fit(topic_model.labels)

tm_layer3 = ct.Corex(n_hidden=1)
tm_layer3.fit(tm_layer2.labels)

vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=features, max_edges=300, prefix='topic-model-hierarchical')
