import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
import pandas as pd
import pdb
from corextopic import vis_topic as vt


df =pd.DataFrame.from_csv('Interview_keyword_min_50.csv')
df = df.iloc[:,1:] 

X = df.to_numpy()



# Sparse matrices are also supported
X = ss.csr_matrix(X)
# Word labels for each column can be provided to the model
# Document labels for each row can be provided


# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=3)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=df.columns,anchors=['social_relations', 'social_relations','social_relations'],anchor_strength=5)

topics = topic_model.get_topics()
for topic_n,topic in enumerate(topics):
    words,mis = zip(*topic)
    topic_str = str(topic_n+1)+': '+','.join(words)
    print(topic_str)

# Train the first layer
topic_model = ct.Corex(n_hidden=20)
topic_model.fit(X)

# Train successive layers
tm_layer2 = ct.Corex(n_hidden=5)
tm_layer2.fit(topic_model.labels)

tm_layer3 = ct.Corex(n_hidden=1)
tm_layer3.fit(tm_layer2.labels)

vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=df.columns, max_edges=300, prefix='topic-model-example')
