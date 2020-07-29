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
features_df = pd.read_csv(input_directory+'keyword_index_merged_segments.csv')
segment_df = pd.read_csv(input_directory+'segment_index_merged.csv')
features = features_df['KeywordLabel'].values.tolist()

# Open the topics
with codecs.open("data_analysis/topics.txt") as topics_file:
        topics = topics_file.read()

topic_list = topics.split('\n\n')

anchors = []
# Check if topic word exists
for topic in topic_list:
    anchor = []
    topic_words = topic.split('\n')
    for topic_word in topic_words:
        if not topic_word.strip() in features_df['KeywordLabel'].tolist():
            pdb.set_trace()
        else:
            anchor.append(topic_word.strip())
    anchors.append(anchor)







# Sparse matrices are also supported
X = ss.csr_matrix(data)



# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=len(anchors))  # Define the number of latent (hidden) topics to use.


topic_model.fit(X, docs=segment_df.updated_id.tolist(),words=features,anchors=anchors, anchor_strength=10)

topics = topic_model.get_topics()

for topic_n,topic in enumerate(topics):
    words,mis = zip(*topic)
    topic_str = str(topic_n)+': '+','.join(words)
    print(topic_str)

top_docs = topic_model.get_top_docs()
for topic_n, topic_docs in enumerate(top_docs):
    docs,probs = zip(*topic_docs)
    docs = [str(element)for element in docs] 
    topic_str = str(topic_n+1)+': '+','.join(docs)
    print(topic_str)

document_topic_matrix = topic_model.labels.astype(int)
np.where(document_topic_matrix.sum(1)==12)[0].shape
print (np.where(document_topic_matrix.sum(1)==0)[0].shape)

(unique, counts) = np.unique(document_topic_matrix,axis=0, return_counts=True)





new_topics = []
for i,element in enumerate(unique):
    nonzeros = np.count_nonzero(element)
    if nonzeros >1:
        if counts[i] >50:

            print (counts[i])
            new_topic = 'topic_'+'_'.join([str(num) for num in np.nonzero(element)[0].tolist()])
            print (new_topic)
            new_topics.append(new_topic)

# Assign a topic to each segment
#doc_indices = np.argsort(-topic_model.log_p_y_given_x.T[6],axis=0)
#doc_indices = np.argsort(-topic_model.log_z.T[6],axis=0)

#doc_indices = np.argsort(-topic_model.log_z.T[6],axis=0)

count_segment_without_topic = 0

segment_topics = []
for i,element in enumerate(document_topic_matrix):
    nonzeros = np.count_nonzero(element)
    if nonzeros > 1:
        topic = 'topic_'+'_'.join([str(num) for num in np.nonzero(element)[0].tolist()])
        if (topic not in new_topics):
            count_segment_without_topic = count_segment_without_topic+1
            segment_topics.append("unknown_topic")
        else:
            segment_topics.append(topic)
    elif nonzeros ==1:
        topic = 'topic_'+str(np.nonzero(element)[0].tolist()[0])
        segment_topics.append(topic)
    else:
        
        segment_topics.append("unknown_topic")

segment_df['topic'] = segment_topics

segment_df.to_csv('data/output/topic_sequencing/segment_topics.csv')

pdb.set_trace()

