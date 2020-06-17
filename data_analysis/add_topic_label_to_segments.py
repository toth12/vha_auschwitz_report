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

# Read the input data
input_directory = constants.output_data_features

# Read the segment index term matrix
data = np.loadtxt(input_directory+'segment_keyword_matrix.txt', dtype=int)

# Read the column index (index terms) of the matrix above
features_df = pd.read_csv(input_directory+'feature_index.csv')

# Read the row index (groups of three segments) of the matrix above
segment_df = pd.read_csv(input_directory+'document_index.csv')

# Get the index labels
features = features_df['KeywordLabel'].values.tolist()

# Open the index terms for the anchored topic modelling
with codecs.open("data_analysis/topic_anchors_birkenau.txt") as topics_file:
        topics = topics_file.read()

topic_list = topics.split('\n\n')

anchors = []


# Check if index term certainly exists
for topic in topic_list:
    anchor = []
    topic_words = ' '.join(topic.split("\n")[1].split(' ')[1:]).split(',')
    for topic_word in topic_words:
        if topic_word =='Mengele Josef':
            anchor.append('Mengele, Josef')
        elif len(topic_word.strip())==0:
            pass
        elif not topic_word.strip() in features_df['KeywordLabel'].tolist():
            pdb.set_trace()
        else:
            anchor.append(topic_word.strip())
    anchors.append(anchor)

# Create a sparse matrix from the numpy segment index term matrix

X = ss.csr_matrix(data)
pdb.set_trace()
# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=len(anchors))  # Define the number of latent (hidden) topics to use.


topic_model.fit(X, docs=segment_df.new_segment_id.tolist(),words=features,anchors=anchors, anchor_strength=10)

topics = topic_model.get_topics()

# Print the key topics

for topic_n,topic in enumerate(topics):
    words,mis = zip(*topic)
    topic_str = str(topic_n)+': '+','.join(words)
    print(topic_str)

# Print the documents most relevant for each topic
top_docs = topic_model.get_top_docs()
for topic_n, topic_docs in enumerate(top_docs):
    docs,probs = zip(*topic_docs)
    docs = [str(element)for element in docs] 
    topic_str = str(topic_n+1)+': '+','.join(docs)
    print(topic_str)


# Get the document topic matrix
document_topic_matrix = topic_model.labels.astype(int)

# Helper functions below to find out if there documents that are labeled with a given number of topics

#p.where(document_topic_matrix.sum(1)==12)[0].shape
#print (np.where(document_topic_matrix.sum(1)==0)[0].shape)


# Find all possible topic combinations (a document can have multiple topics) and how many times they occurs
(unique, counts) = np.unique(document_topic_matrix,axis=0, return_counts=True)




# Create new topics that are the combinations of multiple single topics

# Create an empty list that will hold these new topics
new_topics = []

# Iterate through all possible topic combinations (each is numpy array)
for i,element in enumerate(unique):
    # Find out how many individual topics there are in each topic combination
    nonzeros = np.count_nonzero(element)

    # Topic combinations that consists of only one topic is one of the anchored topic above, no need to create a new topic from
    if nonzeros >1:
        # Count how many instances this topic combination has in the data
        # If occurs more than 50 times, add it to the list of new topics
        if counts[i] >0:
            # Every topic is to be named with a prefix (topic_) followed by all topics that give the topic combination
            new_topic = 'topic_'+'_'.join([str(num) for num in np.nonzero(element)[0].tolist()])
            new_topics.append(new_topic)

# With the new topics (in fact topic combinations), as well as with the anchored topics, define the topic label of each segment

# To follow how many topics have unknown topic (see below what we mean by unknown topic make a counter)
count_segment_without_topic = 0

segment_topics = []

# Iterate through the document topic matrix
for i,element in enumerate(document_topic_matrix):
    # Find the topics of it
    nonzeros = np.count_nonzero(element)

    # If it has more than one topic label
    if nonzeros > 1:
        # Create the topic label
        topic = 'topic_'+'_'.join([str(num) for num in np.nonzero(element)[0].tolist()])
        # If the topic label does not exist (count of topic combination is below the threshold above), label the segment with unknown topic
        if (topic not in new_topics):
            count_segment_without_topic = count_segment_without_topic+1
            segment_topics.append("unknown_topic")
        # Otherwise label it with the topic combination
        else:
            segment_topics.append(topic)
    elif nonzeros ==1:
        topic = 'topic_'+str(np.nonzero(element)[0].tolist()[0])
        segment_topics.append(topic)
    # If zero topic was added to a segment label it with unknown topic
    else:       
        segment_topics.append("unknown_topic")
        count_segment_without_topic = count_segment_without_topic+1


# Update the segment dataframe and add the topic label to it
# For instance: the topic of 10006_47_48_49 (interview code 10006, segments 47,48,49) is topic_2
#        Unnamed: 0      updated_id                                          KeywordID          topic
 #   0               3  10006_47                         ['10983' '12044' '14280']        topic_2
segment_df['topic'] = segment_topics



# Save the dataframe 
segment_df.to_csv('data/output/topic_sequencing/document_index_with_topic_labels.csv')
