import pyemma
import constants
import pandas as pd
import os
from IPython.display import display
import json
import os
from tables import *
import pdb

def print_stationary_distributions(mm, topic_labels):
    #Print the stationary distributions of the top states
    results = []
    pdb.set_trace()
    #topic_labels_active_set = {i:topic_labels[j] for i, j in enumerate(mm.active_set)}
    for i, element in enumerate(mm.pi.argsort()[::-1]):
        #print (i)
        #print (topic_labels[element])
        #print (mm.pi[element])
        #print ('\n')
        try:
            results.append({'topic_name':topic_labels[element],'stationary_prob':mm.pi[element]})
        except:
            pdb.set_trace()
    return results

input_directory = constants.output_data_markov_modelling

path = os.getcwd()
parent = os.path.abspath(os.path.join(path, os.pardir))
input_directory = constants.output_data_markov_modelling
input_directory = "data/output_aid_giving_sociability_expanded/markov_modelling/"
msm = pyemma.load(input_directory+'notwork_w'+'/'+'pyemma_model','simple')
output_directory = input_directory+'notwork_w'+'/'
 # Load the input data
input_directory = constants.output_data_segment_keyword_matrix
input_directory = 'data/output_aid_giving_sociability_expanded/segment_keyword_matrix/'
# Read the column index (index terms) of the matrix above
features_df = pd.read_csv(input_directory + 
                          constants.output_segment_keyword_matrix_feature_index)
features_df = features_df.drop(columns=['index','Unnamed: 0'])

result = print_stationary_distributions(msm,features_df.KeywordLabel.to_list())
df = pd.DataFrame(result)
#df.to_csv(output_directory+'/stationary_probs.csv')
print (df[df.topic_name=='friends'])
pdb.set_trace()