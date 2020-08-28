import pandas as pd 
import pdb
from itertools import islice
import numpy as np
from pyemma import msm,plots
import msmtools
from msmtools.flux import tpt,ReactiveFlux
from pyemma import plots as mplt
import constants
from scipy import sparse
from sklearn import preprocessing
import os
import argparse
import itertools 
from msmtools.estimation import connected_sets,is_connected,largest_connected_submatrix
from scipy.special import softmax


from markov_utils import window,cg_transition_matrix,train_markov_chain,print_stationary_distributions

stationary_probs = []

if __name__ == '__main__':
    metadata_fields = ['complete','complete_m','complete_w','CountryOfBirth','CountryOfBirth_m','CountryOfBirth_w','easy_w','easy_m','medium_m','medium_w','hard_m','hard_w',"notwork","notwork_m","notwork_w","work","work_m","work_w"]
    metadata_fields_to_agregate = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_fields', nargs='+')
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "metadata_fields"):
            for field in value:
                if (field not in metadata_fields):
                    print ("The following metadata_field is not valid")
                    print (field)
                    pdb.set_trace()
                else:
       
                    metadata_fields_to_agregate.append(field)
    
    # Read the input data
    input_directory = constants.output_data_segment_keyword_matrix

    output_directory = constants.output_data_markov_modelling_aggregated_reports
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index)
    del features_df['Unnamed: 0']

    input_directory = constants.output_data_markov_modelling
    stationary_probs_complete = pd.read_csv(input_directory+'/'+'complete'+'/'+'stationary_probs.csv')
    statrionary_prob_selection = stationary_probs_complete[stationary_probs_complete['topic_name']=='camp selections']['stationary_prob'].values[0]

    for element in metadata_fields_to_agregate:
        stationary_probs = pd.read_csv(input_directory+'/'+element+'/'+'stationary_probs.csv')
        del stationary_probs['Unnamed: 0']
        friends_index = stationary_probs[stationary_probs['topic_name']=="friends"]['stationary_prob'].index
        friendship_stationary_prob = stationary_probs[stationary_probs['topic_name']=="friendships"]['stationary_prob'].values[0]
        
        friends_stationary_prob=stationary_probs.iloc[friends_index,stationary_probs.columns.get_loc("stationary_prob")].values[0]
        stationary_probs.iloc[friends_index,stationary_probs.columns.get_loc("stationary_prob")] =friendship_stationary_prob+friends_stationary_prob
        
        stationary_probs = stationary_probs.rename(columns={'topic_name':'KeywordLabel'})
        stationary_probs = stationary_probs.rename(columns={'stationary_prob':'stationary_prob_'+element})
        stationary_probs['stationary_prob_norm_'+element] = statrionary_prob_selection / stationary_probs['stationary_prob_'+element] 
        features_df = features_df.merge(stationary_probs)

    output_file_name = ('|'.join(metadata_fields_to_agregate))+'.csv'
    features_df.to_csv(output_directory+output_file_name)
    