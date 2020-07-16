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

    # Read the input data
    input_directory = constants.output_data_segment_keyword_matrix

    output_directory = constants.output_data_markov_modelling_aggregated_reports
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index_100)
    del features_df['Unnamed: 0']
    keywords = []
    metadata_fields = ['complete','complete_m','complete_w','CountryOfBirth','CountryOfBirth_m','CountryOfBirth_w','easy_w','easy_m','medium_m','medium_w','hard_m','hard_w',"notwork","notwork_m","notwork_w","work","work_m","work_w"]
    
    metadata_fields_to_agregate = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_fields', nargs='+')
    parser.add_argument('--keywords', nargs='+')
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "metadata_fields"):
            for field in value:
                if (field not in metadata_fields):
                    print ("The following metadata_field is not valid")
                    metadata_fields_to_agregate.append(field)
                    
                else:
       
                    metadata_fields_to_agregate.append(field)
        
        if (key == "keywords"):
            for keyword in value:
                if (keyword not in features_df.KeywordLabel.to_list()):
                    print ("The following keyword is not valid")
                    print (keyword)
                    pdb.set_trace()
                else:
       
                    keywords.append(keyword)
    
    # Read the input data
    input_directory = constants.output_data_segment_keyword_matrix

    output_directory = constants.output_data_markov_modelling_aggregated_reports
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index_100)
    del features_df['Unnamed: 0']

    input_directory = constants.output_data_markov_modelling
    

    for keyword in keywords:
        print (keyword)
        for element in metadata_fields_to_agregate:
            pdb.set_trace()
            mean_p = pd.read_csv(input_directory+'/'+element+'/'+'mean_passage.csv')
            mean_p = mean_p.set_index('Unnamed: 0')
            print (mean_p[keyword].sort_values()[0:40])
            
            

  
    