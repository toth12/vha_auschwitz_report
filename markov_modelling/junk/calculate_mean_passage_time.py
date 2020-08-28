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
import pyemma
from tables import *
from markov_utils import window,cg_transition_matrix,train_markov_chain,print_stationary_distributions,calculate_flux, calculate_mean_passage_time_between_states

stationary_probs = []

if __name__ == '__main__':
    metadata_fields = ['complete','complete_m','complete_w','easy_w','easy_m','medium_m','medium_w','hard_m','hard_w',"notwork","notwork_m","notwork_w","work","work_m","work_w"]
    


    bio_data = constants.input_files_biodata_birkenau
    df_biodata = pd.read_csv(constants.input_data+bio_data)
    df_biodata = df_biodata.fillna(0)
    country_of_origins = df_biodata.groupby('CountryOfBirth')['CountryOfBirth'].count().to_frame('Count').reset_index()
    country_of_origins= country_of_origins[country_of_origins.Count>50]
    countries = []
    for element in country_of_origins.CountryOfBirth.to_list():
        metadata_fields.append(element+'_w')
        metadata_fields.append(element+'_m')



    metadata_fields = metadata_fields + country_of_origins.CountryOfBirth.to_list()
    
    # Read the input data
    input_directory = constants.output_data_segment_keyword_matrix

    output_directory = constants.output_data_markov_modelling_aggregated_reports
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index)
    #state_index = features_df.KeywordLabel.to_list()
    

    input_directory = constants.output_data_markov_modelling
    stationary_probs_complete = pd.read_csv(input_directory+'/'+'complete'+'/'+'stationary_probs.csv')

    for element in metadata_fields:
        print (element)
        try:
            mm = pyemma.load(input_directory+'/'+element+'_temp/'+'pyemma_model','simple')
            data = mm.P
            stats = stationary_probs_complete[0:100]
            stats  =stats.rename(columns={'topic_name':'KeywordLabel'})
            nodes = features_df.rename(columns={'Unnamed: 0':'index_original'})


            del stats['Unnamed: 0']
            stats_with_indices = stats.merge(nodes)
            stats_with_indices = stats_with_indices.sort_values("index_original")

            stats_with_indices =stats_with_indices.reset_index()


            original_indices = stats_with_indices.index_original.to_list()
            
            new_data = np.take(data,original_indices,axis=0)
            new_data = np.take(new_data,original_indices,axis=1)


            state_index = stats_with_indices.KeywordLabel.to_list()
            new_data = new_data +1e-12
            new_data = (new_data /new_data.sum(axis=1,keepdims=1))

            mm = train_markov_chain(new_data)
            
            pdb.set_trace()
            mean_p = calculate_mean_passage_time_between_states(mm,state_index)
            mean_p.to_csv(input_directory+'/'+element + '_temp/mean_passage.csv')
        except:
            print ("The following metadata field could not be processed")
            print (element)
            
        
    