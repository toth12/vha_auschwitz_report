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


    bio_data = constants.input_files_biodata_birkenau
    df_biodata = pd.read_csv(constants.input_data+bio_data)
    df_biodata = df_biodata.fillna(0)
    country_of_origins = df_biodata.groupby('CountryOfBirth')['CountryOfBirth'].count().to_frame('Count').reset_index()
    country_of_origins= country_of_origins[country_of_origins.Count>50]
    metadata_fields = metadata_fields + country_of_origins.CountryOfBirth.to_list()

    # Read the input data
    input_directory = constants.output_data_segment_keyword_matrix

    output_directory = constants.output_data_markov_modelling_aggregated_reports
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index)
    del features_df['Unnamed: 0']
    keywords = []
    metadata_fields_to_agregate = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_fields', nargs='+')
    parser.add_argument('--keywords', nargs='+')
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "metadata_fields"):
            for field in value:
                if (field not in metadata_fields):
                    print ("The following metadata_field is not valid")
                    print (field)
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
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index)
    del features_df['Unnamed: 0']

    input_directory = constants.output_data_markov_modelling

    stationary_probs_complete = pd.read_csv(input_directory+'/'+'complete'+'/'+'stationary_probs.csv')
    statrionary_prob_selection = stationary_probs_complete[stationary_probs_complete['topic_name']=='camp selections']['stationary_prob'].values[0]
    statrionary_prob_escape = stationary_probs_complete[stationary_probs_complete['topic_name']=='camp escapes']['stationary_prob'].values[0]
    for keyword in keywords:
        print (keyword)
        for element in metadata_fields_to_agregate:
            stationary_probs = pd.read_csv(input_directory+'/'+element+'/'+'stationary_probs.csv')
            del stationary_probs['Unnamed: 0']

            if keyword == "friends":
                new_value = stationary_probs[stationary_probs['topic_name']==keyword]['stationary_prob'].values[0] + stationary_probs[stationary_probs['topic_name']=="friendships"]['stationary_prob'].values[0]

            try:
                if keyword == "camp food sharing":
                    new_value = stationary_probs[stationary_probs['topic_name']==keyword]['stationary_prob'].values[0] + stationary_probs[stationary_probs['topic_name']=="food sharing"]['stationary_prob'].values[0]
                
            except: 
                new_value = np.nan 
            stationary_probs = stationary_probs.rename(columns={'topic_name':'KeywordLabel'})
            stationary_probs = stationary_probs.rename(columns={'stationary_prob':'stationary_prob_'+element})
            stationary_probs['stationary_prob_norm_sel_'+element] = statrionary_prob_selection / stationary_probs['stationary_prob_'+element]
            stationary_probs['stationary_prob_norm_esca_'+element] =  stationary_probs['stationary_prob_'+element]  / statrionary_prob_escape
           
            

            print (element)
      
            print ('Stationary prob:')
            print (stationary_probs[stationary_probs['KeywordLabel']==keyword]['stationary_prob_'+element].values[0])
            print ('Escape:')
            print (stationary_probs[stationary_probs['KeywordLabel']==keyword]['stationary_prob_norm_esca_'+element].values[0])
            print ("Selection:")
            print (stationary_probs[stationary_probs['KeywordLabel']==keyword]['stationary_prob_norm_sel_'+element].values[0])
           



