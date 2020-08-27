import pandas as pd 
import pdb
import numpy as np
import constants
import argparse
import pyemma
from tables import *

from markov_utils import window,cg_transition_matrix,train_markov_chain,print_stationary_distributions,calculate_flux,calculate_mean_passage_time_between_states

stationary_probs = []

if __name__ == '__main__':
    metadata_fields = ['complete','complete_m','complete_w','easy_w','easy_m','medium_m','medium_w','hard_m','hard_w',"notwork","notwork_m","notwork_w","work","work_m","work_w"]
    




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

    metadata_fields_to_agregate = []
    sources = []
    targets = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_fields', nargs='+')
    parser.add_argument('--source', nargs='+')
    parser.add_argument('--target', nargs='+')
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "metadata_fields"):
            for field in value:
                if (field not in metadata_fields):
                    print ("The following metadata_field is not valid")
                    print (field)
                    pdb.set_trace()
                else:
       
                    metadata_fields_to_agregate.append(field)
        
        if (key == "source"):
            for keyword in value:
                if (keyword not in features_df.KeywordLabel.to_list()):
                    print ("The following keyword is not valid")
                    print (keyword)
                    pdb.set_trace()
                else:
       
                    sources.append(keyword)
        if (key == "target"):
            for keyword in value:
                if (keyword not in features_df.KeywordLabel.to_list()):
                    print ("The following keyword is not valid")
                    print (keyword)
                    pdb.set_trace()
                else:
       
                    targets.append(keyword)
    
    
    state_index = features_df.KeywordLabel.to_list()

    input_directory = constants.output_data_markov_modelling
    stationary_probs_complete = pd.read_csv(input_directory+'/'+'complete'+'/'+'stationary_probs.csv')

    for element in metadata_fields_to_agregate:
        print (element)
      
        mm = pyemma.load(input_directory+element+'_temp/'+'pyemma_model','simple')

        
        
        '''
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
        

       '''
        pdb.set_trace()
        flux = calculate_flux(mm,state_index,sources,targets)

        for tr in flux:
            print (tr)
            print (flux[tr])
        pdb.set_trace()

