import pandas as pd 
import pdb
import numpy as np
import constants
import argparse
import pyemma
from tables import *

from markov_utils import *

stationary_probs = []


def print_flux(flux):
    for tr in flux:
        print (tr)
        print (flux[tr])

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
    features_df = features_df.drop(columns=['Unnamed: 0','index'])

    metadata_fields_to_agregate = []
    sources = []
    targets = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_fields', nargs='+')
    parser.add_argument('--source', nargs='+')
    parser.add_argument('--target', nargs='+')
    parser.add_argument('--flux', nargs='+')
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
        if (key == "flux"):
            flux = float(value[0])
    
    
    state_index = features_df.KeywordLabel.to_list()

    input_directory = constants.output_data_markov_modelling
    stationary_probs_complete = pd.read_csv(input_directory+'/'+'complete'+'/'+'stationary_probs.csv')

    for element in metadata_fields_to_agregate:
        print (element)
      
        mm = pyemma.load(input_directory+element+'/'+'pyemma_model','simple')
        output_file_name_source = '_'.join(sources[0].split())
        output_file_name_target = '_'.join(targets[0].split())
        output_file_name = output_file_name_source + '_' + output_file_name_target+'.png'
        output_directory = input_directory+element+ '/'+ 'tpt_visualization'
        visualize_most_important_paths(mm,flux,features_df,sources[0],targets[0],input_directory+element+ '/')

        
        
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
        
        #flux = calculate_flux(mm,state_index,sources,targets)

        #python markov_modelling/visualize_pathways.py --metadata_fields complete_m --source story_beginning --target story_ending --flux 0.2

        
        
