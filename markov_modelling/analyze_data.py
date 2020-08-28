import pandas as pd 
import pdb
import numpy as np
import constants
import argparse
import pyemma
from tables import *



stationary_probs = []

from markov_utils import visualize_most_important_paths,calculate_flux,print_mean_passage_time


def print_flux(flux):
    for tr in flux:
        print (tr)
        print (flux[tr])

if __name__ == '__main__':
    
    data_directory = constants.output_data_markov_modelling_aggregated_reports
    input_directory = constants.output_data_segment_keyword_matrix
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index)

    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_partition', nargs='+')
    
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "metadata_partition"):
            metadata_partition = value[0]
    
    state_index = features_df.KeywordLabel.to_list()
    input_directory = constants.output_data_markov_modelling
    stationary_probs_complete = pd.read_csv(input_directory+'/'+'complete_temp'+'/'+'stationary_probs.csv')

    # Load the input data
    print ('Loading of the input dataset began; be patient this can take up to 10 - 15 mins')  
    mm = pyemma.load(input_directory+metadata_partition+'_temp/'+'pyemma_model','simple')
    print ("Input data loaded; use the functions as described in the README")
    output_directory = input_directory+metadata_partition+'_temp/'
    calculate_flux(mm,state_index,['friends'],['camp food sharing'],0.1)
    print_mean_passage_time(mm,state_index,'friends',20)
    pdb.set_trace()