#!/usr/bin/env python
# coding: utf-8

import pdb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pyemma
from tqdm.notebook import tqdm
import sys
import constants
import random
import msmtools
from markov_modelling import markov_utils as mu
import json




if __name__ == '__main__':
    # Load the input data
    input_directory = constants.output_data_segment_keyword_matrix

    # Read the segment index term matrix
    data = np.load(input_directory + constants.output_segment_keyword_matrix_data_file.replace('.txt', '.npy'), 
                  allow_pickle=True)

    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory + 
                          constants.output_segment_keyword_matrix_feature_index)

    # Create the row index  of the matrix above
    segment_df = pd.read_csv(input_directory + 
                         constants.output_segment_keyword_matrix_document_index)

    int_codes = segment_df['IntCode'].to_list()


    # Set the output directory
    output_directory_temp = constants.output_data_markov_modelling

    # Read the metadata partitions
    with open(input_directory + "metadata_partitions.json") as read_file:
        metadata_partitions = json.load(read_file)

    for key in metadata_partitions:
        try:
            indices = metadata_partitions[key]
            input_data_set = np.take(data,indices)

            #### This is temporary code

            random_sample=random.sample(range(0,len(data)),100)
            random_sample.sort()
            input_data_set=np.take(data,random_sample)


            #### End of temporary code

            output_directory = output_directory_temp+key+'_temp'

            # Make the output directory
            try:
                os.mkdir(output_directory)
            except:
                pass

            # Estimate fuzzy trajectories
            trajs = mu.estimate_fuzzy_trajectories(input_data_set)

            # Visualize implied timescale and save it
            mu.visualize_implied_time_scale(trajs,output_directory+'/implied_time_scale.png')

            # Estimate the Markov model from the trajectories
            msm = mu.estimate_markov_model_from_trajectories(trajs)

            # Save the markov model

            if os.path.exists(output_directory+'/pyemma_model'):
                os.remove(output_directory+'/pyemma_model')
                   
            msm.save(output_directory+'/pyemma_model', 'simple',overwrite=True)

            # Create histogram to compare stationary distribution with plausibility measure

            mu.prepare_histogram_to_compare_stationary_distribution_with_plausi_measure(msm,trajs, output_directory + '/histogram.png')

            # Get the stationary distributions
            stationary_probs = mu.print_stationary_distributions(msm,features_df.KeywordLabel.to_list())
            pd.DataFrame(stationary_probs).to_csv(output_directory+'/stationary_probs.csv')
        except:
            print ("Training the Markov model with the following metadata field was not possible")
            print (key)
