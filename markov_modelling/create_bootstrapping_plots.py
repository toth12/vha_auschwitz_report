#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import constants
import markov_utils as mu
from tables import *
import json
import pyemma
import tables
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
from pyemma.util.statistics import confidence_interval



if __name__ == '__main__':
    # Load the input data
    input_directory = '../' + constants.output_data_segment_keyword_matrix

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

    samples = {}
    msms = {}
    for key in ['complete_m', 'complete_w']:
        indices = metadata_partitions[key]

        input_data_set = np.take(data,indices)
        # Make sure that interviews with only one segment are not included
        for i in range(0,input_data_set.shape[0]):
            assert (input_data_set[i].shape[0]>1)

        # Estimate fuzzy trajectories
        trajs = mu.estimate_fuzzy_trajectories(input_data_set)

        # Estimate the Markov model from the trajectories
        msm = mu.estimate_markov_model_from_trajectories(trajs)
        
        error_est = estimate_pi_error(trajs, msm, return_samples=True, ntrails=50)
        
        samples[key] = error_est
        msms[key] = msm

    food_sharing_index = 34
    
    for n, k in enumerate(['complete_m', 'complete_w']):
        state_samples = samples[k][:, food_sharing_index]
        plt.hist(state_samples, bins=20, label=f'sample dist {k}', color=f'C{n}')
        
        lower_confidence, upper_confidence = confidence_interval(state_samples, 0.68)
        plt.vlines(lower_confidence, 0, 10,  color=f'C{n}', linestyle=':', label=f'lower conf {k}')
        plt.vlines(upper_confidence, 0, 10,  color=f'C{n}', linestyle='--', label=f'upper conf {k}')
        plt.vlines(msms[k].pi[food_sharing_index], 0, 10, color='k', label='ML estimate' if n==1 else None)
    
        plt.legend()
        plt.title('food sharing')




