#!/usr/bin/env python
# coding: utf-8

import pdb
import numpy as np
import pandas as pd
import os
import constants
#from markov_modelling import markov_utils as mu
import markov_utils as mu
from tables import *
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
    features_df = features_df.drop(columns=['index','Unnamed: 0'])
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
            
            print (key)
            
            indices = metadata_partitions[key]


            input_data_set = np.take(data,indices)
            # Make sure that interviews with only one segment are not included
            for i in range(0,input_data_set.shape[0]):
                assert (input_data_set[i].shape[0]>1)
                
            
            output_directory = output_directory_temp+key
           
            # Make the output directory
            try:
                os.mkdir(output_directory)
            except:
                pass
           
            # Estimate fuzzy trajectories
            #empyt = [element[0] for element in input_data_set if element[0].sum()==0]
            trajs = mu.estimate_fuzzy_trajectories(input_data_set)
            # Visualize implied timescale and save it
            mu.visualize_implied_time_scale(trajs,output_directory+'/implied_time_scale.png')
            mu.visualize_implied_time_scale_bayes(trajs, output_directory+'/implied_time_scales_bay.png')


            # Estimate the Markov model from the trajectories
            msm = mu.estimate_markov_model_from_trajectories(trajs)

            stat_dist_error = mu.estimate_pi_error(trajs, msm, ntrails=25)
            stat_dist_error.to_csv(output_directory + '/stat_dist_error.csv', index=False)
            
            # Create histogram to compare stationary distribution with plausibility measure
            mu.prepare_histogram_to_compare_stationary_distribution_with_plausi_measure(msm, output_directory + '/histogram.png')

            # delete discrete trajectories from msm object (severe speedup with saving/loading)
            msm._dtrajs_active = None
            msm._dtrajs_full = None
            msm._dtrajs_original = None

            # Save the markov model
            msm.save(output_directory+'/pyemma_model', 'simple',overwrite=True)

            bmsm = mu.estimate_bayesian_markov_model_from_trajectories(trajs)
            bmsm._dtrajs_active = None
            bmsm._dtrajs_full = None
            bmsm._dtrajs_original = None
            bmsm.save(output_directory+'/pyemma_model_bayes', 'simple', overwrite=True)


            # Get the stationary distributions
            stationary_probs = mu.print_stationary_distributions(msm,features_df.KeywordLabel.to_list())
            pd.DataFrame(stationary_probs).to_csv(output_directory+'/stationary_probs.csv')
        except KeyboardInterrupt:
            print('Keyboard interrupt, quitting.')
            import sys
            sys.exit()

        #except:
        #    print ("Training the Markov model with the following metadata field was not possible")
        #    print (key)
