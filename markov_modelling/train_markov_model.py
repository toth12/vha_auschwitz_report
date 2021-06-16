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
from msmtools.estimation import connected_sets




if __name__ == '__main__':
    # Load the input data

    # Read the segment index term matrix
    data = np.load(constants.segment_keyword_matrix.replace('.txt', '.npy'), 
                  allow_pickle=True)
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(constants.segment_keyword_matrix_feature_index)
    features_df = features_df.drop(columns=['Unnamed: 0'])
    

    # Metadata partitions file
    metadata_partitions_file = constants.metadata_partitions


    # Set the output directory
    output_directory_temp = constants.output_data_markov_modelling

    # Read the metadata partitions
    with open(metadata_partitions_file) as read_file:
        metadata_partitions = json.load(read_file)
    failed = []
    for key in metadata_partitions:
        try:
            
            print (key)
            #if (key !="Netherlands") and (key !="Netherlands_w"):
                #continue

            
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

            # check input data for long empty segments and split interview there
            # currently split if gap is longer than 2 minutes
            new_input_data_set = []
            for interview in input_data_set:
                _new_input_data_set = mu.split_interview(interview, max_gap=2)
                # make sure to exclude single segments
                new_input_data_set += [_int for _int in _new_input_data_set if len(_int) > 1]
            

            # temp
            #pdb.set_trace()
            #trajs = mu.estimate_fuzzy_trajectories(new_input_data_set[99:100])
            #msm = mu.estimate_markov_model_from_trajectories(trajs,msmlag=1)

            # temp


            trajs = mu.estimate_fuzzy_trajectories(new_input_data_set)
   
            # Visualize implied timescale and save it
            mu.visualize_implied_time_scale(trajs,output_directory+'/implied_time_scale.png')
            mu.visualize_implied_time_scale_bayes(trajs, output_directory+'/implied_time_scales_bay.png')


            # Estimate the Markov model from the trajectories
            msm = mu.estimate_markov_model_from_trajectories(trajs,msmlag=1)
            '''
            if not (msm.count_matrix_full.shape[0] ==len(features_df.KeywordLabel.to_list())):
                pdb.set_trace()

            '''
            
            # Get the active set
            topic_labels_active_set = [features_df.KeywordLabel.to_list()[j] for i, j in enumerate(msm.active_set)]
            df_active_states = pd.DataFrame(topic_labels_active_set,columns=['KeywordLabel'])
            df_active_states.to_csv(output_directory+'/state_index.csv')
            #msm._full2active[97]
    
            #stat_dist_error = mu.estimate_pi_error(trajs, msm, ntrails=25)
            #stat_dist_error.to_csv(output_directory + '/stat_dist_error.csv', index=False)
            
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
            stationary_probs = mu.print_stationary_distributions(msm,df_active_states.KeywordLabel.to_list())

            #pdb.set_trace()
            pd.DataFrame(stationary_probs).to_csv(output_directory+'/stationary_probs.csv')
            
        except:

            print('Keyboard interrupt, quitting.')
            import sys
            failed.append(key)
            continue
        print (failed)

