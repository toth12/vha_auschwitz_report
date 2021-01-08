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
import pdb
import argparse



def estimate_pi_error(dtrajs, orig_msm, ntrails=10, conf_interval=0.68, return_samples=False):
    """
    Estimate boostrap error for stationary probability
    
    :param dtrajs: list of np.array, discrete trajectories
    :param orig_msm: pyemma.msm.MarkovModel
    Only used for reference of lag time and to incorporate ML 
    stationary distribution to data frame
    :param ntrails: int, the number of bootstrap samples to draw. 
    :param conf_interval: float 0 < conf_interval < 1
    
    :return:
    pandas.DataFrame instance containing ML MSM pi and bootstrap error
    """
    from pyemma.util.statistics import confidence_interval
    
    pi_samples = np.zeros((ntrails, orig_msm.nstates))

    for trial in tqdm(range(ntrails)):
        try:
            bs_sample = np.random.choice(len(dtrajs), 
                 size=len(dtrajs), 
                replace=True)
            dtraj_sample = list(np.array(dtrajs)[bs_sample])

            msm = pyemma.msm.estimate_markov_model(dtraj_sample, 
                                                    lag=orig_msm.lag)
            pi_samples[trial, msm.active_set] = msm.pi
        except Exception as e: 
            print(e)
            
    if return_samples:
        return pi_samples
    
    
    std = pi_samples.std(axis=0)
    lower_confidence, upper_confidence = confidence_interval(pi_samples, conf_interval)
    
    probabilities = pd.DataFrame(np.array([orig_msm.active_set, 
                                           orig_msm.pi, 
                                           std, 
                                           lower_confidence, 
                                           upper_confidence]).T,
                    columns=['State', 'StatDist', 'Std', 'LowerConf', 'UpperConf'], )
    
    # type cast to int
    probabilities['State'] = probabilities['State'].astype(int)
    
    
    return probabilities

if __name__ == '__main__':
    # python markov_modelling/create_bootstrapping_plots.py --metadata_fields complete_m complete_w
    metadata_fields = ['complete','complete_m','complete_w','easy_w','easy_m','medium_m','medium_w','hard_m','hard_w',"notwork","notwork_m","notwork_w","work","work_m","work_w"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_fields', nargs='+')
    metadata_fields_to_agregate = []
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "metadata_fields"):
            for field in value:
                if (field not in metadata_fields):
                    print ("The following metadata_field is not valid")
                    print (field)
                    pdb.set_trace()
                else:
                    metadata_fields_to_agregate.append(field)
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

    samples = {}
    msms = {}

    # Make the output directory
    output_directory = constants.output_data_markov_modelling + 'bootstrap/'+ '_'.join(metadata_fields_to_agregate)
    
    try:
        os.mkdir(constants.output_data_markov_modelling + 'bootstrap/')
    except:
        pass
    try:
        os.mkdir(output_directory)
    except:
        pass
    for key in metadata_fields_to_agregate:
        indices = metadata_partitions[key]

        input_data_set = np.take(data,indices)
        # Make sure that interviews with only one segment are not included
        for i in range(0,input_data_set.shape[0]):
            assert (input_data_set[i].shape[0]>1)

        # Estimate fuzzy trajectories
        trajs = mu.estimate_fuzzy_trajectories(input_data_set)

        # Estimate the Markov model from the trajectories
        msm = mu.estimate_markov_model_from_trajectories(trajs)
        
        error_est = estimate_pi_error(trajs, msm, return_samples=True, ntrails=25)
        
        samples[key] = error_est
        msms[key] = msm

    

    for index,KeywordLabel in enumerate(features_df.KeywordLabel.to_list()):
        for n, k in enumerate(metadata_fields_to_agregate):
            state_samples = samples[k][:, index]
            plt.hist(state_samples, bins=20, label=f'sample dist {k}', color=f'C{n}')
            
            lower_confidence, upper_confidence = confidence_interval(state_samples, 0.68)
            plt.vlines(lower_confidence, 0, 10,  color=f'C{n}', linestyle=':', label=f'lower conf {k}')
            plt.vlines(upper_confidence, 0, 10,  color=f'C{n}', linestyle='--', label=f'upper conf {k}')
            plt.vlines(msms[k].pi[index], 0, 10, color='k', label='ML estimate' if n==1 else None)

        plt.legend()
        plt.savefig(output_directory+'/'+KeywordLabel+'.png')
        plt.clf()


