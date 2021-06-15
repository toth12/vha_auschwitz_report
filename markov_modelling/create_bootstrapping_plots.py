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
from collections import Counter
from msmtools.estimation import connected_sets
from matplotlib.pyplot import figure




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
    
    #pi_samples = np.zeros((ntrails, len(orig_msm.nstates)))
    pi_samples = np.zeros((ntrails, orig_msm.count_matrix_full.shape[0]))
    all_states = np.arange(start=0, stop=orig_msm.count_matrix_full.shape[0], step=1)
    for trial in tqdm(range(ntrails)):
        try:
            bs_sample = np.random.choice(len(dtrajs), 
                 size=len(dtrajs), 
                replace=True)
            dtraj_sample = list(np.array(dtrajs)[bs_sample])

            msm = pyemma.msm.estimate_markov_model(dtraj_sample, 
                                                    lag=orig_msm.lag)
            stationary_probs = msm.pi
            if len(connected_sets(msm.count_matrix_full))>1:
                disconnected_states = [element for element in all_states if element not in connected_sets(msm.count_matrix_full)[0]]
                if len(disconnected_states)>0:
                    for element in disconnected_states:
                        stationary_probs = np.insert(stationary_probs,element,0)

            #pi_samples[trial, msm.active_set] = stationary_probs
            
            pi_samples[trial, all_states] = stationary_probs
        except Exception as e:
            pdb.set_trace()
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_fields', nargs='+')
    # Load the input data

    # Read the segment index term matrix
    data = np.load(constants.segment_keyword_matrix.replace('.txt', '.npy'), 
                  allow_pickle=True)

    

    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(constants.segment_keyword_matrix_feature_index)

    metadata_partitions_file = constants.metadata_partitions
    


    
    # Read the metadata partitions
    with open(metadata_partitions_file) as read_file:
        metadata_partitions = json.load(read_file)

    metadata_fields_to_agregate = []
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "metadata_fields"):
            for field in value:
                if (field not in metadata_partitions.keys()):
                    print ("The following metadata_field is not valid")
                    print (field)
                    pdb.set_trace()
                else:
                    metadata_fields_to_agregate.append(field)


    # Set the output directory
    output_directory_temp = constants.output_data_markov_modelling

 
    samples = {}
    msms = {}
    state_indices = {}

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
    ntrails = 100
    for key in metadata_fields_to_agregate:
        indices = metadata_partitions[key]

        input_data_set = np.take(data,indices)
        # Make sure that interviews with only one segment are not included
        for i in range(0,input_data_set.shape[0]):
            assert (input_data_set[i].shape[0]>1)
        new_input_data_set=[]

        # Eliminate empty steps
        for interview in input_data_set:
            new_interview = []
            for row in interview:
                if row.sum()>0:
                    new_interview.append(row)
            new_input_data_set.append(np.vstack(new_interview))
        input_data_set = new_input_data_set

        # Estimate fuzzy trajectories
        trajs = mu.estimate_fuzzy_trajectories(input_data_set)

        # Estimate the Markov model from the trajectories
        msm = mu.estimate_markov_model_from_trajectories(trajs,msmlag=1)
        
        error_est = estimate_pi_error(trajs, msm, return_samples=True, ntrails=ntrails)
        topic_labels_active_set = [features_df.KeywordLabel.to_list()[j] for i, j in enumerate(msm.active_set)]
        samples[key] = error_est
        state_indices[key]=[]
        state_indices[key].extend(topic_labels_active_set)
        msms[key] = msm


    
    aggregate_states = []
    for element in state_indices:
        aggregate_states.extend(state_indices[element])


    joint_states = {x: count for x, count in Counter(aggregate_states).items() if count > 1}
    joint_states = joint_states.keys()
    for index,KeywordLabel in enumerate(features_df.KeywordLabel.to_list()):
        try:
            figure(figsize=(8,8))
            plt.margins(0)
            # Check if it is in the active set for both samples
            if (KeywordLabel in joint_states) == False:
               continue
            else:
                for n, k in enumerate(metadata_fields_to_agregate):
                    #index = state_indices[k].index(KeywordLabel)
                    try:
                        if 'w' in k.split('_'):
                            leg = 'women'
                        else:
                            leg = 'men'
                        state_samples = samples[k][:, index]
                        #plt.figure(figsize=(28,28)) #change your figure size as per your desire heres
                        
                        y, x, _ = plt.hist(state_samples, bins=20, label=f'{leg}', color=f'C{n}')
                        lower_confidence, upper_confidence = confidence_interval(state_samples, 0.68)
                        #plt.vlines(lower_confidence, 0, 10,  color=f'C{n}', linestyle=':', label=f'lower conf {k}')
                        #plt.vlines(upper_confidence, 0, 10,  color=f'C{n}', linestyle='--', label=f'upper conf {k}')
                        
                        plt.vlines(msms[k].pi[msms[k]._full2active[index]], 0, 1400, color='k', label='Model estimate' if n==1 else None,linestyles='dashed')
                        plt.ylabel("Count (R=10000)", size=14)
                        plt.xlabel("Stationary probability", size=14)

                    except:
                        pdb.set_trace()
                
                plt.legend(bbox_to_anchor=(1.05, 1), loc='center left', borderaxespad=0.)
                plt.savefig(output_directory+'/'+KeywordLabel+'.png',bbox_inches='tight')
                plt.clf()
        except:
            pass


