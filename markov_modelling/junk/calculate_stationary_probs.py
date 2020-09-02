import pandas as pd
import pdb
import numpy as np
import msmtools
import constants
import os
from msmtools.estimation import is_connected
from markov_utils import train_markov_chain,print_stationary_distributions,transform_transition_matrix_connected



if __name__ == '__main__':

    # Read the input data
    input_directory = constants.output_data_segment_keyword_matrix

    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index)

    input_directory = constants.output_data_markov_modelling
    output_directory = constants.output_data_markov_modelling

    metadata_fields = ['complete','complete_m','complete_w','CountryOfBirth','CountryOfBirth_m','CountryOfBirth_w','easy_w','easy_m','medium_m','medium_w','hard_m','hard_w',"notwork","notwork_m","notwork_w","work","work_m","work_w"]

    np.set_printoptions(suppress=True)

    for metadata_field in metadata_fields:

        transition_matrix = np.loadtxt(input_directory+metadata_field+'/transition_matrix.np',dtype=np.float64)
        pdb.set_trace()
        assert np.allclose(transition_matrix.sum(axis=1), 1)
        assert msmtools.analysis.is_transition_matrix(transition_matrix)
        assert is_connected(transition_matrix)
        mm = train_markov_chain(transition_matrix)
        stationary_prob = print_stationary_distributions(mm,features_df.KeywordLabel.to_list())
        stationary_prob.to_csv(output_directory+metadata_field+'/stationary_probs.csv')


