import pandas as pd 
import pdb
from itertools import islice
import numpy as np
from pyemma import msm,plots
import msmtools
from msmtools.flux import tpt,ReactiveFlux
from pyemma import plots as mplt
import constants
from scipy import sparse
from sklearn import preprocessing
import os
import argparse
import itertools 
from msmtools.estimation import connected_sets,is_connected,largest_connected_submatrix
from scipy.special import softmax


from train_markov_model_on_labeled_segments import window,cg_transition_matrix,train_markov_chain,print_stationary_distributions

stationary_probs = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', nargs='+')
    parser.add_argument('--to', nargs='+')
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "from"):
            if (value is not None):
                path_start = value
            else:
                path_start = ['12']
        if (key == "to"):
            if (value is not None):
                path_end = value
            else:
                path_end = ['17','13']


    metadata_fields = ['complete','CountryOfBirth','easy','medium','hard',"not_work","work"]
    #metadata_fields = ['complete']  


    np.set_printoptions(suppress=True)


    # Read the input data
    input_directory = constants.output_data_segment_keyword_matrix

    # Read the segment index term matrix
    data = np.loadtxt(input_directory+ constants.output_segment_keyword_matrix_data_file_100, dtype=int)

    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index_100)

    # Read the row index (groups of three segments) of the matrix above
    segment_df = pd.read_csv(input_directory+ constants.output_segment_keyword_matrix_document_index_100)

    

    # Read the input data
    input_directory = constants.input_data
    

    main_output_directory = constants.output_data_topic_sequences+'paths/'+'_'.join(path_start)+'|'+'_'.join(path_end)+'/'

    try:
        os.mkdir(main_output_directory)
    except:
        pass


    path_start = [int(el) for el in path_start]
    path_end = [int(el) for el in path_end]

    # Current work directory

    path = os.getcwd()

    # Read the biodata
    bio_data = constants.input_files_biodata_birkenau
    df_biodata = pd.read_csv(input_directory+bio_data)
    df_biodata = df_biodata.fillna(0)

    IntCodeM = df_biodata[df_biodata.Gender=='M']['IntCode'].to_list()
    IntCodeW = df_biodata[df_biodata.Gender=='F']['IntCode'].to_list()

    int_codes_list = [IntCodeM]

    for d,int_codes in enumerate(int_codes_list):
        
        input_interviews = segment_df[segment_df.IntCode.isin(int_codes)]
        index=input_interviews.index.to_list()
        input_matrix = np.take(data,index,axis=0)
        (unique, counts) = np.unique(input_matrix,axis=0, return_counts=True)
        trajectories = []

        for i,element in enumerate(input_matrix):
            print (i)
            trajectory=np.where(np.all(unique==element,axis=1))[0][0]
            trajectories.append(trajectory)

        tr=[el for el in window(trajectories)]
        count_matrix = np.zeros((unique.shape[0],unique.shape[0])).astype(float)

        for element in tr:
            count_matrix[element[0],element[1]]=count_matrix[element[0],element[1]]+float(1)
      
        count_matrix = count_matrix +1e-12
        transition_matrix = (count_matrix / count_matrix.sum(axis=1,keepdims=1))
        assert np.allclose(transition_matrix.sum(axis=1), 1)
        assert msmtools.analysis.is_transition_matrix(transition_matrix)
        assert is_connected(transition_matrix)

        binary_map = (unique / unique.sum(axis=1,keepdims=1))
        new_tra = cg_transition_matrix(transition_matrix,binary_map)
        new_tra[np.isnan(new_tra)] = 0
        new_tra = new_tra+1e-12
        new_tra= (new_tra / new_tra.sum(axis=1,keepdims=1))
        np.savetxt('transition_matrix'+str(d+1), new_tra, fmt='%.8f')

        assert np.allclose(new_tra.sum(axis=1), 1)
        mm = train_markov_chain(new_tra)
        stationary_prob = print_stationary_distributions(mm,features_df.KeywordLabel.to_list())
        stationary_probs.append(stationary_prob)

    pdb.set_trace()
    pd.merge(pd.DataFrame(stationary_probs[0]),pd.DataFrame(stationary_probs[1]),on='topic_name',suffixes=['_woman','_man']).to_csv('stationary_probs.csv')
        
