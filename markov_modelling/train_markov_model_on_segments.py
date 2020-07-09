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

from train_markov_model_on_labeled_segments import window



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
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index)

    # Read the row index (groups of three segments) of the matrix above
    segment_df = pd.read_csv(input_directory+ constants.output_segment_keyword_matrix_document_index)

    

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

    segment_df = segment_df[segment_df.IntCode.isin(df_biodata.IntCode.to_list())]
    index=segment_df.index.to_list()
    data = np.take(data,index,axis=0)
    (unique, counts) = np.unique(data,axis=0, return_counts=True)
    trajectories = []

    for i,element in enumerate(data):
        print (i)
        trajectory=np.where(np.all(unique==element,axis=1))[0][0]
        trajectories.append(trajectory)

    tr=[el for el in window(trajectories)]
    count_matrix = np.zeros((unique.shape[0],unique.shape[0]))

    for element in tr:
        count_matrix[element[0],element[1]]=count_matrix[element[0],element[1]]+1
    pdb.set_trace()
    transition_matrix = (count_matrix / count_matrix.sum(axis=1,keepdims=1))
