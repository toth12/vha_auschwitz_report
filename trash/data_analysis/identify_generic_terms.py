import constants
import pandas as pd
import codecs
import csv
import pdb
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import datetime
from scipy import stats, integrate
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from random import randint
import json
pd.set_option('display.max_rows', 200)

def check_if_empty(segments):
    generic_terms = ['14231', '14223', '13930', '13929', '13214', '14230', '14229', '13926', '14049', '14605', '14307', '16285', '12498', '13215', '13931', '14241', '14235', '13310', '7601', '14233', '7528', '17206', '14232', '13018', '16192', '7624', '14226', '14225']
    keyword_ids = segments['KeywordID'].tolist()
    keyword_ids_filtered = [element for element in keyword_ids if element not in generic_terms]
    if len(keyword_ids_filtered)>0:
        return False
    else:
        return True



if __name__ == '__main__':

    # Get the input file names and input directories
    # Input files are the segments data and the biodata about each interviewee

    input_directory = constants.input_data
    output_directory = constants.output_data_features
    input_files = constants.input_files_segments
    bio_data = constants.input_files_biodata

    # Add the full path to the input files
    input_files = [input_directory+i for i in input_files]


    # Read the segment input files into panda dataframe
    df = pd.concat([pd.read_csv(el) for el in input_files])
    

    keyword_counts = df.groupby(['KeywordID','KeywordLabel'])['KeywordID'].count().to_frame(name="Count").reset_index().sort_values("Count",ascending=False).reset_index()


    segments = df.groupby("SegmentID")["KeywordID"].unique().to_frame(name="KeywordID").reset_index()
    segments['empty'] = segments.apply(check_if_empty,axis=1)

    for element in segments[segments['empty']==True]:

        pdb.set_trace()


