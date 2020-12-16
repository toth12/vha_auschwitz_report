"""Replaces those keywords that occur in less than 25 interviews with their respective parent node

As output it constructs a new segment data containing only Jewish survivors and the simplified keywords.
"""


import json
import constants
import pdb
import codecs
from anytree.importer import DictImporter
from anytree import search
import csv
import pandas as pd
from tqdm.auto import tqdm
import numpy as np



if __name__ == '__main__':

    story_beginning_ids = [13310,14233,7601,7528,10983,16328,14232,14226,16123]
    input_directory = constants.input_data
    output_directory = input_directory
    
    input_files = constants.input_files_segments
    input_files = [input_directory+i for i in input_files]

    df = pd.concat([pd.read_csv(el) for el in input_files])

    story_beginning_patterns = ['deportation from','deportation to','transfer from','transfer to Auschwitz']

    for element in story_beginning_patterns:
        relevant_ids = df[df.KeywordLabel.str.contains(element)].KeywordID.drop_duplicates().tolist()
        story_beginning_ids.extend(relevant_ids)



    ### Cast datatypes

    ## Start with segments data

    df_numeric = ['IntCode','SegmentID','InTapenumber','OutTapenumber','KeywordID','SegmentNumber']
    df_string = ['IntervieweeName','KeywordLabel']
    df_time = ['InTimeCode','OutTimeCode']


    # Cast to numeric
    for col in df_numeric:
        df[col] = pd.to_numeric(df[col])

    # Cast to string
    for col in df_string:
        df[col] = df[col].astype('string')

    # Cast to temporal

    for col in df_time:
        df[col] = pd.to_datetime(df[col], format="%H:%M:%S:%f")

    df_segments = df.groupby(['IntCode'])['SegmentNumber'].unique().to_frame().reset_index()
    df_segments['min'] = df_segments['SegmentNumber'].apply(lambda x: np.array(x).min())
    df_segments['max'] = df_segments['SegmentNumber'].apply(lambda x: np.array(x).max())
    to_drop = []
    for row in tqdm(df_segments[0:20].iterrows()):
        int_code = row[1]["IntCode"]
        minimum = row[1]["min"]
        first = df[(df.IntCode==int_code)&(df.SegmentNumber==minimum)]
        first_filtered = first[first.KeywordID.isin(story_beginning_ids)]
        
        if first_filtered.shape[0] > 0:
            for i,element in enumerate(first_filtered.index.to_list()):
                if i ==0:
                    story_beginning=df.iloc()[element].copy()
                    story_beginning['KeywordID']=88888
                    story_beginning['KeywordLabel']='story_beginning'
                    df.iloc()[element] = story_beginning
                else:

                    to_drop.append(element)




        maximum = row[1]["max"]

    df = df.drop(df.index[to_drop])
    pdb.set_trace()