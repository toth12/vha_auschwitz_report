"""Replaces those keywords that occur in less than 25 interviews with their res
pective parent node

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
pd.set_option('display.max_rows', 500)

#40279

if __name__ == '__main__':

    #story_beginning_ids = [13310,14233,7601,7528,10983,16328,14232,14226,16123]
    #story_ending_ids = [13310,14233,7601,7528,16297,16192,13929,14226,14232,13930,16162]
    story_beginning_ids = []
    story_ending_ids = []
    input_directory = constants.input_data
    output_directory = input_directory
    
    input_files = constants.input_files_segments
    input_files = [input_directory+i for i in input_files]

    df = pd.concat([pd.read_csv(el) for el in input_files])
    df = df.reset_index()

    '''
    df = df_orig[0:500000]
    df = df.append(df_orig[df_orig.IntCode==40279])
    df = df.reset_index()
    '''
 

    


    story_beginning_patterns = ['deportation to Auschwitz','transfer to Auschwitz']
    story_ending_patterns = ['transfer from Auschwitz','camp liberation']

    for element in story_beginning_patterns:
        relevant_ids = df[df.KeywordLabel.str.contains(element)].KeywordID.drop_duplicates().tolist()
        story_beginning_ids.extend(relevant_ids)

    for element in story_ending_patterns:
        relevant_ids = df[df.KeywordLabel.str.contains(element)].KeywordID.drop_duplicates().tolist()
        story_ending_ids.extend(relevant_ids)



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
    '''
    df['index_original'] = df.index
    segments_to_change = df[df.KeywordID.isin(story_beginning_ids)]
    segments_to_change['KeywordLabel']="arrival"
    segments_to_change['KeywordID']=88888
    #df = df.drop(segments_to_change.index)
    df = df.append(segments_to_change)
    '''
    segments_to_change = df[df.KeywordID.isin(story_beginning_ids)]
    print ("Number of iterations needed: "+str(len(segments_to_change)))
   
    for row_index in tqdm(segments_to_change.index):
        story_beginning=df.iloc()[row_index ].copy()
        story_beginning['KeywordID']=88888
        story_beginning['KeywordLabel']='story_beginning'
        df.iloc()[row_index]=story_beginning
   

    segments_to_change = df[df.KeywordID.isin(story_ending_ids)]

    print ("Number of iterations needed: "+str(len(segments_to_change)))
    
    for row_index in tqdm(segments_to_change.index):
        story_beginning=df.iloc()[row_index ].copy()
        story_beginning['KeywordID']=9999
        story_beginning['KeywordLabel']='story_ending'
        df.iloc()[row_index]=story_beginning

    df.to_csv(output_directory+constants.input_files_segments_story_end_beginning_distinguished[0])

    pdb.set_trace()

    df_segments = df.groupby(['IntCode'])['SegmentNumber'].unique().to_frame().reset_index()
    df_segments['min'] = df_segments['SegmentNumber'].apply(lambda x: np.array(x).min())
    df_segments['max'] = df_segments['SegmentNumber'].apply(lambda x: np.array(x).max())
    to_drop = []
    print ("Number of iterations needed: "+str(len(df_segments)))
    
    for row in tqdm(df_segments.iterrows()):
        int_code = row[1]["IntCode"]
        minimum = row[1]["min"]
        maximum = row[1]["max"]


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


        last = df[(df.IntCode==int_code)&(df.SegmentNumber==maximum)]
        last_filtered = last[last.KeywordID.isin(story_ending_ids)]
        
        if last_filtered.shape[0] > 0:
            for i,element in enumerate(last_filtered.index.to_list()):
                if i ==0:
                    story_ending=df.iloc()[element].copy()
                    story_ending['KeywordID']=9999
                    story_ending['KeywordLabel']='story_ending'
                    df.iloc()[element] = story_ending
                else:

                    to_drop.append(element)

    df = df.drop(df.index[to_drop])
    print (df[df.IntCode==40279])
    pdb.set_trace()
    df.to_csv(output_directory+constants.input_files_segments_story_end_beginning_distinguished[0])

    # Convenience method to check if the process came through as expected
    
    '''
    for interview in df_segments[0:20].iterrows():
        pdb.set_trace()
        print (df[df.IntCode==interview[1]['IntCode']])

    '''
        