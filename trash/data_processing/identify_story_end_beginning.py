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



if __name__ == '__main__':

    story_beginning_ids = []
    story_ending_ids = []
    input_directory = constants.input_data
    output_directory = input_directory
    
    input_files = constants.input_files_segments
    input_files = [input_directory+i for i in input_files]

    df = pd.concat([pd.read_csv(el) for el in input_files])
    df = df.reset_index()

    story_beginning_patterns = ['deportation to Auschwitz','transfer to Auschwitz','first impressions']
    story_ending_patterns = ['transfer from Auschwitz','camp liberation','forced marches','death marches']

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
 
    segments_to_change = df[df.KeywordID.isin(story_beginning_ids)]
    print ("Number of iterations needed: "+str(len(segments_to_change)))
   
    for row_index in tqdm(segments_to_change.index):
        story_beginning=df.iloc()[row_index ].copy()
        story_beginning['KeywordID']=88888
        story_beginning['KeywordLabel']='arrival'
        df.iloc()[row_index]=story_beginning
   

    segments_to_change = df[df.KeywordID.isin(story_ending_ids)]

    print ("Number of iterations needed: "+str(len(segments_to_change)))
    
    for row_index in tqdm(segments_to_change.index):
        story_ending=df.iloc()[row_index ].copy()
        story_ending['KeywordID']=9999
        story_ending['KeywordLabel']='departure'
        df.iloc()[row_index]=story_ending

    df.to_csv(output_directory+constants.input_files_segments_story_end_beginning_distinguished[0])


    # Convenience method to check if the process came through as expected
    
    '''
    for interview in df_segments[0:20].iterrows():
        pdb.set_trace()
        print (df[df.IntCode==interview[1]['IntCode']])

    '''
        