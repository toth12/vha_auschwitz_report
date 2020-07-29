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

pd.set_option('display.max_rows', 50)


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
    generic_terms = ['14231', '14223', '13930', '13929', '13214', '14230', '14229', '13926', '14049', '14605', '14307', '16285', '12498', '13215', '13931', '14241', '14235', '13310', '7601', '14233', '7528', '17206', '14232', '13018', '16192', '7624', '14226', '14225']



    # Read the segment input files into panda dataframe

    csv_data = []
    for el in input_files[0:1]:

        f = codecs.open(el,"rb","utf-8")
        csvread = csv.reader(f,delimiter=',')
        csv_data_temp = list(csvread)
        columns = csv_data_temp[0]
        #Drop the first line as that is the column
        del csv_data_temp[0:1]
        csv_data.extend(csv_data_temp)

    columns[0] = "IntCode"
    f = None
    csvread = None
    csv_data_temp = None
    df = pd.DataFrame(csv_data,columns=columns)
    csv_data = None


    for element in generic_terms:

        print (df[df["KeywordID"]==element].reset_index().iloc()[0]["KeywordLabel"])
    pdb.set_trace()
    # Update segments that are "empty" i.e. contain only generic terms

    segments = df.groupby(["IntCode","SegmentID","SegmentNumber"])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()
    segments['empty'] = segments.apply(check_if_empty,axis=1)
    print (len(segments[segments['empty']==True]))
    number_of_empty_segments = 0
    total = len(segments[segments['empty']==True])
    n = 0

    int_codes=segments.IntCode.unique()
    updated_segments = pd.DataFrame(columns=segments.columns)
    for f,int_code in enumerate(int_codes):
        print (f)
        print (len(int_codes))
        segments_interviewee=segments[segments["IntCode"].isin([int_code])]
        for element in segments_interviewee.iterrows():

            if element[1]['empty']:

                PreviousSegmentNumber = int(element[1]["SegmentNumber"]) - 1

            

        

                #Find the previous segment id
                previous_segment = segments_interviewee[segments_interviewee['SegmentNumber'] ==str(PreviousSegmentNumber)]

                if len(previous_segment)>0:

                    
                    segments_interviewee.at[element[0],'KeywordID']=previous_segment.reset_index()['KeywordID'][0]
                    



        
        updated_segments=updated_segments.append(segments_interviewee)
    pdb.set_trace()
        