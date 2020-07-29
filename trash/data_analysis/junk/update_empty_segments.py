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
    for el in input_files:

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

    # Update segments that are "empty" i.e. contain only generic terms

    segments = df.groupby(["IntCode","SegmentID","SegmentNumber"])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()
    segments['empty'] = segments.apply(check_if_empty,axis=1)
    print (len(segments[segments['empty']==True]))
    number_of_empty_segments = 0
    total = len(segments[segments['empty']==True])
    n = 0
    pdb.set_trace()
    for element in segments[segments['empty']==True].iterrows():
        IntCode = element[1]["IntCode"]
        PreviousSegmentNumber = int(element[1]["SegmentNumber"]) - 1
        print ('\n')
        print (n)
        print (total)
        print ('\n')
        n = n+1

        

        #Find the previous segment id
        previous_segment = df[(df["IntCode"]==IntCode) & (df['SegmentNumber'] ==str(PreviousSegmentNumber))]


        

            #iterate through all keywords in the previous segment and if a keyword is not a generic term update
        for row in previous_segment.iterrows():
            
            new_row = row[1].copy()
            new_row['SegmentNumber'] = element[1]["SegmentNumber"]
            new_row['SegmentID'] = element[1]["SegmentID"]
            df = df.append(new_row)
    
        
    #df.to_csv("segments_updated.csv")

