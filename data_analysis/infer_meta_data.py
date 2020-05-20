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
from matplotlib.lines import Line2D
import mpld3
from plotly import express as px
import plotly
import plotly.figure_factory as ff
import sys
import re
from datetime import timedelta


def find_year(keywords,type):
    result = []
    for element in keywords:
        # Make sure that time refers to Poland
        if "Poland" in element:
            temp = re.findall(r'\d{4}', element) 
            res = list(map(int, temp)) 
            if len(res)>0:
                for year in res:
                    result.append(year)
    result.sort()
    
    if len(result) ==0:
        return np.nan
    else:
        if type == 'earliest':

            return int(result[0])
        else:
            return int(result[-1])



def infer_latest_earliest_year(bio_data,segment_data):

    # Create interview code keyword dataframe
    intcode_kwords = segment_data.groupby('IntCode')['KeywordLabel'].apply(list).to_frame(name="Keywords").reset_index()

    # Find all years in interview code keyword dataframe
    intcode_kwords['earliest_year'] = intcode_kwords.Keywords.apply(find_year,type='earliest')

    intcode_kwords['latest_year'] = intcode_kwords.Keywords.apply(find_year,type='latest')

    # Delete keywords

    del intcode_kwords['Keywords']

    intcode_kwords['IntCode'] = pd.to_numeric(intcode_kwords['IntCode'])
    bio_data = bio_data.merge(intcode_kwords)

    return bio_data

def infer_length_of_stay(bio_data):
    bio_data['length_of_stay'] = bio_data['latest_year'] - bio_data['earliest_year']
    return bio_data

def infer_number_of_segments(bio_data,segment_data):
    number_of_segments = segment_data.groupby('IntCode')['SegmentNumber'].unique().to_frame(name="Segments").reset_index()
    number_of_segments['number_of_segments']=number_of_segments['Segments'].apply(lambda x: len(x))
    number_of_segments['IntCode'] = pd.to_numeric(number_of_segments['IntCode'])
    del number_of_segments["Segments"]
    bio_data = bio_data.merge(number_of_segments)
    return bio_data


def infer_old_new_system(bio_data,segment_data):

    df = segment_data
    # Transform InTimeCode OutTimeCode into temporal data

    df['InTimeCode'] = pd.to_datetime(df['InTimeCode'], format = "%H:%M:%S:%f")



    df['OutTimeCode'] = pd.to_datetime(df['OutTimeCode'], format = "%H:%M:%S:%f")

    # Drop duplicated segments

    df_segment_length = df.drop_duplicates('SegmentID')

    # Eliminate those segments where the OutTapeNumber and the InTapenumber are not the same (in these cases we cannot estimate the lenght)

    not_equal=len(df_segment_length[df_segment_length['OutTapenumber']!=df_segment_length['InTapenumber']])

    
    df_segment_length = df_segment_length[df_segment_length['OutTapenumber']==df_segment_length['InTapenumber']]



    # Calculate the segment length (not the ones above)

    df_segment_length['segment_lenght'] = df_segment_length['OutTimeCode'] - df_segment_length['InTimeCode']

    # Get those interview codes that contain segments longer than one minutes




    
    old_system_in_codes = df_segment_length[df_segment_length.segment_lenght>timedelta(minutes=1)]['IntCode'].unique().tolist()
    
    segmentation_system = ['old' if str(x)  in old_system_in_codes else 'new' for x in bio_data['IntCode'].tolist()]
    bio_data['segmentation_system']=segmentation_system
    return bio_data

def infer_total_length_of_segments(bio_data,segment_data):

    df = segment_data
    # Transform InTimeCode OutTimeCode into temporal data

    df['InTimeCode'] = pd.to_datetime(df['InTimeCode'], format = "%H:%M:%S:%f")



    df['OutTimeCode'] = pd.to_datetime(df['OutTimeCode'], format = "%H:%M:%S:%f")

    # Drop duplicated segments

    df_segment_length = df.drop_duplicates('SegmentID')

    # Eliminate those segments where the OutTapeNumber and the InTapenumber are not the same (in these cases we cannot estimate the lenght)

    not_equal=len(df_segment_length[df_segment_length['OutTapenumber']!=df_segment_length['InTapenumber']])

    
    df_segment_length = df_segment_length[df_segment_length['OutTapenumber']==df_segment_length['InTapenumber']]



    # Calculate the segment length (not the ones above)

    df_segment_length['segment_lenght'] = df_segment_length['OutTimeCode'] - df_segment_length['InTimeCode']

    # Calculate how much time an interview speaks about Auschwitzy
    df_interview_segment_length = df_segment_length.groupby(['IntCode'])['segment_lenght'].agg('sum')

    df_interview_segment_length = pd.DataFrame({'IntCode':df_interview_segment_length.index, 'length':df_interview_segment_length.values})

    df_interview_segment_length = df_interview_segment_length.sort_values('length')

    # Calculate in minutes

    df_interview_segment_length['length_in_minutes']= np.round(df_interview_segment_length['length'] / np.timedelta64(1, 'm'))

    df_interview_segment_length['IntCode'] = pd.to_numeric(df_interview_segment_length['IntCode'])
    bio_data = bio_data.merge(df_interview_segment_length)
    return bio_data




if __name__ == '__main__':

    # Read the original datafiles

    input_directory = constants.input_data
    input_files = constants.input_files_segments
    input_files = [input_directory+i for i in input_files]

    # Read the input files into panda dataframe

    # Read the segments data
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
    df = pd.DataFrame(csv_data,columns=columns)

    # Read the biodata

    # Get the bio data
    bio_data = constants.input_files_biodata
    df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']

    # Get the IntCode of Jewish survivors
    IntCode = df_biodata[df_biodata['ExperienceGroup']=='Jewish Survivor']['IntCode'].to_list()
    IntCode = [str(el) for el in IntCode]

    # Leave only Jewish survivors
    df = df[df['IntCode'].isin(IntCode)]

    df_biodata = df_biodata[df_biodata['IntCode'].isin(IntCode)]

    # Eliminate those two interviews that are connected to two persons
    df_biodata = df_biodata[~df_biodata['IntCode'].duplicated(keep=False)]
    IntCode = df_biodata.IntCode.tolist()
    IntCode = [str(el) for el in IntCode]
    df = df[df['IntCode'].isin(IntCode)]

    df_biodata = infer_latest_earliest_year(df_biodata,df)

    df_biodata = infer_length_of_stay(df_biodata)
    df_biodata = infer_number_of_segments(df_biodata,df)
    df_biodata = infer_old_new_system(df_biodata,df)

    df_biodata = infer_total_length_of_segments(df_biodata,df)

    #new_set = df_biodata[(((df_biodata['segmentation_system']=='new') & (df_biodata['number_of_segments']>5)) | ((df_biodata['segmentation_system']=='old') & (df_biodata['length']>timedelta(minutes=5))))& (df_biodata['earliest_year']>1942)]


    pdb.set_trace()

