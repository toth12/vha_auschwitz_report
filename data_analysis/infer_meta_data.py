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

    intcode_kwords = intcode_kwords.astype({"IntCode":int})
    
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
    
    # Average segment length in the old system

    #df_segment_length[df_segment_length['IntCode'].isin(old_system_in_codes)]['segment_lenght'].mean()


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



def was_in_Birkenau_helper(keywords):
    number_of_times = 0
    for element in keywords:
        if any((('Birkenau' in string) or ('Auschwitz (Poland : Concentration Camp)(generic)' in string)) for string in element):
            number_of_times = number_of_times+1
        
    return number_of_times

def was_in_Birkenau(bio_data,segment_data):
    df = segment_data
    ss = df.groupby(['IntCode','SegmentID'])['KeywordLabel'].apply(list).to_frame(name="Keywords").reset_index()
    intcode_kwords = ss.groupby('IntCode')['Keywords'].apply(list).to_frame(name="Keywords").reset_index()

    intcode_kwords['Birkenau_mentioned_times'] = intcode_kwords.Keywords.apply(was_in_Birkenau_helper)
    del intcode_kwords["Keywords"]
    intcode_kwords['IntCode'] = pd.to_numeric(intcode_kwords['IntCode'])
    bio_data = bio_data.merge(intcode_kwords)
    # Calculate whether Birkenau occurs at least 2/3 of all interview segments
    bio_data['Birkenau_segment_percentage'] = bio_data['Birkenau_mentioned_times'] / bio_data['number_of_segments']

    return bio_data

def is_transfer_route_helper(keywords):
    if '15803' in keywords:
        return True
    else:
        return False

def is_transfer_route(bio_data,segment_data):
    df = segment_data
    df = df.groupby(['IntCode'])['KeywordID'].apply(list).to_frame(name="KeywordID").reset_index()
    df['is_transfer_route'] = df.KeywordID.apply(is_transfer_route_helper)
    df = df.astype({"IntCode":int})

    bio_data=bio_data.merge(df)

    return bio_data


def infer_forced_labour(bio_data,segment_data):
    df=segment_data.groupby("IntCode")['KeywordLabel'].apply(list).to_frame(name="Keywords").reset_index()
    df['forced_labor'] = df.Keywords.apply(infer_forced_labour_helper)
    df = df.astype({"IntCode":int})
    del df["Keywords"]
    #Check the number of people who did forced labour
    #df[df.forced_labour_type.str.len().eq(0)]
    
    bio_data = bio_data.merge(df)
    bio_data['forced_labour_type'] = bio_data.forced_labor.apply(infer_forced_labour_type_helper)
    bio_data = bio_data.join(pd.get_dummies(bio_data['forced_labour_type'].apply(pd.Series).stack()).sum(level=0))
    
    #To find the women who did forced labour
    #bio_data[(bio_data.forced_labor.str.len().eq(1))&(bio_data.Gender=="F")]
    return bio_data

def infer_forced_labour_helper(KeywordLabels):
    result = []

    for element in KeywordLabels:
        #if len(forced_labour_typology[forced_labour_typology.forced_labor==element]) >0:
        if element in forced_labour_typology.forced_labor.to_list():
            result.append(element)
    return result
def infer_forced_labour_type_helper(KeywordLabels):
    result = []

    if len(KeywordLabels)>0:
        for element in KeywordLabels:
            if element in forced_labour_typology[forced_labour_typology['type']=='easy'].forced_labor.to_list():
                result.append('easy')
            elif element in forced_labour_typology[forced_labour_typology['type']=='medium'].forced_labor.to_list():
                result.append('medium')
            elif element in forced_labour_typology[forced_labour_typology['type']=='hard'].forced_labor.to_list():
                result.append('hard')
            
        result = list(set(result))
        return result
    else:
        return []

#bio_data[bio_data.forced_labour_type.str.len().eq(0)]
#bio_data[bio_data.forced_labor.str.len().eq(0)]

if __name__ == '__main__':

    # Read the original datafiles

    input_directory = constants.input_data
    input_files = constants.input_files_segments
    input_files = [input_directory+i for i in input_files]
    output_folder = constants.input_data

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
    
    forced_labour_typology=pd.read_csv(input_directory+"forced_labour_typology.csv")

    # Identify the type of forced labour the person did

    df_biodata = infer_forced_labour(df_biodata,df)


    # Find transfer routes

    df_biodata = is_transfer_route(df_biodata,df)





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
    df_biodata = was_in_Birkenau(df_biodata,df)

    df_biodata.to_csv(input_directory+"biodata_with_inferred_fields.csv")

    #df_biodata[df_biodata.Birkenau_mentioned_times >0]

    #df_biodata[((df_biodata.Birkenau_segment_percentage>0.7)&(df_biodata.earliest_year>1943)&(df_biodata.is_transfer_route==False)&(df_biodata.length_in_minutes>10)&(df_biodata.Gender=='M')&(df_biodata.forced_labour_type.str.len().eq(0)))]

    pdb.set_trace()

    #new_set = df_biodata[(((df_biodata['segmentation_system']=='new') & (df_biodata['number_of_segments']>5)) | ((df_biodata['segmentation_system']=='old') & (df_biodata['length']>timedelta(minutes=5))))& (df_biodata['earliest_year']>1942)]

    
    #new_set = df_biodata[(((df_biodata['segmentation_system']=='new') & (df_biodata['number_of_segments']>5)) | ((df_biodata['segmentation_system']=='old') & (df_biodata['length']>timedelta(minutes=5))))& (df_biodata['earliest_year']>1942) & (df_biodata['Birkenau_segment_percentage']>0.66)]
    #birkenau = df_biodata[(((df_biodata['segmentation_system']=='new') & (df_biodata['number_of_segments']>5)) | ((df_biodata['segmentation_system']=='old') & (df_biodata['length']>timedelta(minutes=5))))& (df_biodata['earliest_year']>1942) & (df_biodata['Birkenau_segment_percentage']>0.66)]
    #birkenau.to_csv(output_folder+'/biodata_birkenau.csv')
    #df_biodata['Birkenau_segment_percentage']>0.66

   

