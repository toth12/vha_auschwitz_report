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


def find_subcamps(keywords):
    result = []
    for element in keywords:
        # Make sure that time refers to Poland
        if "II-Birkenau" in element:
            result.append(element)
    if len(result)>2:
        return True
    else:
        return False

if __name__ == '__main__':

    # Read the data


    input_directory = constants.input_data
    output_directory = constants.output_chi2_test
    input_files = constants.input_files_segments

    input_files = [input_directory+i for i in input_files]

    # Read the input files into panda dataframe

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

    # Filter out non Jewish survivors

    # Get the bio data
    bio_data = constants.input_files_biodata
    df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']

    
    # Get the IntCode of Jewish survivors
    IntCode = df_biodata[df_biodata['ExperienceGroup']=='Jewish Survivor']['IntCode'].to_list()
    IntCode = [str(el) for el in IntCode]

    # Leave only Jewish survivors
    df = df[df['IntCode'].isin(IntCode)]

    df_biodata = df_biodata[df_biodata['IntCode'].isin(IntCode)]

    # Create interview code keyword dataframe
    intcode_kwords = df.groupby('IntCode')['KeywordLabel'].apply(list).to_frame(name="Keywords").reset_index()

    # Find all years in interview code keyword dataframe
    intcode_kwords['earliest_year'] = intcode_kwords.Keywords.apply(find_year,type='earliest')

    intcode_kwords['latest_year'] = intcode_kwords.Keywords.apply(find_year,type='latest')
    intcode_kwords['was_in_Birkenau'] = intcode_kwords.Keywords.apply(find_subcamps)

    intcode_kwords['length_of_stay'] = intcode_kwords['latest_year'] - intcode_kwords['earliest_year']

    birkenau_survivors = intcode_kwords[(intcode_kwords['earliest_year']>1942) & (intcode_kwords['was_in_Birkenau']==True)]['IntCode'].to_list()

    birkenau_survivors = intcode_kwords[intcode_kwords['was_in_Birkenau']==True]['IntCode'].to_list()

    birkenau_survivors = intcode_kwords[intcode_kwords['earliest_year']>1942]['IntCode'].to_list()

    # Filter biodata to have only Birkenau Survivors

    df_biodata = df_biodata[df_biodata['IntCode'].isin(birkenau_survivors)]
    len(df_biodata[df_biodata['Gender']=="F"])
    len(df_biodata[df_biodata['Gender']=="M"])

    # Find the ones that contain at least five segments

    df = df[df['IntCode'].isin(birkenau_survivors)]

    number_of_segments = df.groupby('IntCode')['SegmentNumber'].unique().to_frame(name="Segments").reset_index()
    number_of_segments['number_of_segments']=number_of_segments['Segments'].apply(lambda x: len(x))
    number_of_segments[number_of_segments['number_of_segments']>5]

    # intcode_kwords[intcode_kwords['length_of_stay']==1]

    '''
    subcamps = []
    [subcamps.extend(element) for element in intcode_kwords.subcamps.to_list()]
    subcamps =list(set(subcamps)
    '''

    pdb.set_trace()