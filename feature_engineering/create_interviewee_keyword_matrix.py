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

if __name__ == '__main__':

    # Read the data


    input_directory = constants.input_data
    output_directory = constants.output_data_features
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
        del csv_data_temp[0:10]
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


    df["IntCode"] = df.IntCode.map(lambda x: int(x))

    kws = df.groupby(['KeywordID','KeywordLabel'])['IntCode'].unique().map(lambda x: len(x)).to_frame(name="TotalNumberIntervieweeUsing").reset_index()
    kws =kws[kws['KeywordLabel'].str.islower()]

    kws_needed = kws[kws.TotalNumberIntervieweeUsing>4][['KeywordID','KeywordLabel']]
    keywords = kws_needed.reset_index()[['KeywordID','KeywordLabel']]
    df = df[df['KeywordID'].isin(kws_needed['KeywordID'])]

    interviewee_keyword = df.groupby(['IntCode'])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()
    interviewee_keyword_matrix =np.zeros(shape=(len(interviewee_keyword),len(keywords)))
    for i,element in enumerate(interviewee_keyword.iterrows()):
        for keyword in element[1]['KeywordID']:
            keyword_index = keywords[keywords.KeywordID==keyword].index[0]
            interviewee_keyword_matrix[i, keyword_index] = 1


    np.savetxt(output_directory+'interviewee_keyword_matrix.txt', interviewee_keyword_matrix, fmt='%d')
    interviewee_keyword.to_csv(output_directory+'interview_index.csv')
    keywords.to_csv(output_directory+'keyword_index.csv')

    segment_keyword = df.groupby(['SegmentID'])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()
    segment_keyword_matrix =np.zeros(shape=(len(segment_keyword),len(keywords)))
    for i,element in enumerate(segment_keyword.iterrows()):
        for keyword in element[1]['KeywordID']:
            keyword_index = keywords[keywords.KeywordID==keyword].index[0]
            segment_keyword_matrix[i, keyword_index] = 1
    np.savetxt(output_directory+'segment_keyword_matrix.txt', interviewee_keyword_matrix, fmt='%d')
    segment_keyword.to_csv(output_directory+'segment_index.csv')


pdb.set_trace()

    