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



def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]
    
    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.
    
    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    result = []
    for element in zip(*[lst[i::n] for i in range(n)]):
        result.append(element)
    return result

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




    df["IntCode"] = df.IntCode.map(lambda x: int(x))

    '''
    kws = df.groupby(['KeywordID','KeywordLabel'])['IntCode'].unique().map(lambda x: len(x)).to_frame(name="TotalNumberIntervieweeUsing").reset_index()
    kws =kws[kws['KeywordLabel'].str.islower()]

    kws_needed = kws[kws.TotalNumberIntervieweeUsing>4][['KeywordID','KeywordLabel']]
    keywords = kws_needed.reset_index()[['KeywordID','KeywordLabel']]
    df = df[df['KeywordID'].isin(kws_needed['KeywordID'])]
    '''
    df_intcode_segment = df.groupby("IntCode")['SegmentNumber'].unique().to_frame(name="SegmentID").reset_index()

    df_intcode_segment.SegmentID.apply(lambda x: sorted(x,reverse=False))
    df_intcode_segment ['SegmentID']= df_intcode_segment.SegmentID.apply(group,n=3)
    
    # Find the ones where no group of three segments could be identified

    int_codes_to_delete = df_intcode_segment[df_intcode_segment.astype(str)['SegmentID'] != '[]']['IntCode'].tolist()
    df_intcode_segment = df_intcode_segment[df_intcode_segment['IntCode'].isin(int_codes_to_delete)]

    # Delete these elements from the original dataframe

    df = df[df['IntCode'].isin(int_codes_to_delete)]

    for i,element in enumerate(df_intcode_segment.iterrows()):
        print (i)
        intcode = element[1]["IntCode"]
        for SegmentIDs in element[1]['SegmentID']:
            new_id = str(intcode)+'_'+'_'.join(SegmentIDs)
            for SegmentID in SegmentIDs:
                df.loc[df[((df.IntCode==intcode) & (df.SegmentNumber == SegmentID))].index,'SegmentID']=new_id

               


    pdb.set_trace()

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

    