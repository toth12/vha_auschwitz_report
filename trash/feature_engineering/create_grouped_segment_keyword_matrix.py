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
    df = pd.concat([pd.read_csv(el) for el in input_files])


    # Get the bio data of each interviewee
    df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']
    
    # Get the IntCode of Jewish survivors
    IntCode = df_biodata[df_biodata['ExperienceGroup']=='Jewish Survivor']['IntCode'].to_list()
    
    # Change the IntCode from int to str
    IntCode = [str(el) for el in IntCode]

    # Eliminate non Jewish Survivors from both the biodata and the segment data
    df = df[df['IntCode'].isin(IntCode)]
    df_biodata = df_biodata[df_biodata['IntCode'].isin(IntCode)]



    # Change the IntCode from str to int in the segment data
    df["IntCode"] = df.IntCode.map(lambda x: int(x))


    # Create a new segment id that is the combination of interview code and segment number
    df['new_segment_id']=df.IntCode.astype(str)+'_'+df.SegmentNumber.astype(str)

   

    # Eliminate those index terms that occur in less than 25 interviews
    kws = df.groupby(['KeywordID', 'KeywordLabel'])['IntCode'].unique().map(lambda x: len(x)).to_frame(name="TotalNumberIntervieweeUsing").reset_index()
    
    kws_needed = kws[kws.TotalNumberIntervieweeUsing>25][['KeywordID','KeywordLabel']]
    
    # Eliminate very generic index terms
    kws_needed = kws_needed[~kws_needed['KeywordID'].isin(generic_terms)]


    keywords = kws_needed.reset_index()[['KeywordID','KeywordLabel']]
    df = df[df['KeywordID'].isin(kws_needed['KeywordID'])]
    
     # Save the keywords that is used, this will be the feature index
    keywords.to_csv(output_directory+'feature_index.csv')


    # Create the segment_keyword table  through Groupby the updated ids

    """
                   updated_id                                          KeywordID
    0      10000_62_63_64                                            [10853]
    1      10000_65_66_67                                            [15098]
    2      10004_57_58_59                                            [10698]
    3      10006_47_48_49                              [10983, 12044, 14280]
    """
    segment_keyword = df.groupby(['new_segment_id'])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()

    
    # Create an empty np array that will hold this
    segment_keyword_matrix =np.zeros(shape=(len(segment_keyword),len(keywords)))
    
    # Iterate through the segment_keyword table
    for i,element in enumerate(segment_keyword.iterrows()):
        for keyword in element[1]['KeywordID']:
            keyword_index = keywords[keywords.KeywordID==keyword].index[0]
            segment_keyword_matrix[i, keyword_index] = 1

    # Save the segment keyword matrix
    np.savetxt(output_directory+'segment_keyword_matrix.txt', segment_keyword_matrix, fmt='%d')
    segment_keyword.to_csv(output_directory+'document_index.csv')



    