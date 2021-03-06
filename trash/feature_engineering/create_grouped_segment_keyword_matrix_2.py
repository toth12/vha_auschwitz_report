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


def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]
    
    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.
    
    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    result = []
    if lst['segmentation_system'] == "old":
        n = 1
    # Check if old or new system

    lst = lst['SegmentID'].tolist()
    for element in zip(*[lst[i::n] for i in range(n)]):
        result.append(element)
    return result



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
    #df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']
    input_directory_biodata = constants.input_data_filtered
    bio_data = constants.input_files_biodata_birkenau
    df_biodata = pd.read_csv(input_directory_biodata+bio_data)

    # Get the IntCode of Jewish survivors
    IntCode = df_biodata[df_biodata['ExperienceGroup']=='Jewish Survivor']['IntCode'].to_list()
    # Change the IntCode from int to str
    IntCode = [str(el) for el in IntCode]

    # Eliminate non Jewish Survivors from both the biodata and the segment data
    df = df[df['IntCode'].isin(IntCode)]
    df_biodata = df_biodata[df_biodata['IntCode'].isin(IntCode)]




    # Update segments that are "empty" i.e. contain only generic terms
    
    segments = df.groupby(["IntCode","SegmentID","SegmentNumber"])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()
    segments['empty'] = segments.apply(check_if_empty,axis=1)
    print (len(segments[segments['empty']==True]))
    for element in segments[segments['empty']==True].iterrows():
        IntCode = element[1]["IntCode"]
        PreviousSegmentNumber = int(element[1]["SegmentNumber"]) - 1
        

        #Find the previous segment id
        previous_segment = df[(df["IntCode"]==IntCode) & (df['SegmentNumber'] ==str(PreviousSegmentNumber))]
        if len(previous_segment)>0:

            #iterate through all keywords in the previous segment and if a keyword is not a generic term update
            for row in previous_segment.iterrows():
                if row[1]["KeywordID"] not in generic_terms:
                    new_row = row[1].copy()
                    new_row['SegmentNumber'] = element[1]["SegmentNumber"]
                    new_row['SegmentID'] = element[1]["SegmentID"]
                    df = df.append(new_row)
                    
        
    pdb.set_trace()

    # Change the IntCode from str to int in the segment data
    df["IntCode"] = df.IntCode.map(lambda x: int(x))

    # Create a dataframe in which each record is an IntCode and the list of SegmentIDS belonging to that IntCode"
    """
           IntCode                                          SegmentID
    0            2                          [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 2...

    """
    
    # Find out what segmentation system was used
    df_segmentation_system = df_biodata[['IntCode','segmentation_system']]
    df_intcode_segment = df.groupby("IntCode")['SegmentNumber'].unique().to_frame(name="SegmentID").reset_index()

    # Merge with the segmentation system

    df_intcode_segment = df_intcode_segment.merge(df_segmentation_system)
    
    # Make sure that segment ids are sorted
    df_intcode_segment.SegmentID.apply(lambda x: sorted(x,reverse=False))
    pdb.set_trace()

    # Create groups of three segments from the list segments ids
    """
           IntCode                                          SegmentID
   0            2  [(16, 17, 18), (19, 20, 21), (22, 23, 24), (25...

    """
    df_intcode_segment ['SegmentID']= df_intcode_segment.apply(group,n=1,axis=1)



    
    # Find those IntCode where at least one group of three segments could be identified

    #int_codes_to_keep = df_intcode_segment[df_intcode_segment.astype(str)['SegmentID'] != '[]']['IntCode'].tolist()

    # Keep only these intcode
    #df_intcode_segment = df_intcode_segment[df_intcode_segment['IntCode'].isin(int_codes_to_keep)]

    # Keep only these intcodes in the original segment dataframe

   # df = df[df['IntCode'].isin(int_codes_to_keep)]

    """
    When merging every three segments, the resulting new segment has an id that is the combination of the original 
    three segment numbers and the interview code. The following lines prepare this
    """

    df['new_segment_id']=df.IntCode.astype(str)+'_'+df.SegmentNumber.astype(str)

    new_segment_ids = []
    for i,element in enumerate(df_intcode_segment.iterrows()):
        intcode = element[1]["IntCode"]
        for SegmentIDs in element[1]['SegmentID']:
            new_id = str(intcode)+'_'+'_'.join(SegmentIDs)
            for SegmentID in SegmentIDs:
                new_segment_ids.append([str(intcode)+'_'+SegmentID,new_id])


   

    df_new_segment_ids = pd.DataFrame(new_segment_ids,columns=["new_segment_id","updated_id"])

    """
    df_new_segment_ids is a dataframe that holds the combination of int_code and segment id, plus the new combined segment id of three segments

           new_segment_id        updated_id
0                2_16        2_16_17_18
    
    Since the new_segment_id is also in the segments dataframe (created above), the df_new_segment_ids and segments dataframe can be merged
    """

    df = df.merge(df_new_segment_ids)

    # The resulting table contains the membership of every segment to a group of consecutive three segments with the new combined id (updated_id)

    """
        
                IntCode     IntervieweeName SegmentID SegmentNumber  ... DateKeywordCreated IsGroup new_segment_id        updated_id
    0             2  Rosalie Greenfield    514929            16  ...        08 Dec 1995       0           2_16        2_16_17_18
    1             2  Rosalie Greenfield    514929            16  ...        29 Oct 1996       0           2_16        2_16_17_18
    2             2  Rosalie Greenfield    514929            16  ...        09 Dec 1996       0           2_16        2_16_17_18
    3             2  Rosalie Greenfield    514940            17  ...        08 Dec 1995       0           2_17        2_16_17_18
    4             2  Rosalie Greenfield    514940            17  ...        12 Apr 1996       0           2_17        2_16_17_18

    """

    # Eliminate those index terms that occur in less than 25 interviews

    kws = df.groupby(['KeywordID', 'KeywordLabel'])['IntCode'].unique().map(lambda x: len(x)).to_frame(name="TotalNumberIntervieweeUsing").reset_index()
    
    kws_needed = kws[kws.TotalNumberIntervieweeUsing>50][['KeywordID','KeywordLabel']]
    # eliminate very generic keywords

    keywords_ids_to_eliminate= ['16103','12754','14049','14605','8057','7624','7531','10592','13214','12497','13215','13310','7601','14233','16192','14226','14232','13929','13930','7528','13018','13926','13931','12498','16285']




    

    kws_needed = kws_needed[~kws_needed['KeywordID'].isin(keywords_ids_to_eliminate)]

     # Eliminate all prisoners keywords

    #kws_needed = kws_needed[~kws_needed['KeywordLabel'].str.contains('prisoners')]


    # Eliminate prisoner functionaries
    kws_needed = kws_needed[~kws_needed['KeywordLabel'].str.contains('prisoner functionaries')]


    keywords = kws_needed.reset_index()[['KeywordID','KeywordLabel']]
    df = df[df['KeywordID'].isin(kws_needed['KeywordID'])]


    



    # Save the keywords that is used

    keywords.to_csv(output_directory+'keyword_index_merged_segments_birkenau.csv')


    # Create the segment_keyword table 

    segment_keyword = df.groupby(['updated_id'])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()
    pdb.set_trace()

    # Groupby the updated ids; this contains the new id of every three segments and the index ids in them

    """
                   updated_id                                          KeywordID
    0      10000_62_63_64                                            [10853]
    1      10000_65_66_67                                            [15098]
    2      10004_57_58_59                                            [10698]
    3      10006_47_48_49                              [10983, 12044, 14280]

    

    """

    #Eliminate those entries that have less than two keyword ids
    #segment_keyword = segment_keyword[segment_keyword['KeywordID'].str.len()>2]


    # Create the segment keyword matrix, the position of every keyword is defined by its position in the keywords table above
    
    # Create an empty np array that will hold this
    segment_keyword_matrix =np.zeros(shape=(len(segment_keyword),len(keywords)))

    # Iterate through the segment_keyword table
    for i,element in enumerate(segment_keyword.iterrows()):
        for keyword in element[1]['KeywordID']:
            keyword_index = keywords[keywords.KeywordID==keyword].index[0]
            segment_keyword_matrix[i, keyword_index] = 1   
    # Save the segment keyword matrix

    np.savetxt(output_directory+'segment_keyword_matrix_merged_birkenau.txt', segment_keyword_matrix, fmt='%d')
    segment_keyword.to_csv(output_directory+'segment_index_merged_birkenau.csv')

    #27      12288                              camp social relations
    #segment_keyword_matrix[:,27].sum()


    