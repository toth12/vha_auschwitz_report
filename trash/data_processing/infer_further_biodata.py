"""Infers further biodata about each interviewee

The input is the original segment and biodata; this script infers each
interviewee's arrival and leaving year, length of stay, type of force labour
he / she did. The output is the panda datagframe, which is the copy of the
original biodata with updated metadata fields

"""


import constants
import pandas as pd
import codecs
import csv
import numpy as np
import re
from datetime import timedelta
import pdb
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_datetime64_any_dtype as is_timedelta64_ns_dtype




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
    del number_of_segments["Segments"]
    bio_data = bio_data.merge(number_of_segments)
    return bio_data


def infer_old_new_system(bio_data,segment_data):

    df = segment_data
    
    # Drop duplicated segments

    df_segment_length = df.drop_duplicates('SegmentID')

    # Eliminate those segments where the OutTapeNumber and the InTapenumber are not the same (in these cases we cannot estimate the lenght)

    not_equal=len(df_segment_length[df_segment_length['OutTapenumber']!=df_segment_length['InTapenumber']])

    
    df_segment_length = df_segment_length[df_segment_length['OutTapenumber']==df_segment_length['InTapenumber']]



    # Calculate the segment length (not the ones above)

    df_segment_length['segment_lenght'] = df_segment_length['OutTimeCode'] - df_segment_length['InTimeCode']

    # Get those interview codes that contain segments longer than one minutes


    old_system_in_codes = df_segment_length[df_segment_length.segment_lenght>timedelta(minutes=1)]['IntCode'].unique().tolist()

    segmentation_system = ['old' if x in old_system_in_codes else 'new' for x in bio_data['IntCode'].tolist()]

    bio_data['segmentation_system']=segmentation_system
    return bio_data
    
    # Average segment length in the old system

    #df_segment_length[df_segment_length['IntCode'].isin(old_system_in_codes)]['segment_lenght'].mean()


def infer_total_length_of_segments(bio_data,segment_data):

    df = segment_data

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
    bio_data = bio_data.merge(intcode_kwords)
    # Calculate whether Birkenau occurs at least 2/3 of all interview segments
    bio_data['Birkenau_segment_percentage'] = bio_data['Birkenau_mentioned_times'] / bio_data['number_of_segments']

    return bio_data

def is_transfer_route_helper(keywords):
    if 15803 in keywords:
        return True
    else:
        return False

def is_transfer_route(bio_data,segment_data):
    df = segment_data
    df = df.groupby(['IntCode'])['KeywordID'].apply(list).to_frame(name="KeywordID").reset_index()
    df['is_transfer_route'] = df.KeywordID.apply(is_transfer_route_helper)
    bio_data=bio_data.merge(df)

    return bio_data


def infer_forced_labour(bio_data,segment_data):
    df=segment_data.groupby("IntCode")['KeywordLabel'].apply(list).to_frame(name="Keywords").reset_index()
    df['forced_labor'] = df.Keywords.apply(infer_forced_labour_helper)
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
    output_file = constants.input_files_biodata_with_inferred_fields

    # Read the input files into panda dataframe

    # Read the segments data
    
    df = pd.concat([pd.read_csv(el) for el in input_files])
    
    
    # Read the biodata

    # Get the bio data
    bio_data = constants.input_files_biodata
    df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']


    ### Cast datatypes

    ## Start with segments data

    df_numeric = ['IntCode','SegmentID','InTapenumber','OutTapenumber','KeywordID']
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


    ## Type cast on biodata
    df_biodata_string = ['InterviewTitle','IntervieweeName','Gender','ExperienceGroup','CityOfBirth','CountryOfBirth','InterviewCity','InterviewCountry','InterviewLanguage','HistoricEvent','OrganizationName']
    df_biodata_numeric = ['IntCode']
    df_biodata_time = ['DateOfBirth','InterviewDate']

    # Cast to numeric
    for col in df_biodata_numeric:
        df_biodata[col] = pd.to_numeric(df_biodata[col])

    # Cast to string
    for col in df_biodata_string:
        df_biodata[col] = df_biodata[col].astype('string')

    # Cast to temporal

    for col in df_biodata_time:
        df_biodata[col] = pd.to_datetime(df_biodata[col],errors='coerce')

    

    # Get the IntCode of Jewish survivors
    IntCode = df_biodata[df_biodata['ExperienceGroup']=='Jewish Survivor']['IntCode'].to_list()
    

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

    # Make sure that the inferred fields are in the correct datatypes
    

    df_biodata_list = ['forced_labor','forced_labour_type','KeywordID']
    df_biodata_bool = ['is_transfer_route']
    df_biodata_string.extend(['segmentation_system'])
    df_biodata_numeric.extend(['earliest_year', 'latest_year','length_of_stay', 'number_of_segments','Birkenau_mentioned_times','Birkenau_segment_percentage','easy', 'hard','medium','length_in_minutes'])
    df_biodata_delta_time = ['length']

    # Check lists
    for col in df_biodata_list:
        assert isinstance(df_biodata[col].iloc()[2], list)

    # Check boolean
    for col in df_biodata_bool:
        assert df_biodata[col].dtypes.name == 'bool'

    # Check string
    for col in df_biodata_string:
        assert is_string_dtype(df_biodata[col])

    # Check numeric
    for col in df_biodata_numeric:
        assert is_numeric_dtype(df_biodata[col])

    # Check temporal length
    for col in df_biodata_delta_time:
        df_biodata[col].dtype == 'timedelta64[ns]'

    df_biodata.to_csv(input_directory+output_file)