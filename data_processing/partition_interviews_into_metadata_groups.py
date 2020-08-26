'''Takes row index of the segment - keyword matrix and returns row indices corresponding to metadata fields'''

#!/usr/bin/env python
# coding: utf-8

import pdb
import pandas as pd
import constants
import json

def prepare_input_data(metadata_fields):

    result = {}
    # Find the relevant interview ids
    interview_codes = []
    metadata_field_names = []

    for metadata_field in metadata_fields:
        if "_w" in metadata_field:
            interview_codes_to_filter = IntCodeW
        elif "_m" in metadata_field:
            interview_codes_to_filter = IntCodeM
        else:
            interview_codes_to_filter = []

        if "CountryOfBirth" in metadata_field:
            country_of_origins = df_biodata.groupby('CountryOfBirth')['CountryOfBirth'].count().to_frame(
                'Count').reset_index()
            country_of_origins = country_of_origins[country_of_origins.Count > 50]
            country_of_origins = country_of_origins.CountryOfBirth.tolist()
            for el in country_of_origins:
                interview_codes_temp = df_biodata[df_biodata['CountryOfBirth'] == el].IntCode.to_list()
                if len(interview_codes_to_filter) > 1:
                    interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
                    if "_w" in metadata_field:
                        el = el + '_w'
                    else:
                        el = el + '_m'
                    interview_indices = segment_df[segment_df['IntCode'].isin(interview_codes_temp)].index.to_list()
                    result[el] = interview_indices
                else:
                    interview_indices = segment_df[segment_df['IntCode'].isin(interview_codes_temp)].index.to_list()
                    result[el]=interview_indices
        else:
            if "complete" in metadata_field:
                interview_codes_temp = df_biodata.IntCode.to_list()
                if len(interview_codes_to_filter) > 1:
                    interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
        
            elif (('easy' in metadata_field) or ('medium' in metadata_field) or ('hard' in metadata_field)):
                interview_codes_temp = df_biodata[df_biodata[metadata_field.split('_')[0]] == 1].IntCode.tolist()

                if len(interview_codes_to_filter) > 1:
                    interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]               

            elif "notwork" in metadata_field:
                interview_codes_temp = df_biodata[
                    (df_biodata['easy'] == 0) & (df_biodata['hard'] == 0) & (df_biodata['medium'] == 0)].IntCode.tolist()
                if len(interview_codes_to_filter) > 1:
                    interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
            elif "work" in metadata_field:
                interview_codes_temp = df_biodata[
                    (df_biodata['easy'] == 1) | (df_biodata['hard'] == 1) | (df_biodata['medium'] == 1)].IntCode.tolist()
                if len(interview_codes_to_filter) > 1:
                    interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]    

            interview_indices = segment_df[segment_df['IntCode'].isin(interview_codes_temp)].index.to_list()
            result[metadata_field]=interview_indices
    return result

    
    

if __name__ == '__main__':
    
    # Read the biodata
    input_directory = constants.input_data
    bio_data = constants.input_files_biodata_birkenau
    df_biodata = pd.read_csv(input_directory+bio_data)
    df_biodata = df_biodata.fillna(0)
    IntCodeM = df_biodata[df_biodata.Gender=='M']['IntCode'].to_list()
    IntCodeW = df_biodata[df_biodata.Gender=='F']['IntCode'].to_list()
    metadata_fields = ['complete','complete_m','complete_w','CountryOfBirth','CountryOfBirth_m','CountryOfBirth_w','easy_w','easy_m','medium_m','medium_w','hard_m','hard_w',"notwork","notwork_m","notwork_w","work","work_m","work_w"]
    

    # Read the data index
    input_directory = constants.output_data_segment_keyword_matrix
    # Read the row index (groups of three segments) of the matrix above
    segment_df = pd.read_csv(input_directory + 
                         constants.output_segment_keyword_matrix_document_index)

    int_codes = segment_df['IntCode'].to_list()


    input_indices = prepare_input_data(metadata_fields)

    with open(input_directory+'metadata_partitions.json', 'w') as outfile:
        json.dump(input_indices, outfile)



