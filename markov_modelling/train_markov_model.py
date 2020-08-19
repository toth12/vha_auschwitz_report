import pandas as pd
import numpy as np
import msmtools
import constants
import os
from msmtools.estimation import is_connected
from markov_utils import train_markov_chain,window,cg_transition_matrix,train_markov_chain,print_stationary_distributions,post_process_topic_sequences
import sys
import pyemma
from tqdm.auto import tqdm

def prepare_input_data(metadata_field):
    # Find the relevant interview ids
    interview_codes = []
    metadata_field_names = []
    if "_w" in metadata_field:
        interview_codes_to_filter = IntCodeW
    elif "_m" in metadata_field:
        interview_codes_to_filter = IntCodeM
    else:
        interview_codes_to_filter = []

    if "complete" in metadata_field:
        interview_codes_temp = df_biodata.IntCode.to_list()

        if len(interview_codes_to_filter) > 1:
            interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
        interview_codes.append(interview_codes_temp)
        metadata_field_names.append(metadata_field)

    elif (('easy' in metadata_field) or ('medium' in metadata_field) or ('hard' in metadata_field)):
        interview_codes_temp = df_biodata[df_biodata[metadata_field.split('_')[0]] == 1].IntCode.tolist()

        if len(interview_codes_to_filter) > 1:
            interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
        interview_codes.append(interview_codes_temp)
        metadata_field_names.append(metadata_field)
    elif "notwork" in metadata_field:
        interview_codes_temp = df_biodata[
            (df_biodata['easy'] == 0) & (df_biodata['hard'] == 0) & (df_biodata['medium'] == 0)].IntCode.tolist()
        if len(interview_codes_to_filter) > 1:
            interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
        interview_codes.append(interview_codes_temp)
        metadata_field_names.append(metadata_field)
    elif "work" in metadata_field:
        interview_codes_temp = df_biodata[
            (df_biodata['easy'] == 1) | (df_biodata['hard'] == 1) | (df_biodata['medium'] == 1)].IntCode.tolist()
        if len(interview_codes_to_filter) > 1:
            interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
        interview_codes.append(interview_codes_temp)
        metadata_field_names.append(metadata_field)
    elif "CountryOfBirth" in metadata_field:
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
            interview_codes.append(interview_codes_temp)
            metadata_field_names.append(el)

    output_data = {}
    segment_indices = {}

    print(len(interview_codes), len(interview_codes[0]))
    # print(interview_codes)

    for sublist, fieldname in zip(interview_codes, metadata_field_names):
        output_data[fieldname] = []
        segment_indices[fieldname] = []
        for element in tqdm(sublist):
            # print(type(element), element)
            segment_index = segment_df[segment_df.IntCode.isin([element])].index.to_list()
            input_matrix = np.take(data, segment_index, axis=0)
            if input_matrix.size == 0:
                continue

            output_data[fieldname].append(input_matrix)
            segment_indices[fieldname].append(segment_df[segment_df.IntCode.isin([element])])

    return output_data, metadata_field_names, segment_indices


if __name__ == '__main__':


     # Read the input data
    input_directory = constants.output_data_segment_keyword_matrix

    # Read the segment index term matrix
    data = np.loadtxt(input_directory+ constants.output_segment_keyword_matrix_data_file, dtype=int)

   
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index)

    # Read the row index (groups of three segments) of the matrix above
    segment_df = pd.read_csv(input_directory+ constants.output_segment_keyword_matrix_document_index)

    # Read the input data
    input_directory = constants.input_data
    output_directory = constants.output_data_markov_modelling
    path = os.getcwd()

    # Read the biodata
    bio_data = constants.input_files_biodata_birkenau
    df_biodata = pd.read_csv(input_directory+bio_data)
    df_biodata = df_biodata.fillna(0)

    IntCodeM = df_biodata[df_biodata.Gender=='M']['IntCode'].to_list()
    IntCodeW = df_biodata[df_biodata.Gender=='F']['IntCode'].to_list()



    metadata_fields = ['complete','complete_m','complete_w','CountryOfBirth','CountryOfBirth_m','CountryOfBirth_w','easy_w','easy_m','medium_m','medium_w','hard_m','hard_w',"notwork","notwork_m","notwork_w","work","work_m","work_w"]
   
    

    np.set_printoptions(suppress=True)

 

    for metadata_field in metadata_fields:

        # Prepare the input data (trajectories)
        input_data_sets_dict, metadata_field_names, segment_indices_dict = prepare_input_data(metadata_field)

        for metadata_field_name in metadata_field_names:


            document_index = segment_indices_dict[metadata_field_name].groupby(['IntCode','SegmentNumber'])['KeywordLabel'].apply(list).to_frame('KeywordSequence').reset_index()
            print(metadata_field_name)

            trajectories = []
            for input_data_set in input_data_sets_dict[metadata_field_name]:

                # make sure this is binary assignment
                assert np.unique(np.sum(input_data_set, axis=1)) == np.array([1]), 'not one-hot encoded'

                # convert one-hot encoding to state definitions
                _t = np.argmax(input_data_set, axis=1)
                trajectories.append(_t)

            #TODO: validate choice of lag time
            mm = pyemma.msm.estimate_markov_model(trajectories, 1, reversible=False)
            transition_matrix = mm.transition_matrix

            print('active set fraction: ', mm.active_state_fraction)

            #TODO: re-write to take care of active set < full set
            stationary_prob = print_stationary_distributions(mm, features_df.KeywordLabel.to_list())
            pd.DataFrame(stationary_prob).to_csv(output_directory+metadata_field_name+'/stationary_probs.csv')

            if os.path.exists(output_directory+metadata_field_name+'/pyemma_model'):
                os.remove(output_directory+metadata_field_name+'/pyemma_model')

            mm.save(output_directory+metadata_field_name+'/pyemma_model', 'simple',overwrite=True)

            document_index.to_csv(output_directory+metadata_field_name+'/document_index.csv')
            np.savetxt(output_directory+metadata_field_name+'/transition_matrix.np', mm.transition_matrix, fmt='%.8f')
