import pandas as pd
import pdb
import numpy as np
import msmtools
import constants
import os
from msmtools.estimation import is_connected
from markov_utils import train_markov_chain,window,cg_transition_matrix,train_markov_chain,print_stationary_distributions,post_process_topic_sequences
import sys



def prepare_input_data(metadata_field):
    # Find the relevant interview ids
    interview_codes = []
    metadata_field_names = []
    if "_w" in metadata_field:
            interview_codes_to_filter=IntCodeW
    elif "_m" in metadata_field:
            interview_codes_to_filter=IntCodeM
    else:
            interview_codes_to_filter =[]
    
    if "complete" in metadata_field:
        interview_codes_temp = df_biodata.IntCode.to_list()

        if len(interview_codes_to_filter)>1:
            interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
        interview_codes.append(interview_codes_temp)
        metadata_field_names.append(metadata_field)

    elif (('easy' in metadata_field) or ('medium' in metadata_field) or ('hard' in metadata_field)):
        interview_codes_temp = df_biodata[df_biodata[metadata_field.split('_')[0]]==1].IntCode.tolist()

        if len(interview_codes_to_filter)>1:
            interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
        interview_codes.append(interview_codes_temp)
        metadata_field_names.append(metadata_field)
    elif "notwork" in metadata_field:
        interview_codes_temp = df_biodata[(df_biodata['easy']==0)&(df_biodata['hard']==0)&(df_biodata['medium']==0)].IntCode.tolist()
        if len(interview_codes_to_filter)>1:
            interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
        interview_codes.append(interview_codes_temp)
        metadata_field_names.append(metadata_field)
    elif "work" in metadata_field:
        interview_codes_temp = df_biodata[(df_biodata['easy']==1)|(df_biodata['hard']==1)|(df_biodata['medium']==1)].IntCode.tolist()
        if len(interview_codes_to_filter)>1:
            interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
        interview_codes.append(interview_codes_temp)
        metadata_field_names.append(metadata_field)
    elif "CountryOfBirth" in metadata_field:
        country_of_origins = df_biodata.groupby('CountryOfBirth')['CountryOfBirth'].count().to_frame('Count').reset_index()
        country_of_origins= country_of_origins[country_of_origins.Count>50]
        country_of_origins = country_of_origins.CountryOfBirth.tolist()
        for el in country_of_origins:
            interview_codes_temp = df_biodata[df_biodata['CountryOfBirth']==el].IntCode.to_list()
            if len(interview_codes_to_filter)>1:
                interview_codes_temp = [f for f in interview_codes_temp if f in interview_codes_to_filter]
                if "_w" in metadata_field:
                    el = el+'_w'
                else:
                    el = el+'_m'
            interview_codes.append(interview_codes_temp)
            metadata_field_names.append(el)

    output_data = []
    segment_indices = []

    for element in interview_codes:
        segment_index= segment_df[segment_df.IntCode.isin(element)].index.to_list()
        input_matrix = np.take(data,segment_index,axis=0)
        output_data.append(input_matrix)
        segment_indices.append(segment_df[segment_df.IntCode.isin(element)])

    return output_data,metadata_field_names,segment_indices


if __name__ == '__main__':


     # Read the input data
    input_directory = constants.output_data_segment_keyword_matrix

    # Read the segment index term matrix
    data = np.loadtxt(input_directory+ constants.output_segment_keyword_matrix_data_file_100, dtype=int)
   
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory+constants.output_segment_keyword_matrix_feature_index_100)

    # Read the row index (groups of three segments) of the matrix above
    segment_df = pd.read_csv(input_directory+ constants.output_segment_keyword_matrix_document_index_100)

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
        input_data_sets,metadata_field_names,segment_indices=prepare_input_data(metadata_field)

        for f,input_data_set in enumerate(input_data_sets):

            try:
                
                document_index = segment_indices[f].groupby(['IntCode','SegmentNumber'])['KeywordLabel'].apply(list).to_frame('KeywordSequence').reset_index()
                print (metadata_field_names[f])


                
                (unique, counts) = np.unique(input_data_set,axis=0, return_counts=True)
                trajectories = []
                for i,element in enumerate(input_data_set):

                    trajectory = np.where(np.all(unique==element,axis=1))[0][0]
                    trajectories.append(trajectory)
            
                tr = [el for el in window(trajectories)]
                count_matrix = np.zeros((unique.shape[0],unique.shape[0])).astype(float)

                

                for element in tr:

                    count_matrix[element[0],element[1]]=count_matrix[element[0],element[1]]+float(1)


                count_matrix = count_matrix +1e-12
                transition_matrix = (count_matrix / count_matrix.sum(axis=1,keepdims=1))
                assert np.allclose(transition_matrix.sum(axis=1), 1)
                assert msmtools.analysis.is_transition_matrix(transition_matrix)
                assert is_connected(transition_matrix)


                binary_map = (unique / unique.sum(axis=1,keepdims=1))
                new_tra = cg_transition_matrix(transition_matrix,binary_map)
                new_tra[np.isnan(new_tra)] = 0
                new_tra = new_tra+1e-12
                new_tra= (new_tra / new_tra.sum(axis=1,keepdims=1))
                assert np.allclose(new_tra.sum(axis=1), 1)
                assert msmtools.analysis.is_transition_matrix(new_tra)
                assert is_connected(new_tra)

                try:
                    os.mkdir(output_directory+metadata_field_names[f])
                except:
                    pass

                mm = train_markov_chain(new_tra)

                stationary_prob = print_stationary_distributions(mm,features_df.KeywordLabel.to_list())
                pd.DataFrame(stationary_prob).to_csv(output_directory+metadata_field_names[f]+'/stationary_probs.csv')

                if os.path.exists(output_directory+metadata_field_names[f]+'/pyemma_model'):
                    os.remove(output_directory+metadata_field_names[f]+'/pyemma_model')
               
                mm.save(output_directory+metadata_field_names[f]+'/pyemma_model', 'simple',overwrite=True)
                
                document_index.to_csv(output_directory+metadata_field_names[f]+'/document_index.csv')
                np.savetxt(output_directory+metadata_field_names[f]+'/transition_matrix.np', new_tra, fmt='%.8f')
           
            except ValueError:
                error_type, error_instance, traceback = sys.exc_info()
                error_instance.args = (error_instance.args[0] + ' <modification>',)
                
                print ("The transition matrix could not be made in case of the folllowing metadata field:")
                print (metadata_field_names[f]) 
                
                continue
