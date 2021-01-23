# coding: utf-8

import pdb
import numpy as np
import pandas as pd
import os
import constants
from tables import *
import json
from tqdm.auto import tqdm




if __name__ == '__main__':
    # Load the input data
    # feature_cover_term_map[~feature_cover_term_map.KeywordLabel.isin(features_df.KeywordLabel.to_list())]
    # len(result[0][0]) == len(cover_term_index)
    input_directory = constants.output_data_segment_keyword_matrix
    input_segment_keyword_matrix = constants.output_segment_keyword_matrix_data_file_original
    input_document_index = constants.output_segment_keyword_matrix_document_index
    input_feature_index = constants.output_segment_keyword_matrix_feature_index_original

    # Read the segment index term matrix
    data = np.load(input_directory + input_segment_keyword_matrix, 
                  allow_pickle=True)
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory + 
                          input_feature_index)


    feature_cover_term_map= pd.read_csv(constants.input_data+constants.feature_map)
    cover_term_index = pd.DataFrame(feature_cover_term_map['CoverTerm'].unique(),columns=['CoverTerm']).sort_values('CoverTerm')

    result = []
    for interview in tqdm(data):
      interview_result = []
      for segment in interview:
        merge_table_indices = features_df.copy()
        merge_table_covering_terms = feature_cover_term_map.copy()
        
        merge_table_indices['count']=segment

        merge_table=merge_table_covering_terms.merge(merge_table_indices)
        
        merge_table = merge_table.groupby('CoverTerm')['count'].sum().to_frame('count').reset_index()
        segment_coverterm_matrix = cover_term_index.merge(merge_table)['count'].to_numpy()
        interview_result.append(segment_coverterm_matrix)

      interview_result = np.stack(interview_result)
      assert len(interview_result) > 1
      result.append(interview_result)
    result = np.array(result)

    
    output_directory = constants.output_data_segment_keyword_matrix
    output_segment_keyword_matrix = constants.output_segment_keyword_matrix_data_file
    output_feature_index = constants.output_segment_keyword_matrix_feature_index

    np.save(output_directory + output_segment_keyword_matrix, result)
  
    cover_term_index = cover_term_index.rename(columns={"CoverTerm":"KeywordLabel"})
    cover_term_index.reset_index(drop=True).to_csv(output_directory + output_feature_index)


    '''
    
    dif = [element for element in cover_term_index.CoverTerm.to_list() if element not in merge_table.CoverTerm.to_list()]
    if len(dif)>0:
      pdb.set_trace()

    '''