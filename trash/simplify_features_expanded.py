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


    input_directory = constants.input_directory
    input_segment_keyword_matrix = constants.segment_keyword_matrix_original
    input_feature_index = constants.segment_keyword_matrix_original_feature_index

    # Read the segment index term matrix
    data = np.load(input_segment_keyword_matrix, 
                  allow_pickle=True)
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_feature_index)


    feature_cover_term_map= pd.read_csv(constants.feature_map_expanded)
    

    cover_term_index = pd.DataFrame(feature_cover_term_map['CoverTerm'].unique(),columns=['CoverTerm']).sort_values('CoverTerm')
    #change

    final_terms = []
    for f,element in enumerate(features_df.KeywordLabel.to_list()):
        if element not in feature_cover_term_map.KeywordLabel.to_list():
            final_terms.append('camp '+ element)
        else:
            final_terms.append(element)

    features_df['KeywordLabel'] = final_terms
    
    #change


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

    
    output_segment_keyword_matrix = constants.segment_keyword_matrix_expanded
    output_feature_index = constants.segment_keyword_matrix_feature_index_expanded

    np.save(output_segment_keyword_matrix, result)
    cover_term_index = cover_term_index.rename(columns={"CoverTerm":"KeywordLabel"})
    cover_term_index.reset_index().to_csv(output_feature_index)


    '''
    
    dif = [element for element in cover_term_index.CoverTerm.to_list() if element not in merge_table.CoverTerm.to_list()]
    if len(dif)>0:
      pdb.set_trace()

    '''