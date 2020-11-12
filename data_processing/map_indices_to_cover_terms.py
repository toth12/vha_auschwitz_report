# coding: utf-8

import pdb
import numpy as np
import pandas as pd
import os
import constants
from tables import *
import json




if __name__ == '__main__':
    # Load the input data
    input_directory = constants.output_data_segment_keyword_matrix

    # Read the segment index term matrix
    data = np.load(input_directory + constants.output_segment_keyword_matrix_data_file.replace('.txt', '.npy'), 
                  allow_pickle=True)
    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory + 
                          constants.output_segment_keyword_matrix_feature_index)

    # Create the row index  of the matrix above
    segment_df = pd.read_csv(input_directory + 
                         constants.output_segment_keyword_matrix_document_index)

    feature_cover_term_map= pd.read_csv('feature_map.csv')
    cover_term_index = pd.DataFrame(feature_cover_term_map['CoverTerm'].unique(),columns=['CoverTerm']).sort_values('CoverTerm')


    for interview in data:
      for segment in interview:
        merge_table_indices = features_df.copy()
        merge_table_covering_terms = feature_cover_term_map.copy()
        
        merge_table_indices['count']=segment

        merge_table=merge_table_covering_terms.merge(merge_table_indices)
        
        merge_table = merge_table.groupby('CoverTerm')['count'].sum().to_frame('count').reset_index()
        
        dif = [element for element in cover_term_index.CoverTerm.to_list() if element not in merge_table.CoverTerm.to_list()]
        pdb.set_trace()