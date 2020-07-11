"""Creates a segment-keyword matrix, feature index, and document; leaves only those keyword that in at least 100 interviews
"""

import constants
import pandas as pd
import pdb
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_count', nargs='+')
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "min_count"):
            if (value is not None):
                min_count = int(value[0])
            else:
                min_count = 25
    # Get the input file names and input directories
    # Input files are the segments data and the biodata about each interviewee



    input_directory = constants.input_data
    output_directory = constants.output_data_segment_keyword_matrix
    input_file = constants.input_segments_with_simplified_keywords

    if min_count ==25:
        output_segment_keyword_matrix = constants.output_segment_keyword_matrix_data_file
        output_document_index = constants.output_segment_keyword_matrix_document_index 
        output_feature_index = constants.output_segment_keyword_matrix_feature_index
    else:
        output_segment_keyword_matrix = constants.output_segment_keyword_matrix_data_file_100
        output_document_index = constants.output_segment_keyword_matrix_document_index_100
        output_feature_index = constants.output_segment_keyword_matrix_feature_index_100
   
    df = pd.read_csv(input_directory + input_file)

    # Eliminate those index terms that occur in less than 100 interviews
    kws = df.groupby(['KeywordID', 'KeywordLabel'])['IntCode'].unique().map(lambda x: len(x)).to_frame(name="TotalNumberIntervieweeUsing").reset_index()
    kws_needed = kws[kws.TotalNumberIntervieweeUsing > 0][['KeywordID' , 'KeywordLabel']]

    keywords = kws_needed.reset_index()[['KeywordID', 'KeywordLabel']]
    df = df[df['KeywordID'].isin(kws_needed['KeywordID'])]

    # Save the keywords that is used, this will be the feature index
    keywords.to_csv(output_directory + output_feature_index)
    segment_keyword = df.groupby(['IntCode', 'SegmentID', 'SegmentNumber'])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()

    # Create an empty np array that will hold this
    segment_keyword_matrix = np.zeros(shape=(len(segment_keyword),len(keywords)))

    # Iterate through the segment_keyword table
    for i, element in enumerate(segment_keyword.iterrows()):
        for keyword in element[1]['KeywordID']:
            keyword_index = keywords[keywords.KeywordID == keyword].index[0]
            segment_keyword_matrix[i, keyword_index] = 1

    # Save the segment keyword matrix

    np.savetxt(output_directory + output_segment_keyword_matrix, segment_keyword_matrix, fmt='%d')
    segment_keyword.to_csv(output_directory + output_document_index)
