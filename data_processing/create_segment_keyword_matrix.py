"""Creates a segment-keyword matrix, feature index, and document; leaves only those keyword that in at least 100 interviews
"""

import constants
import pandas as pd
import pdb
import numpy as np
import argparse
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_count', nargs='+')
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "min_count"):
            if (value is not None):
                min_count = int(value[0])
            else:
                min_count = 0
    # Get the input file names and input directories
    # Input files are the segments data and the biodata about each interviewee



    input_directory = constants.input_data
    output_directory = constants.output_data_segment_keyword_matrix
    input_file = constants.input_segments_with_simplified_keywords

    if min_count ==0:
        output_segment_keyword_matrix = constants.output_segment_keyword_matrix_data_file
        output_document_index = constants.output_segment_keyword_matrix_document_index 
        output_feature_index = constants.output_segment_keyword_matrix_feature_index
    else:
        output_segment_keyword_matrix = constants.output_segment_keyword_matrix_data_file+'_'+str(min_count)
        output_document_index = constants.output_segment_keyword_matrix_document_index+'_'+str(min_count)
        output_feature_index = constants.output_segment_keyword_matrix_feature_index+'_'+str(min_count)
   
    df = pd.read_csv(input_directory + input_file)

    # Eliminate those index terms that occur in less than 100 interviews
    kws = df.groupby(['KeywordID', 'KeywordLabel'])['IntCode'].unique().map(lambda x: len(x)).to_frame(name="TotalNumberIntervieweeUsing").reset_index()
    kws_needed = kws[kws.TotalNumberIntervieweeUsing > min_count][['KeywordID' , 'KeywordLabel']]

    keywords = kws_needed.reset_index()[['KeywordID', 'KeywordLabel']]
    df = df[df['KeywordID'].isin(kws_needed['KeywordID'])]

    # Save the keywords that is used, this will be the feature index
    keywords.to_csv(output_directory + output_feature_index)
    
    #error introduced: https://github.com/toth12/vha_auschwitz_report/commit/80ccd72f3af82566e50a12f6ba5ebdc3e5f111b3#diff-50e843ca76aa2b9fa111c59a2d8d19ef
    # old version: segment_keyword = df.groupby(['IntCode', 'SegmentID', 'SegmentNumber','KeywordLabel'])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()
    
    # version 1 segment_keyword = df.groupby(['SegmentID'])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()
    # version 2
    segment_keyword = df.groupby(['IntCode', 'SegmentNumber','KeywordLabel'])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()

    # Create an empty np array that will hold this
    segment_keyword_matrix = np.zeros(shape=(len(segment_keyword),len(keywords)))

    # Iterate through the segment_keyword table
    segment_keyword_matrices = []
    intcodes = segment_keyword['IntCode'].to_numpy()
    segnums = segment_keyword['SegmentNumber'].to_numpy()
   
    kw_ids = segment_keyword['KeywordID'].to_numpy().astype(int)
    intcodes_final = []
    
    for intcode in tqdm(np.unique(intcodes)):
    #for intcode in np.unique(intcodes):
        kw_in_segm = kw_ids[intcodes == intcode]
        segnums_in_segm = segnums[intcodes == intcode]
        number_of_segments = len(set(segnums_in_segm.tolist()))

        #if number_of_segments == 1:
            #continue
        
        intcodes_final.append(intcode)
        unique_segments = list(set(segnums_in_segm.tolist()))
        unique_segments.sort()
        segment_keyword_matrix_single = np.zeros((number_of_segments, len(keywords)))
        
        for keyword, segnum in zip(kw_in_segm, segnums_in_segm):
            keyword_index = keywords[keywords.KeywordID == keyword].index[0]
            segment_index = unique_segments.index(segnum)
            segment_keyword_matrix_single[segment_index, keyword_index] += 1
    
        segment_keyword_matrices.append(segment_keyword_matrix_single)
    

    segment_keyword_matrix = np.array(segment_keyword_matrices)
    assert len(segment_keyword_matrix) ==len(intcodes_final)
    
    # Save the segment keyword matrix
    np.save(output_directory + output_segment_keyword_matrix, segment_keyword_matrix)
    #np.savetxt(output_directory + output_segment_keyword_matrix, segment_keyword_matrix, fmt='%d')
    segment_keyword = pd.DataFrame(intcodes_final,columns=["IntCode"])
    segment_keyword.to_csv(output_directory + output_document_index)
