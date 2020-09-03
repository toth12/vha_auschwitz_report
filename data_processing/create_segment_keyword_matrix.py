"""Creates a segment-keyword matrix, feature index, and document; leaves only those keyword that in at least 100 interviews
"""

import constants
import pandas as pd
import pdb
import numpy as np
import argparse
from tqdm.auto import tqdm
import more_itertools as mit



def group_segments(data,max_gap=5):
    result = [list(group) for group in mit.consecutive_groups(data)]
    #check if merge is possible
    if len(result)>1:
        new_result = []
        for i,group in enumerate(result):
            if i == 0:
                new_result.append(group)
            else:
                first_element = group[0]
                last_element_prev_group = result[i-1][-1]
                
                if (first_element - last_element_prev_group) <=max_gap:
                    new_result[-1].extend(group)
                else:
                    new_result.append(group)
        return new_result

    else:    
        return result


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
        output_segment_keyword_matrix_txtfmt = constants.output_segment_keyword_matrix_data_file_txtfmt
        output_document_index = constants.output_segment_keyword_matrix_document_index 
        output_feature_index = constants.output_segment_keyword_matrix_feature_index
    else:
        output_segment_keyword_matrix = constants.output_segment_keyword_matrix_data_file+'_'+str(min_count)
        output_segment_keyword_matrix_txtfmt = constants.output_segment_keyword_matrix_data_file_txtfmt + '_' + str(min_count)
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
    segment_keyword = df.groupby(['IntCode', 'SegmentID', 'SegmentNumber','KeywordLabel'])["KeywordID"].unique().to_frame(name="KeywordID").reset_index()

    intcodes = segment_keyword['IntCode'].to_numpy()
    segnums = segment_keyword['SegmentNumber'].to_numpy()
    kw_ids = segment_keyword['KeywordID'].to_numpy().astype(int)

    intcode_index = []
    interview_lengths = []
    segment_keyword_matrices = []
    
    for intcode in tqdm(np.unique(intcodes)):
        kw_in_segm = kw_ids[intcodes == intcode]
        segnums_in_segm = segnums[intcodes == intcode]
        group_num = 0
        segnums_in_segm_grouped = group_segments(segnums_in_segm)
        for f,group in enumerate(segnums_in_segm_grouped):
            if len(group) >1:
                intcode_index.append(str(intcode)+'_'+str(f))
                kw_in_segm_group = kw_in_segm[group_num:group_num+len(group)]
                group_num = group_num + len(group)
                l = np.array(group).max() - np.array(group).min()
                interview_lengths.append(l)
                segment_keyword_matrix_single = np.zeros((l+1, len(keywords)))
                
                for keyword, segnum in zip(kw_in_segm_group, group):
                    keyword_index = keywords[keywords.KeywordID == keyword].index[0]

                    # add one, don't overwrite -> enable multiple keywords per segment
                    segment_keyword_matrix_single[segnum - np.array(group).min(), keyword_index] += 1
                
                segment_keyword_matrices.append(segment_keyword_matrix_single)
            else:
                continue
        


    segment_keyword_matrix = np.array(segment_keyword_matrices)

    # Test 53029
    #indices_53029 = [intcode_index.index(element) for element in intcode_index if element.split('_')[0]=='53029']
    #lenghts_53029 = np.array([len(element) for element in segment_keyword_matrices[indices_53029[0]:indices_53029[-1]]]).sum()
 
    #assert len(segment_index) == len(segment_keyword_matrices)
    n_interviews = len(interview_lengths)
    total_minutes = sum(interview_lengths)
    print(f'total interviews: {n_interviews}')
    print(f'total minutes: {total_minutes}')
    # Create the index
    segment_index = pd.DataFrame(intcode_index,columns=['segment_index'])
    segment_index['IntCode'] = segment_index.segment_index.apply((lambda x: x.split('_')[0]))



    # loading huge text files is very slow, also saving in binary format
    np.save(output_directory + output_segment_keyword_matrix, segment_keyword_matrix)

    # Save the segment keyword matrix
    np.savetxt(output_directory + output_segment_keyword_matrix_txtfmt, np.vstack(segment_keyword_matrix), fmt='%d')

    # Save the Segment Index index

    segment_index.to_csv(output_directory + output_document_index)

   
