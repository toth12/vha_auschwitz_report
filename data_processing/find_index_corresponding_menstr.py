#!/usr/bin/env python
# coding: utf-8

'''Creates a two dimensional count matrix: columns are women and men, rows are the keywords; it does a pairwise comparison
of each feature with a contingency table (number of women mentioning a topic, number of men mentioning a topic, number of women not mentioning a topic
number of men not mentioning a topic; tests if there is a statistical signficance between them and measures the odds ratio for women and men
finally does a Bonferroni correction for the final statistical significance test'''

import pdb
import numpy as np
import pandas as pd
import os
import constants
import json
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
import argparse


if __name__ == '__main__':

    # Load the metadata fields
    # python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields complete_m complete_w
    
    metadata_fields = ['complete','complete_m','complete_w','easy_w','easy_m','medium_m','medium_w','hard_m','hard_w',"notwork","notwork_m","notwork_w","work","work_m","work_w"]
    
    # Add countries

    bio_data = constants.input_files_biodata_birkenau
    df_biodata = pd.read_csv(constants.input_data+bio_data)
    df_biodata = df_biodata.fillna(0)
    country_of_origins = df_biodata.groupby('CountryOfBirth')['CountryOfBirth'].count().to_frame('Count').reset_index().CountryOfBirth.to_list()
    country_of_origins_with_gender = []
    # Add male and female
    for element in country_of_origins:
        country_of_origins_with_gender.append(element+'_w')
        country_of_origins_with_gender.append(element+'_m')
    metadata_fields = metadata_fields + country_of_origins_with_gender

    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_fields', nargs='+')
    metadata_fields_to_agregate = []
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "metadata_fields"):
            for field in value:
                if (field not in metadata_fields):
                    print ("The following metadata_field is not valid")
                    print (field)
                    pdb.set_trace()
                else:
                    metadata_fields_to_agregate.append(field)

    
    # Load the input data
    input_directory = constants.output_data_segment_keyword_matrix
    #input_directory = '/Users/gmt28/Documents/Workspace/vha_auschwitz_report_public/vha_auschwitz_report/data/output_aid_giving_sociability_expanded/output/segment_keyword_matrix/'
    # Read the segment index term matrix
    data = np.load(input_directory + constants.output_segment_keyword_matrix_data_file.replace('.txt', '.npy'), 
                  allow_pickle=True)

    # Read the column index (index terms) of the matrix above
    features_df = pd.read_csv(input_directory + 
                          constants.output_segment_keyword_matrix_feature_index)
    
  
    #features_df = pd.DataFrame(features_df.sort_values('index')["KeywordLabel"].to_list(),columns=["KeywordLabel"])
    
    # Create the row index  of the matrix above
    segment_df = pd.read_csv(input_directory + 
                         constants.output_segment_keyword_matrix_document_index)

    int_codes = segment_df['IntCode'].to_list()

    # Set the output directory
    output_directory = constants.output_data_report_statistical_analysis
    #output_directory = '/Users/gmt28/Documents/Workspace/vha_auschwitz_report_public/vha_auschwitz_report/data/output_aid_giving_sociability_expanded/output/reports_statistical_analysis/'
    output_file = 'strength_of_association_odds_ratio_'+'_'.join(metadata_fields_to_agregate)+'.csv'
    # Read the metadata partitions
    with open(input_directory + "metadata_partitions.json") as read_file:
        metadata_partitions = json.load(read_file)

    # First check for women and then men
    partial_results = []
    totals = []
    metadata_partitions_temp = {}
    metadata_partitions_temp['complete_w'] = metadata_partitions['complete_w']
    metadata_partitions = metadata_partitions_temp
    # Get the relevant data
    for element in metadata_fields_to_agregate:
        interview_keyword_matrices = []
        indices = metadata_partitions[element]
        input_data_set = np.take(data,indices)
        totals.append(len(input_data_set))
        print (totals)
        # Iterare through the individual interviews (represented as a segment-keyword matrix)
        for interview in input_data_set:

            # Binarize each interview and transform them into a one dimensional matrix (each interview represented as a set of keywords)
            int_keyword_matrix =interview.sum(0)

            # binarize it (we are just checking if a topic is mentioned independently from how many times)
            int_keyword_matrix = np.where(int_keyword_matrix > 0, 1, 0)

            # Add the result to the lists that holds them
            interview_keyword_matrices.append(int_keyword_matrix)

        # Transform the individual interview keyword matrices into one count matrix (women or men  - keywords with total count)
        interview_keyword_matrices = np.vstack(interview_keyword_matrices)

        # Make the count matrix and add it to the list that stores it
        feature_counts = interview_keyword_matrices.sum(0)
        partial_results.append(feature_counts)
        pdb.set_trace()
    # Make a count matrix with rows as features and the first column for women and the second one for women
    complete_result = np.vstack(partial_results).T
   