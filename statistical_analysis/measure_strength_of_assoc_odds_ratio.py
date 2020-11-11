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

    int_codes = segment_df['IntCode'].to_list()

    # Set the output directory
    output_directory = constants.output_data_report_statistical_analysis
    output_file = 'strength_of_association_men_women_odds_ratio.csv'

    # Read the metadata partitions
    with open(input_directory + "metadata_partitions.json") as read_file:
        metadata_partitions = json.load(read_file)

    # First check for women and then men
    metadata_fields = ['complete_w','complete_m']
    partial_results = []
    totals = []

    # Get the relevant data
    for element in metadata_fields:
        interview_keyword_matrices = []
        indices = metadata_partitions[element]
        input_data_set = np.take(data,indices)

        ### For every every interview create a one dimensional interview keyword matrix as a first step

        # Get the total number of women first and then men in the sample (later to be used for the multiple comparison test)
        totals.append(len(input_data_set))

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

    # Make a count matrix with rows as features and the first column for women and the second one for women
    complete_result = np.vstack(partial_results).T

    # Use camp menstruation (a women topic) as a check point
    # Women definitely discuss this topic more than men

    index_mens = features_df[features_df['KeywordLabel']=='camp menstruation'].index[0]
    assert complete_result[index_mens][0] > complete_result[index_mens][1]

    # Make a pairwise comparison of all features
    final_results = []
    for i,row in enumerate(complete_result):

        # Create a contingency table for every feature
        mentioned_w = row[0]
        mentioned_m = row[1]

        # Skip the topic if mentioned less than 5 by either women or men due to statistical insignificance

        if (mentioned_w<0) or (mentioned_m <0):
            continue

        not_mentioned_w =  totals[0]-mentioned_w
        not_mentioned_m =  totals[1]-mentioned_m

        # Apply T-test and calculate odds ratios (a topic mentioned and it is a woman who mentions it and a topic mentioned and it is a man who mentions)
        # see, https://stackoverflow.com/questions/61023380/calculating-odds-ratio-in-python
        # Calculate odds ratio for the women (upper left corner of the contingency table)

        contingency_w = [[mentioned_w,mentioned_m],[not_mentioned_w,not_mentioned_m]]
        oddsratio_w, pvalue_w = stats.fisher_exact(contingency_w)

        # Calculate odds ratio for the men

        contingency_m = [[mentioned_m,mentioned_w],[not_mentioned_m,not_mentioned_w]]
        oddsratio_m, pvalue_m = stats.fisher_exact(contingency_m)

        # Save results into a dictionary, (p_value same for men and women)

        part_result = {'topic_word':features_df.iloc()[i].KeywordLabel,'p_value':pvalue_w,'women':oddsratio_w,'men':oddsratio_m,'count_w':mentioned_w,"count_m":mentioned_m}
        final_results.append(part_result)

    # Put results into a panda df
    df_final_results = pd.DataFrame(final_results)

    # Make a Bonferroni correction
    df_final_results['significance_Bonferroni_corrected'] = multipletests(df_final_results['p_value'], method='bonferroni')[0]

    # Sort results according to p_value
    df_final_results = df_final_results.sort_values('p_value')

    df_final_results.to_csv(output_directory+output_file)