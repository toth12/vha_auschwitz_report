"""Constants used throughout this project."""


# Input folder

input_directory = "data/input/"


# Input data sets

segment_keyword_matrix_original = input_directory + "segment_keyword_matrix_original.npy"
segment_keyword_matrix_original_feature_index = input_directory + "feature_index_original.csv"

feature_map = input_directory + "feature_map_no_friends_no_food_sharing_2.csv"
feature_map_expanded = input_directory + "feature_map_cover_term_expanded.csv"

segment_keyword_matrix_feature_index = input_directory + "feature_index.csv"
segment_keyword_matrix = input_directory + "segment_keyword_matrix.npy"

segment_keyword_matrix_feature_index_expanded = input_directory + "feature_index_expanded.csv"
segment_keyword_matrix_expanded = input_directory + "segment_keyword_matrix_expanded.npy"

metadata_partitions = input_directory + "metadata_partitions.json"

# Output directories

output_directory = "data/output/"
output_data_report_statistical_analysis = output_directory+'reports_statistical_analysis/'
output_data_report_statistical_analysis_expanded = output_directory+'reports_statistical_analysis_expanded/'


output_data_markov_modelling= output_directory+'markov_modelling/'