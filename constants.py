"""Constants used throughout this project."""


# Input folder

input_data = "data/input/"
input_data_filtered = "data/input/filtered/"

# Input data sets

input_files_segments = ["Auschwitz_segments_03112020_1.csv","Auschwitz_segments_03112020_2.csv"]
input_files_biodata='biodata.xlsx'
input_files_term_hierarchy = 'termhierarchy_3.json'
input_segments_with_simplified_keywords = "all_segments_only_Jewish_survivors_generic_terms_deleted_below_25_replaced_for_parent_node.csv"

# Filtered input data

input_files_biodata_birkenau='biodata_birkenau.csv'
input_files_biodata_with_inferred_fields='biodata_with_inferred_fields.csv'
feature_map = 'feature_map.csv'

# Output data folders

output_data = "data/output/"
output_data_statistical_analysis = output_data+'statistical_analysis/'
output_data_segment_keyword_matrix = output_data +"segment_keyword_matrix/"
output_data_report_statistical_analysis = output_data+'reports_statistical_analysis/'
output_chi2_test = output_data+'chi2test/'
output_chi2_test_birkenau = output_data+'chi2test_birkenau/'
output_data_features = "data/output/features/"
output_data_filtered_nodes = "data/output/filtered_nodes/"
output_data_topic_sequences= output_data+'topic_sequencing/'
output_data_topic_sequence_preprocessed= output_data+'topic_sequences_preprocessed/'
output_data_markov_modelling= output_data+'markov_modelling/'
output_data_markov_modelling_aggregated_reports= output_data+'markov_modelling/aggregated_reports/'
# Output files


output_segment_keyword_matrix_data_file_100 = "segment_keyword_matrix_100.txt"
output_segment_keyword_matrix_document_index_100 = "document_index_100.csv"
output_segment_keyword_matrix_feature_index_100 = "feature_index_100.csv"
output_segment_keyword_matrix_document_index = "document_index.csv"
output_segment_keyword_matrix_feature_index = "feature_index.csv"
output_segment_keyword_matrix_data_file = "segment_keyword_matrix.npy"
output_segment_keyword_matrix_data_file_txtfmt = "segment_keyword_matrix.txt"