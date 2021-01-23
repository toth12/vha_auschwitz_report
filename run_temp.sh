#!/bin/bash




cp data/input/feature_maps/feature_map.csv data/input/feature_map.csv

python data_processing/identify_story_end_beginning.py
python data_processing/simplify_features.py
python data_processing/infer_further_biodata.py
python data_processing/create_segment_keyword_matrix.py

python data_processing/map_indices_to_cover_terms.py
python data_processing/infer_birkenau_survivors.py
python data_processing/partition_interviews_into_metadata_groups.py