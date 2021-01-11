#!/bin/bash
mkdir -p data/{output/{markov_modelling,reports_statistical_analysis,statistical_analysis,segment_keyword_matrix},input}
mkdir -p data/output/statistical_analysis/plots/{Gender,CountryOfBirth}
mkdir -p data/output/reports_statistical_analysis/{plots,tables}


cp data/input/feature_maps/feature_map_with_friends_food_sharing.csv data/input/feature_map.csv

python data_processing/identify_story_end_beginning.py
python data_processing/simplify_features.py
python data_processing/infer_further_biodata.py
python data_processing/create_segment_keyword_matrix.py
python data_processing/map_indices_to_cover_terms.py
python data_processing/infer_birkenau_survivors.py
python data_processing/partition_interviews_into_metadata_groups.py
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields complete_w complete_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields notwork_w notwork_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields work notwork
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields work_m work_w
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields notwork_w notwork_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Poland_w Poland_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Hungary_w Hungary_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Romania_w Romania_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Russia_w Russia_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Germany_w Germany_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Greece_w Greece_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Yugoslavia\ \(historical\)_w Yugoslavia\ \(historical\)_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Netherlands_w Netherlands_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields France_w France_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Austria_w Austria_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Italy_w Italy_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Austria-Hungary\ \(historical\)_w Austria-Hungary\ \(historical\)_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Czechoslovakia\ \(historical\)_w Czechoslovakia\ \(historical\)_m
python markov_modelling/train_markov_model.py
python markov_modelling/create_bootstrapping_plots.py --metadata_fields complete_m complete_w
pytest markov_modelling/test_msm.py
python markov_modelling/create_bootstrapping_plots.py --metadata_fields work notwork
python markov_modelling/create_bootstrapping_plots.py --metadata_fields work_w work_m


mv data/output data/output_aid_giving_sociability_expanded

mkdir -p data/{output/{markov_modelling,reports_statistical_analysis,statistical_analysis,segment_keyword_matrix},input}
mkdir -p data/output/statistical_analysis/plots/{Gender,CountryOfBirth}
mkdir -p data/output/reports_statistical_analysis/{plots,tables}

cp data/input/feature_maps/feature_map_no_friends_no_food_sharing.csv data/input/feature_map.csv

python data_processing/identify_story_end_beginning.py
python data_processing/simplify_features.py
python data_processing/infer_further_biodata.py
python data_processing/create_segment_keyword_matrix.py
python data_processing/map_indices_to_cover_terms.py
python data_processing/infer_birkenau_survivors.py
python data_processing/partition_interviews_into_metadata_groups.py
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields complete_w complete_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields notwork_w notwork_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields work notwork
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields work_m work_w
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields notwork_w notwork_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Poland_w Poland_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Hungary_w Hungary_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Romania_w Romania_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Russia_w Russia_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Germany_w Germany_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Greece_w Greece_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Yugoslavia\ \(historical\)_w Yugoslavia\ \(historical\)_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Netherlands_w Netherlands_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields France_w France_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Austria_w Austria_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Italy_w Italy_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Austria-Hungary\ \(historical\)_w Austria-Hungary\ \(historical\)_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields Czechoslovakia\ \(historical\)_w Czechoslovakia\ \(historical\)_m
python markov_modelling/train_markov_model.py
python markov_modelling/create_bootstrapping_plots.py --metadata_fields complete_m complete_w
#python markov_modelling/create_bootstrapping_plots.py --metadata_fields work notwork
#python markov_modelling/create_bootstrapping_plots.py --metadata_fields work_w work_m
pytest markov_modelling/test_msm.py