#python data_processing/identify_story_end_beginning.py
#python data_processing/simplify_features.py
#python data_processing/infer_further_biodata.py
python data_processing/create_segment_keyword_matrix.py
python data_processing/map_indices_to_cover_terms.py
python data_processing/infer_birkenau_survivors.py
python data_processing/partition_interviews_into_metadata_groups.py
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields complete_w complete_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields notwork_w notwork_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields work notwork
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields work_m work_w
python statistical_analysis/measure_strength_of_assoc_odds_ratio.py --metadata_fields notwork_w notwork_m
python markov_modelling/train_markov_model.py
python markov_modelling/create_bootstrapping_plots.py --metadata_fields complete_m complete_w
#python markov_modelling/create_bootstrapping_plots.py --metadata_fields work notwork
#python markov_modelling/create_bootstrapping_plots.py --metadata_fields work_w work_m