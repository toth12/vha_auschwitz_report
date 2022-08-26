#!/bin/bash

set -x

mkdir -p data/output/{markov_modelling,reports_statistical_analysis,reports_statistical_analysis_expanded}




python simplify_features.py

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
python markov_modelling/create_bootstrapping_plots.py --metadata_fields work_m work_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields notwork_m notwork_w

python markov_modelling/create_bootstrapping_plots.py --metadata_fields Poland_m Poland_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Hungary_m Hungary_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Romania_m Romania_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Russia_m Russia_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Germany_m Germany_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Greece_m Greece_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Yugoslavia\ \(historical\)_m Yugoslavia\ \(historical\)_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Netherlands_m Netherlands_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields France_m France_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Austria_m Austria_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Italy_m Italy_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Austria-Hungary\ \(historical\)_m Austria-Hungary\ \(historical\)_w
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Czechoslovakia\ \(historical\)_m Czechoslovakia\ \(historical\)_w



python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields complete_w complete_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields notwork_w notwork_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields work notwork
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields work_m work_w
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields notwork_w notwork_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Poland_w Poland_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Hungary_w Hungary_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Romania_w Romania_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Russia_w Russia_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Germany_w Germany_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Greece_w Greece_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Yugoslavia\ \(historical\)_w Yugoslavia\ \(historical\)_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Netherlands_w Netherlands_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields France_w France_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Austria_w Austria_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Italy_w Italy_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Austria-Hungary\ \(historical\)_w Austria-Hungary\ \(historical\)_m
python statistical_analysis/measure_strength_of_assoc_odds_ratio_expanded.py --metadata_fields Czechoslovakia\ \(historical\)_w Czechoslovakia\ \(historical\)_m





