#!/bin/bash
#python markov_modelling/train_markov_model.py



python markov_modelling/create_bootstrapping_plots.py --metadata_fields work notwork
python markov_modelling/create_bootstrapping_plots.py --metadata_fields work_w work_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields notwork_w notwork_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Poland_w Poland_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Hungary_w Hungary_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Romania_w Romania_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Russia_w Russia_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Germany_w Germany_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Greece_w Greece_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Yugoslavia\ \(historical\)_w Yugoslavia\ \(historical\)_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Netherlands_w Netherlands_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields France_w France_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Austria_w Austria_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Italy_w Italy_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Austria-Hungary\ \(historical\)_w Austria-Hungary\ \(historical\)_m
python markov_modelling/create_bootstrapping_plots.py --metadata_fields Czechoslovakia\ \(historical\)_w Czechoslovakia\ \(historical\)_m