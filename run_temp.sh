#!/bin/bash

set -x

mkdir -p data/{output/{markov_modelling,reports_statistical_analysis,statistical_analysis,segment_keyword_matrix},input}
mkdir -p data/output/statistical_analysis/plots/{Gender,CountryOfBirth}
mkdir -p data/output/reports_statistical_analysis/{plots,tables}




python markov_modelling/create_bootstrapping_plots.py --metadata_fields complete_m complete_w