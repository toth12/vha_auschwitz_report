# VHA Auschwitz Reports

This repository contains python workflows to investigate testimonies by survivors of Auschwitz-Birkenau concentration camp. Testimonies that these workflows analyze were provided by the USC Shoah Foundation.

## Data set

The dataset that Shoah Foundation gave consists of three elements:
1. Segment data (Auschwitz_segments_03112020_1.csv,Auschwitz_segments_03112020_2.csv)

This contains all testimony segments in which survivors address their stay in Auschwitz. It is a table in which every row corresponds to a testimony segment and the topic words attached to it. Since multiple topics words (sometimes refered as keywords here) are typically attached to a segment, multiple rows can represent a given segment. For instance, the following three rows represent the segment 16 (SegmentID: 514929) of interview 2 (IntCode 2) with interviewee named Rosalie Greenfield :



2   Rosalie Greenfield  514929  16  1   00:15:00:00 1   00:16:00:00 7601    Auschwitz II-Birkenau (Poland : Death Camp)

2   Rosalie Greenfield  514929  16  1   00:15:00:00 1   00:16:00:00 13310   Oświęcim (Kraków, Poland)

2   Rosalie Greenfield  514929  16  1   00:15:00:00 1   00:16:00:00 14226   Poland 1944 (July 22) - 1945 (January 16)

This interview segment has been annotated with three topic words:Auschwitz II-Birkenau (Poland : Death Camp),Oświęcim (Kraków, Poland),Poland 1944 (July 22) - 1945 (January 16). The unique ids (KeywordID) of these topic words: 7601,13310,14226.

The record also tells the beginning and the end of the segment:00:15:00:00 and 00:16:00:00

The segment data has been further processed (see below), most importantly, a segment-keyword matrix was constructed.

2. Biodata (biodata.xlsx)

This contains basic bio data about each interviewee and metadata about the interview: gender, country of birth, place where the interview was recorded, interview language, etc.

The biodata is connected with the segment data through the IntCode column.

3. The topic word hiearchy (termhierarchy_3.json):

The Shoah Foundation's topic word system is a hierarchical tree; this file contains this tree. The topic words in this file are connected with the segment data through the KeywordID.

## Getting Started

1. Git clone this library
2. Install python requirements (code compatible with Python 3 only and tested only on Macintosh):

```
pip install -r requirements.txt
```

3. Create the data directory (ignored by git) with the following command:

```
mkdir -p data/{output/{markov_modelling,reports_statistical_analysis,statistical_analysis,segment_keyword_matrix},input}
```


```
mkdir -p data/output/statistical_analysis/plots/{Gender,CountryOfBirth}
```

```
mkdir -p data/output/reports_statistical_analysis/{plots,tables}
```

4. Get the following input files and copy them to data/input:

* Auschwitz_segments_03112020_1.csv
* Auschwitz_segments_03112020_2.csv
* biodata.xlsx
* termhierarchy_3.json (todo: rename this)
* forced_labour_typology.csv

## Workflows available in this repo:

* Preprocessing of the data (prerequisit of running any workflow below)
* Baisc Statistical analysis of the data set
* Chi2 significance test and strength of association (odds ration) of index terms for Gender and CountryOfOrigin
* Markov State Modelling of the data set

## Preprocessing of the data

### Simplify keywords:

The Shoah Foundation's keyword system is a hierarchical tree; with this script, those nodes (i.e. keywords) that are leaves and used in less then 25 interviews are replaced for their parent node (for instance, the parent node of "suicide attampt" is "suicide"). This script also eliminates testimony segments by non-Jewish survivors. Furthermore, those keywords that describe places, names, and historical events are also removed.

```
python data_processing/simplify_features.py
```

Input data:
* complete input data as defined above

Output data (saved to data/input folder):
* all_segments_only_Jewish_survivors_generic_terms_deleted_below_25_replaced_for_parent_node.csv

### Infer further biodata:

This scripts infers further bio information about interviewees from the segment data: interviewee's arrival year to and leaving year from the Auschwitz  complex, his or her length of stay in the Auschwitz complex, type of force labour he / she did when staying in Auschwitz, whether he or she was on transfer route, whether she or he was in Birkenau, type (easy,medium,hard) of forced labour she or he did:

```
python data_processing/infer_further_biodata.py
```

Input:

* complete input data as defined above

Output:

* biodata_with_inferred_fields.cvs (saved to data/input)

This ia new biodata file with new columns

### Identify Birkenau survivors:

Identifies those survivors who stayed in Birkenau beween 1943 and 1945.

```
python data_processing/infer_birkenau_survivors.py
```

Input:

* biodata_with_inferred_fields.cvs (saved to data/input)

Output:

* biodata_birkenau.csv (saved to data/input)

### Create a segment-keyword matrix:

Creates a segment-keyword matrix from the data pre-processed above; rows are the segments and columns are keywords. If a segment has been annotated with a given keyword, the script sets the corresponding column to one, otherwise it remains 0. This matrix is therefore a binary matrix. The scripts also creates a keyword index and a segment index in a csv file. 
```
python data_processing/create_segment_keyword_matrix.py
```


Input:
* all_segments_only_Jewish_survivors_generic_terms_deleted_below_25_replaced_for_parent_node.csv

Output:
* data/output/segment_keyword_matrix/segment_keyword_matrix.txt
* data/output/segment_keyword_matrix/feature_index.csv
* data/output/segment_keyword_matrix/document_index.csv

## Baisc Statistical analysis of the data set

This workflow makes a basic descriptive statistical analysis of the biodata and the segment data (entire data set); results of this is written to data/output/report_statistical_analaysis folder. The output is plots (in the plots folder), tables (in the tables folder), and a written report (report.txt)

Run the following code from the main project folder (use python3):
```
python statistical_analysis/make_statistical_analysis.py
```

Input data:

* 'data/input/Auschwitz_segments_03112020_1.csv'
* 'data/input/Auschwitz_segments_03112020_2.csv'
* 'data/input/biodata.xlsx'

Output data: see the output folder

## Chi2 significance test and strength of association

This workflow applies chi2test of significance and odds ratio analysis between two categorical variables (CountryOfOrigin and Gender) and index terms for survivors who stayed in Birkenau between 1943 and 1945:

Input data:

* all_segments_only_Jewish_survivors_generic_terms_deleted_below_25_replaced_for_parent_node.csv (saved to data/input)
* biodata_birkenau.csv (saved to data/input)

```
python statistical_analysis/make_chi_2_test.py Gender
```

Output data:
* data/output/statistical_analysis/chi_test_filtered_gender_with_strenght_of_assoc.csv
* data/output/statistical_analysis/plots/Gender
* data/output/statistical_analysis/plots/F.html'
* data/output/statistical_analysis/plots/M.html

```
python statistical_analysis/make_chi_2_test.py CountryOfBirth
```

Output data:
* data/output/statistical_analysis/chi_test_filtered_country_of_birth_with_strenght_of_assoc.csv
* data/output/statistical_analysis/plots/{CountryName}.html
* data/output/statistical_analysis/plots/CountryOfOrigin/'


## Markov Chain analysis of the data set:

```
python markov_modelling/train_markov_model.py
```

First, partitions the segment-keyword matrix into subsets based on the metadata. For instance, it creates a segment keyword matrix that contains segments only by women survivors of Birkenau (the new document index is saved). Next, it finds all unique topic word combinations (named higher level topic in the paper), which will be the states in the first Markov State model; it creates a null count matrix with the topic combinations as rows and columns. After this it creates a trajectory list, i.e. list of 'higher level topics' that follow each other. It iterates through the reshaped segment-keyword matrix, detects the corresponding topic combination, and appends it to the trajectory list. Next, each pair of topic combinations is identified, and the previously created null count matrix is updated accordingly. Following this step, the count matrix is transformed into a transition matrix, again higher level topics as rows and columns. With a technique of fuzzy markov chain, the transition matrix with higher level topics is transformed into a new transiton matrix with the original topic words. Finally, this is used to train the final Markov State Model. It also calculates the stationary probabilities of different topic words.

See the metadata field in the Appendix here




Input data:
* data/output/segment_keyword_matrix/segment_keyword_matrix.txt
* data/output/segment_keyword_matrix/feature_index.csv
* data/output/segment_keyword_matrix/document_index.csv
* data/input/biodata_birkenau.csv

Output data:
* data/output/markov_modelling/{metadata_field_name}/document_index.csv
* data/output/markov_modelling/{metadata_field_name}/pyemma_model
* data/output/markov_modelling/{metadata_field_name}/stationary_probs.csv
* data/output/markov_modelling/{metadata_field_name}/transition_matrix.np



This reshaped segment-keyword matrix is viewed as one complete trajectory. 


### Calculate mean passage time for every metadat fields (i.e women, men, work, not_work)

```
python markov_modelling/calculate_mean_passage_time.py
```

Calculates mean passage time between each state (i.e topic) that are in the list of the first 100 most probable states (the probability of each state is its stationary probability) in the complete data and saves the result in a csv file.

Input:
* data/output/markov_modelling/complete/stationary_probs.csv
* data/output/markov_modelling/{metadata_field_name}/pyemma_model
* data/output/segment_keyword_matrix/feature_index.csv

Output:
* data/output/markov_modelling/{metadata_field_name}/mean_passage.csv

### Utility functions for data analysis

1. Print stationary probability of a given topic in a given sub-data

```
python markov_modelling/compare_stationary_probs.py --metadata_fields work_w --keywords 'camp food sharing'
```

This prints the stationary probability of 'camp food sharing' in the work_w subdata, i.e the stationary probability of 'camp food sharing' in testimonies of women who worked (work_w).

2. Print trajectories between topics:
```
python markov_modelling/print_trajectory.py --metadata_field complete_m --source 'camp social relations' --target 'food sharing'
```

This prints the possible trajectories between 'camp social relations' and 'food sharing', including the flux of a given trajectory, in testimonies of all men (complete_m).

3. Get closest topic to a given topic through the mean passage time

```
python markov_modelling/get_mean_passage_time.py --metadata_field complete --keywords 'camp food sharing'
```

Prints the closest topics to 'camp food sharing' in all testimonies (complete). Proximity defined through the mean passage time.

# Appendix

## Metadata fields:

complete: all Birkenau testimonies
complete_m: Birkenau testimonies of men
complete_w: Birkenau testimonies of women
easy_w: Birkenau testimonies of women who did easy forced labour
easy_m: Birkenau testimonies of men who did easy forced labour
medium_m: Birkenau testimonies of men who did medium hard forced labour
medium_w: Birkenau testimonies of women who did medium hard forced labour
hard_m: Birkenau testimonies of men who did  hard forced labour
hard_w: Birkenau testimonies of women who did  hard forced labo
notwork: Birkenau testimonies of those who did not work
notwork_m: Birkenau testimonies of men who did not work
notwork_w: Birkenau testimonies of women who did not work
work: Birkenau testimonies of those who worked
work_m: Birkenau testimonies of those who worked
work_w: Birkenau testimonies of those women who worked
{country}: Birkenau testimonies of victims from this country
{country_w}: Birkenau testimonies of women from this country
{country_m}: Birkenau testimonies of men from this country


