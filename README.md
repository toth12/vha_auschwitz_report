# VHA Auschwitz Reports

This repository contains python workflows to investigate testimonies by survivors of Auschwitz-Birkenau concentration camp. Testimonies that these workflows analyze were provided by the USC Shoah Foundation.

## Data set

todo add a description of data set

## Getting Started

1. Git clone this library
2. Install python requirements

```
pip install -r requirements.txt
```

3. Create the data directory (ignored by git) with the following command:

```
mkdir -p data/{output/{chi2test,features,filtered_nodes,reports_statistical_analysis,topic_sequencing},input} 
```


4. Get the following input files and copy them to data/input:

* Auschwitz_segments_03112020_1.csv
* Auschwitz_segments_03112020_2.csv
* biodata.xlsx

## Workflows available in this repo:

* Baisc Statistical analysis of the data set
* Chi2 significance test and strength of association (odds ration) of index terms for Gender and CountryOfOrigin
* Topic modelling of the data set
* Markov Chain analysis of topic sequences in the data set


## Baisc Statistical analysis of the data set

This workflow makes a basic descriptive statistical analysis of the biodata and the segment data; results of this is written to data/output/report_statistical_analaysis folder. The output is plots (in the plots folder), tables (in the tables folder), and a written report (report.txt)

Run the following code from the main project folder (use python3):
```
python data_analysis/make_statistical_analysis.py
```

Input data:

* 'data/input/Auschwitz_segments_03112020_1.csv'
* 'data/input/Auschwitz_segments_03112020_2.csv'
* 'data/input/biodata.xlsx'

Output data: see the output folder

## Chi2 significance test and strength of association

This workflow applies chi2test of significance and odds ratio analysis between two categorical variables (CountryOfOrigin and Gender) and index terms. As a first step, index terms belonging to a given common category (for instance, suicide with sub terms camp-suicide, deportation-suicide, etc) are merged and replaced with the common category (i.e camp suicide is becoming suicide). List of index terms (ids) belonging to a given category are in data/output/filtered_nodes/node_filter_1_output.json. As an output, the workflow produces plots and tables with results of chi2test and odds ratio analysis for every index term and every categorical variable.

Input data:

* 'data/input/Auschwitz_segments_03112020_1.csv'
* 'data/input/Auschwitz_segments_03112020_2.csv'
* 'data/input/biodata.xlsx'

Run the following code from the main project folder (use python3) to do the chi2test and odds ratio analysis for gender:

```
python data_analysis/chi2test.py Gender
```

Output data:

* 'data/output/chi2test/plots/Gender'
* 'data/output/chi2test/plots/F.html'
* 'data/output/chi2test/plots/M.html'
* 'data/output/chi2test/chi_test_filtered_gender_with_strenght_of_assoc.csv'

```
python data_analysis/chi2test.py CountryOfOrigin
```

Output data:

* 'data/output/chi2test/plots/CountryOfOrigin/'
* 'data/output/chi2test/plots/{CountryName}.html'
* 'data/output/chi2test/chi_test_filtered_country_of_birth_with_strenght_of_assoc.csv'



## Markov Chain analysis of topic sequences in the data set:

This workflow first transforms the segment data into a segment-index term matrix. Next it accomplishes an anchored topic modelling over the segment-index term matrix and gives a topic label to each segment. As a result, each interview is represented not only as a sequence of segments, but as a sequence of topics. As a last step, the workflow creates a transition matrix from the topic sequences (each topic is a state in the transition matrix), and trains a Markov model from the transition matrix.

### Create a three segments - index terms matrix:

Run the following code (takes a few minutes) from the main project folder (use python3):
```
python feature_engineering/create_grouped_segment_keyword_matrix.py
```

Input data:

* 'data/input/Auschwitz_segments_03112020_1.csv'
* 'data/input/Auschwitz_segments_03112020_2.csv'
* 'data/input/biodata.xlsx'

Output data:

* 'data/output/features/segment_index_merged.csv'
* 'data/output/features/segment_keyword_matrix_merged.txt'
* 'data/output/features/keyword_index_merged_segments.csv'



This script contains the following key steps and resulting representations:
* merges every three segments of an interview into one new segment unit
* every new segment unit has an id that is the combination of the interview id plus the original three segment ids, for instance: 2_16_17_18 is from interview 2 and is the combination of segments 16, 17, and 18
* every new segment unit is represented as the combination of index terms (only ids) attached to them, for instance 10006_47_48_49 is the list of the following index terms [10983, 12044, 14280]
* eliminates those new segment units that consist of only 1 or 2 index terms
* eliminates those interviews where no group of three segments could be identified
* eliminates those index terms that occur in less than 50 interviews in total
* creates a numpy segment - index matrix; every row represents a new segment unit, and every column represents an index term
* creates a lookup index for the segment - index matrix; segment_index_merged.csv is a panda dataframe where every row has an "updated_id", which is the new segment id (see above) of the corresponding row in the segment - index matrix; keyword_index_merged_segments.csv is a panda dataframe, every row (with keyword id and keyword label) corresponds to the columns of segment - index matrix

### Give a topic label to each group of three segments:


Run the following code from the main project folder (use python3):
```
python data_analysis/add_topic_label_to_segments.py
```

Input data:

* 'data/output/features/segment_keyword_matrix_merged.txt'
* 'data/output/features/keyword_index_merged_segments.csv'
* 'data/output/features/segment_index_merged.csv'
* 'data_analysis/topic_anchors.txt'

Output data:

* 'data/output/topic_sequencing/segment_topics.csv'

This script contains the following key steps:
* accomplishes the anchored (sometimes called seeded) topic modelling with Corex over the previously constructed segment-topic matrix; topic seeds were identified earlier and they are in data_analysis/topic_anchors.txt
* produces a segment topic matrix (each topic is one of topic seeds above);
* adds a topic label to each segment; since each segment can have multiple topics, topic labels can be also the combinations of single topics or if no topic can be identified, their topic label is "unknown topic"
* saves the topic label of each segment in a panda dataframe (segment_topics.csv)


### Create a transition matrix between each topics and train a markov chain:

Run the following code from the main project folder (use python3):
```
python data_analysis/create_transition_matrix_and_train_markov_chain.py
```


Input data:
* 'data/output/topic_sequencing/segment_topics.csv'
* 'data/input/biodata.xlsx'

This script contains the following key steps:
* creates a transition matrix between topics for men and women but ignores transition from unknown_topic or to unknown_topic
* removes those topics to which or from which no transition takes place
* prints the best path between two key topics: selection and camp_liquidation / transfer




