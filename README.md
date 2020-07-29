# VHA Auschwitz Reports

This repository contains python workflows to investigate testimonies by survivors of Auschwitz-Birkenau concentration camp. Testimonies that these workflows analyze were provided by the USC Shoah Foundation.

## Data set

See the description of the data set in data_desc.md in this repository (todo: creat data desc)

(todo:eliminate dead code)

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
(todo: update at the end)

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
(todo: reconsider this and set it up for 100 as a basic script)

Input:
* all_segments_only_Jewish_survivors_generic_terms_deleted_below_25_replaced_for_parent_node.csv

Output:
* data/output/segment_keyword_matrix/segment_keyword_matrix.txt
* data/output/segment_keyword_matrix/feature_index.csv
* data/output/segment_keyword_matrix/document_index.csv

## Baisc Statistical analysis of the data set

(todo:check if this is working and rewrite it then)

This workflow makes a basic descriptive statistical analysis of the biodata and the segment data; results of this is written to data/output/report_statistical_analaysis folder. The output is plots (in the plots folder), tables (in the tables folder), and a written report (report.txt)

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

First, partitions the segment-keyword matrix into subsets based on the metadata. For instance, it creates a segment keyword matrix that contains segments only by women survivors of Birkenau (the new document index is saved). Next, it finds all unique topic word combinations (named higher level topic in the paper), which will be the states in the first Markov State model; it creates a null count matrix with the topic combinations as rows and columns. After this it creates a trajectory list, i.e. list of 'higher level topics' that follow each other. It iterates through the reshaped segment-keyword matrix, detects the corresponding topic combination, and appends it to the trajectory list. Next, each pair of topic combinations is identified, and the previously created null count matrix is updated accordingly. Following this step, the count matrix is transformed into a transition matrix, again higher level topics as rows and columns. With a technique of fuzzy markov chain, the transition matrix with higher level topics is transformed into a new transiton matrix with the original topic words. Finally, this is used to train the final Markov State Model. Also calculates the stationary probabilities of different topic words.




Input data:
* data/output/segment_keyword_matrix/segment_keyword_matrix_100.txt
* data/output/segment_keyword_matrix/feature_index_100.csv
* data/output/segment_keyword_matrix/document_index_100.csv
* data/input/biodata_birkenau.csv

Output data:
* data/output/markov_modelling/{metadata_field_name}/document_index.csv
* data/output/markov_modelling/{metadata_field_name}/pyemma_model
* data/output/markov_modelling/{metadata_field_name}/stationary_probs.csv
* data/output/markov_modelling/{metadata_field_name}/transition_matrix.np




This reshaped segment-keyword matrix is viewed as one complete trajectory. 



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




