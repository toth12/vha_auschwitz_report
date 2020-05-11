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

3. Create the data directory (ignored by git) by running the following bash script:

```
 
```


4. Get the following input files and copy them to data/input:

* Auschwitz_segments_03112020_1.csv
* Auschwitz_segments_03112020_2.csv
* biodata.xlsx

## Workflows available in this repo:

* Statistical analysis of the data set
* Chi2 test of the data set
* Topic modelling of the data set
* Markov Chain analysis of topic sequences in the data set

## Markov Chain analysis of topic sequences in the data set:

This workflow first transforms the segment data into a segment-index term matrix. Next it accomplishes an anchored topic modelling over the segment-index term matrix and gives a topic label to each segment. As a result, each interview is represented not only as a sequence of segments, but as a sequence of topics. As a last step, the workflow creates a transition matrix from the topic sequences (each topic is a state in the transition matrix), and trains a Markov model from the transition matrix.

### Create a three segments - index terms matrix:

Run the following code (takes a few minutes) from the main project folder (use python3):
```
python3 feature_engineering/create_grouped_segment_keyword_matrix.py
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
python3 data_analysis/add_topic_label_to_segments.py
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
python3 create_transition_matrix_and_train_markov_chain.py
```


Input data:
* 'data/output/topic_sequencing/segment_topics.csv'
* 'data/input/biodata.xlsx'

This script contains the following key steps:
* creates a transition matrix between topics for men and women but ignores transition from unknown_topic or to unknown_topic
* removes those topics to which or from which no transition takes place
* prints the best path between two key topics: selection and camp_liquidation / transfer




### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
