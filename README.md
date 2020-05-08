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

## Topic modelling of the data set:

This workflow extracts the key topics underlying the segment data. For topic modelling, it uses the Corex topic modelling alghorithms. The workflow consists of the following sub-steps

### Create a three segments - index terms matrix:

Run the following code (takes a few minutes):
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
* creates a lookup index for the segment - index matrix; segment_index_merged.csv is a panda dataframe where every row has an "updated_id", which is the new segment id of the corresponding row in the segment - index matrix; keyword_index_merged_segments.csv is a panda dataframe, every row (with keyword id and keyword label) corresponds to the columns of segment - index matrix




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
