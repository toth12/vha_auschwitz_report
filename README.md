# VHA Auschwitz Reports

This repository contains python code to investigate testimonies by survivors of the Auschwitz-Birkenau concentration camp. Testimonies that these workflows analyze were provided by the USC Shoah Foundation and they are preserved in the Visual History Archive of the Foundation.

# Description of research project:

The USC Shoah Foundation’s Visual History Archive preserves 55.000 interviews with Holocaust survivors. These interviews together document victims' experience in Nazi death camps. However, with 13 years of cumulative interview time, it is highly challenging to study the ensemble of all interviews; conceptually, it is specifically challenging to study the thousands of human narratives as dynamic processes unfolding over time. In this study we elaborated a computational approach and a small scale conceptual framework to study a highly important subset of the 55.000 interviews: 6628 testimonies by Jewish survivors of the Auschwitz-Birkenau death camp. To represent the ensemble of all possible topics about which Auschwitz-Birkenau survivors talk we applied the concept of state space. We used the Markov State Model to model this state space as a dynamical system. The model is used to learn all possible sequences of topics realized in the 6628 Auschwitz testimonies. The Markov State Model along with the Transition Path Theory allowed us to compare the way women and men remember about their time in the camp. We found that acts of solidarity and social bonds are ones of the most important topics in their narratives. However, we found that women are much more likely to recall acts of solidarity and social relations. Furthermore, we found that men and women remember about these topics in different contexts. Our results indicate that men and women behave very differently in such an extreme environment as Auschwitz-Birkenau. Our results suggest that not only were women more likely to recall solidarity and social relations in their testimonies but they were also more likely to experience them in the camp. 

The result of this research was published in:

Toth et al., Studying Large-Scale Behavioral Differences in Auschwitz-Birkenau with Simulation of Gendered Narratives, Digital Humanities Quarterly 16:3, http://www.digitalhumanities.org/dhq/vol/16/3/000622/000622.html

This repository makes our findings reproducable. Code was developed in the Python environment described below. Reproducability in any other environment is not guaranteed. The time to run the code in this repository depends on the computational resources available; in some cases it can take an entire day. 


## Data set

To run this code you need to download the base dataset from here:

Toth, Gabor Mihaly, 2021, "Replication Data for the VHA Auschwitz Report Project", https://doi.org/10.7910/DVN/JFH2BJ, Harvard Dataverse, V2

The dataset consists of three elements:
1. List of segment-keyword matrices: segment_keyword_matrix_original.npy

Each testimony is represented as a segment - keyword matrix; rows correspond to segments and columns correspond to keywords or topic words

2. The column index of the matrices above: feature_index_original.csv

3. Metadata partitions: metadata_partitions.json

Interviews are partitioned in terms of different biodata infos of interviewees; each partition is a key in the json file and the value attached to the key is a list of index positions. Each index position corresponds to an element in the list of segment - keyword matrices, i.e.to an interview represented as a segment-keyword matrix.


## Getting Started


1. Git clone this library

```
git clone https://github.com/toth12/vha_auschwitz_report.git
```

2. Install python requirements with conda (code compatible with Python 3 and tested with Python 3.7.2 on Mac OS Sierra 10.12.6 and on Linux Ubuntu 16.4):


First, create a virtual environment:
```
conda create -n au_env python=3.7.2
```

Second, activate the environment:

```
conda activate au_env
```

Third, install libraries needed:

```
pip install -r requirements.txt
```

Fourth, install pyemma with conda:

```
conda config --add channels conda-forge
conda install -c conda-forge pyemma=2.5.9
```

Fifth, install wordcloud with conda:

```
conda install -c conda-forge wordcloud==1.8.1
```

3. Add your path to the python path

First, get your current path:

```
pwd
```

Second, add your path:

```
conda develop yourpath
```

4. Create the data/input directory from the project folder
```
mkdir -p data/input/
```

5. Download the dataset from the following link, unzip the data files and copy them to the data/input folder

## Run the code

You can run all research code with one bash script:

```
bash run.sh
```

Be patient, running this code can take several hours

## Data output

In the folder data/output/ you will find the results. 

In data/output/markov_modelling/{metadata field} you will find the results of markov training:

- stationary probability of topics: "stationary_probs.csv"
- markov state model: "pyemma_model"
- bayesian markov state model: "pyemma_model_bayes"
- results of implied time scale tests: "implied_time_scale.png" and "implied_time_scale_bayes.png"
- index of activate states in the markov model: "state_index.csv"

In data/output/markov_modelling/bootstrap{metadata field_metadata field} you will find the results of bootstrapping. This aims to uncover if the difference between two groups (represented through metadata partitioning), for instance men and women, in terms of the same topics' stationary probability is statististically significant. For each topic you will find a plot that shows the results of bootstrapping. 

In data/output/reports_statistical_analysis you find the results of odds ratio analysis and Fisher test. These compare the number of times a given keyword occurs in testimonies of one group and another group (again men and women). They tell whether the difference is statistically significant and reveal whether a keyword correlates positively or negatively with the two groups. In the csv file strength_of_association_odds_ratio_{group 1}_{group 2}.csv each row describes the Fisher test and the result of odds ratio analysis for a given keyword.

## Data analysis

In the notebooks directory you will find jupyther notebooks in which data analysis is accomplished; here you will also find data visualizations. To run the notebooks you need to have jupyther installed on your machine. 


```
conda install -c conda-forge jupyterlab
```


After installation:



```
cd notebooks
```

Create symlink to the output folder:

```
ln -s ../data/output output
```

Run the notebooks:

```
jupyter notebook
```

Each notebook analyzes different aspects of the dataset.

# Appendix

## Metadata fields:

* complete: all Birkenau testimonies
* complete_m: Birkenau testimonies of men
* complete_w: Birkenau testimonies of women
* easy_w: Birkenau testimonies of women who did easy forced labour
* easy_m: Birkenau testimonies of men who did easy forced labour
* medium_m: Birkenau testimonies of men who did medium hard forced labour
* medium_w: Birkenau testimonies of women who did medium hard forced labour
* medium_w: Birkenau testimonies of women who did medium hard forced labour
* medium_w: Birkenau testimonies of women who did medium hard forced labour
* hard_m: Birkenau testimonies of men who did  hard forced labour
* hard_w: Birkenau testimonies of women who did  hard forced labo
* notwork: Birkenau testimonies of those who did not work
* notwork_m: Birkenau testimonies of men who did not work
* notwork_w: Birkenau testimonies of women who did not work
* work: Birkenau testimonies of those who worked
* work_m: Birkenau testimonies of those who worked
* work_w: Birkenau testimonies of those women who worked
* {country}: Birkenau testimonies of victims from this country
* {country_w}: Birkenau testimonies of women from this country
* {country_m}: Birkenau testimonies of men from this country


