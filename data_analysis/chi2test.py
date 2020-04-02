import constants
import pandas as pd
import codecs
import csv
import pdb
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import datetime
from scipy import stats, integrate
from scipy.stats import chisquare
from scipy.stats import chi2_contingency


def chi2test(df,df_biodata,category):

    df_category = df_biodata[[category,'IntCode']]
    df = df.merge(df_category,how='left',on="IntCode")

    df = pd.concat([df, df[category].str.get_dummies()], axis=1)

    agg_pipeline = {}
    for element in df[category].unique():
        if pd.isna(element):
            continue
        agg_pipeline[element]='sum'

   
    contingency = df.groupby(['KeywordID','KeywordLabel','IntCode']).agg(agg_pipeline).reset_index()

    for key in agg_pipeline:

        contingency[key] = contingency[key].apply(lambda x: 0 if x <1 else 1)
        contingency[key] = contingency[key].apply(lambda x: 0 if x <1 else 1)
    
    contingency = contingency.groupby(['KeywordID',"KeywordLabel"]).agg(agg_pipeline).reset_index()

    total = {}
    for key in agg_pipeline:
        total[key] = len(df_biodata[df_biodata[category]==key])



    result = []
    for element in contingency.iterrows():
        total_obs = []
        for key in agg_pipeline:
            number_of_obs = element[1][key]
            number_of_not_obs = total[key] - number_of_obs
            obs = np.array([[number_of_obs,number_of_not_obs]])
            total_obs.append(obs)
        total_obs = np.vstack(total_obs)
        
        if total_obs[:,0].sum() ==0:
            continue
        test_result = chi2_contingency(total_obs)

        test_stat = test_result[0]
        p_value = test_result[1]
        results_for_individual_cat = total_obs[:,0] / test_result[3][:,0]
        partial_result = [element[1]['KeywordID'],element[1]['KeywordLabel'],test_stat,p_value]
        partial_result.extend(results_for_individual_cat.tolist())
        result.append(partial_result)
        

    column_labels = [element for element in agg_pipeline.keys()]
    columns = ['KeywordID','KeywordLabel','test_stat','p_value']
    columns.extend(column_labels)
    df_chi = pd.DataFrame(result,columns=columns)
    df_chi = df_chi.sort_values('test_stat',ascending=False)
    return (df_chi)


if __name__ == '__main__':

    # Read the data


    input_directory = constants.input_data
    output_directory =  constants.output_data_report_statistical_analysis
    input_files = constants.input_files_segments

    input_files = [input_directory+i for i in input_files]

    # Read the input files into panda dataframe

    csv_data = []
    for el in input_files[0:1]:

        f = codecs.open(el,"rb","utf-8")
        csvread = csv.reader(f,delimiter=',')
        csv_data_temp = list(csvread)
        columns = csv_data_temp[0]
        #Drop the first line as that is the column
        del csv_data_temp[0:10]
        csv_data.extend(csv_data_temp)



    columns[0] = "IntCode"
    df = pd.DataFrame(csv_data,columns=columns)

    # Filter out non Jewish survivors

    # Get the bio data
    bio_data = constants.input_files_biodata
    df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']




    # Get the IntCode of Jewish survivors
    IntCode = df_biodata[df_biodata['ExperienceGroup']=='Jewish Survivor']['IntCode'].to_list()
    IntCode = [str(el) for el in IntCode]



    # Leave only Jewish survivors
    df = df[df['IntCode'].isin(IntCode)]

    df_biodata = df_biodata[df_biodata['IntCode'].isin(IntCode)]
    df["IntCode"] = df.IntCode.map(lambda x: int(x))

    result = chi2test(df,df_biodata,'Gender')

# Read data finished


    pdb.set_trace()

