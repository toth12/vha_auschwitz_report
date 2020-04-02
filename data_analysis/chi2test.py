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

gender = df_biodata[['Gender','IntCode']]
df["IntCode"] = df.IntCode.map(lambda x: int(x))
df = df.merge(gender,how='left',on="IntCode")
df = pd.concat([df, df.Gender.str.get_dummies()], axis=1)
contingency = df.groupby(['KeywordID','KeywordLabel','IntCode']).agg({'F': 'sum', 'M': 'sum'}).reset_index()

contingency['M'] = contingency['M'].apply(lambda x: 0 if x <1 else 1)
contingency['F'] = contingency['F'].apply(lambda x: 0 if x <1 else 1)
contingency = contingency.groupby(['KeywordID',"KeywordLabel"]).agg({'F': 'sum', 'M': 'sum'}).reset_index()

total_f=len(df_biodata[df_biodata['Gender']=="F"])
total_m=len(df_biodata[df_biodata['Gender']=="M"])

result = []
for element in contingency.iterrows():

    female = element[1]['F']
    male = element[1]['M']
    not_female = total_f - element[1]['F']
    not_male = total_m - element[1]['M']
    obs = np.array([[male,not_male],[female, not_female]])
    test_stat = chi2_contingency(obs)[0]
    res_female= obs[1][0]/chi2_contingency(obs)[3][1][0]
    res_male= obs[0][0]/chi2_contingency(obs)[3][0][0]
    result.append([element[1]['KeywordID'],element[1]['KeywordLabel'],res_female,res_male, test_stat])
    



df_chi = pd.DataFrame(result,columns=['KeywordID','KeywordLabel','female','male','test_stat'])
df_chi = df_chi.sort_values('test_stat',ascending=False)


#obs = np.array([[df_suicides.iloc[0]['IntCode'], df_suicides.iloc[1]['IntCode']], [df_suicides.iloc[2]['IntCode'],df_suicides.iloc[3]['IntCode']]])

#chi2_contingency(obs)
pdb.set_trace()

