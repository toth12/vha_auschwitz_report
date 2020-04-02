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

suicides = df[df.KeywordLabel.str.contains('trial witnesses')][['IntCode']].IntCode.unique().tolist()

df_biodata['suicide_discussed'] = [True if str(x) in suicides else False for x in df_biodata['IntCode'] ]
df_suicides = df_biodata.groupby(["Gender",'suicide_discussed'])[['IntCode']].count().reset_index()

obs = np.array([[df_suicides.iloc[0]['IntCode'], df_suicides.iloc[1]['IntCode']], [df_suicides.iloc[2]['IntCode'],df_suicides.iloc[3]['IntCode']]])

chi2_contingency(obs)
pdb.set_trace()

