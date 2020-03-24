"""This prepares a descriptive statistical analysis of interview segments"""

"""As input, it takes the interview segmenrs, and as output it creates a report
"""

import constants
import pandas as pd
import codecs
import csv
import pdb
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest



input_directory = constants.input_data
output_directory =  constants.output_data_report_statistical_analysis
input_files = constants.input_files_segments

input_files = [input_directory+i for i in input_files]

# Read the input files into panda dataframe

csv_data = []
for el in input_files:
    f = codecs.open(el,"rb","utf-8")
    csvread = csv.reader(f,delimiter=',')
    csv_data_temp = list(csvread)
    columns = csv_data_temp[0]
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

# Initiate an empty string to hold the report data
report = 'This is a statistical description of Auschwitz segments (only Jewish survivors)\n\n'

# Add basic information about the input data

number_data_points = len(df)

report += "Total number of datapoints in input data: "+str(number_data_points)+"\n\n" 
report += "Total number of segments: "+str(len(df['SegmentID'].unique()))+"\n\n" 
report += "Total number of interviewees (Jewish Survivors): "+str(len(IntCode))+"\n\n" 

# Analyze segment data

report += "Lenght of time an interviewee discusses Auschwitz:\n"

# Calculate the length of time a person speaks about Auschwitz


# Transform InTimeCode OutTimeCode into temporal data

df['InTimeCode'] = pd.to_datetime(df['InTimeCode'], format = "%H:%M:%S:%f")



df['OutTimeCode'] = pd.to_datetime(df['OutTimeCode'], format = "%H:%M:%S:%f")

# Drop duplicated segments

df_segment_length = df.drop_duplicates('SegmentID')

# Eliminate those segments where the OutTapeNumber and the InTapenumber are not the same

not_equal=len(df_segment_length[df_segment_length['OutTapenumber']!=df_segment_length['InTapenumber']])

text = "It was not possible to measure segment lenght because the OutTapenumber and Intapnumber are not the same in case of the following number of segents: "+str(not_equal)

report += text + "\n"
df_segment_length = df_segment_length[df_segment_length['OutTapenumber']==df_segment_length['InTapenumber']]



# Calculate the segment length (not the ones above)
df_segment_length['segment_lenght'] = df_segment_length['OutTimeCode'] - df_segment_length['InTimeCode']


# Calculate how much time an interview speaks about Auschwitzy
df_interview_segment_length = df_segment_length.groupby(['IntCode'])['segment_lenght'].agg('sum')

df_interview_segment_length = pd.DataFrame({'IntCode':df_interview_segment_length.index, 'length':df_interview_segment_length.values})

df_interview_segment_length = df_interview_segment_length.sort_values('length')

# Calculate in minutes

time = pd.DatetimeIndex(df_interview_segment_length['length'])
df_interview_segment_length['length_in_minutes']=time.hour * 60 + time.minute


# Write out the result to a table
df_interview_segment_length.to_csv(output_directory+'tables/'+'interviewee_total_segment_lenght.csv')
report += "Average length: "+str(df_interview_segment_length['length_in_minutes'].mean())+"\n"
report += "Median length: "+str(df_interview_segment_length['length_in_minutes'].median())+"\n"
report += "Maximum length: "+str(df_interview_segment_length['length_in_minutes'].max())+"\n"



# Remove outliers
length_in_minutes = np.array(df_interview_segment_length['length_in_minutes'].values.tolist())

# Reshape it to two dimensional array
length_in_minutes = np.reshape(length_in_minutes, (-1, 1))


clf = IsolationForest(behaviour = 'new', max_samples=12000, random_state = 1, contamination= 'auto')
preds = clf.fit_predict(length_in_minutes)


# Render a boxplot
sns.set_style("whitegrid")
ax = sns.boxplot(data=df_interview_segment_length['length_in_minutes'])
plt.savefig(output_directory+'plots/interviewee_total_segment_boxplot.png')
plt.clf()
# Render the histogram
ax = sns.distplot(df_interview_segment_length['length_in_minutes'])
plt.savefig(output_directory+'plots/interviewee_total_segment_histogram.png')
plt.clf()




df_segment_length[df_segment_length['\ufeffIntCode']=='10']

