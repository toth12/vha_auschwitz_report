"""This prepares a descriptive statistical analysis of interview segments"""

"""As input, it takes the interview segmenrs, and as output it creates a report
"""

import constants
import pandas as pd
import codecs
import csv
import pdb


input_directory = constants.input_data
input_files = constants.input_files_segments

input_files = [input_directory+'/'+i for i in input_files]


# Read the input files into panda dataframe

csv_data = []
for el in input_files:
    f = codecs.open(el,"rb","utf-8")
    csvread = csv.reader(f,delimiter=',')
    csv_data_temp = list(csvread)
    columns = csv_data_temp[0]
    csv_data.extend(csv_data_temp[1:])


columns[0] = "IntCode"
df = pd.DataFrame(csv_data,columns=columns)

# Initiate an empty string to hold the report data

report = 'This is a statistical description of Auschwitz segments\n\n'

# Add basic information about the input data

number_data_points = len(df)

report += "Total number of datapoints: "+str(number_data_points)+".\n" 

# Analyze segment data

# Calculate the length of time a person speaks about Auschwitz


# Transform InTimeCode OutTimeCode into temporal data



df['InTimeCode'] = pd.to_datetime(df['InTimeCode'], format = "%H:%M:%S:%f")


df['OutTimeCode'] = pd.to_datetime(df['OutTimeCode'], format = "%H:%M:%S:%f")

# Drop duplicated segments



df_segment_length = df.drop_duplicates('SegmentID')

# Eliminate those segments where the OutTapeNumber and the InTapenumber are not the same

not_equal=len(df_segment_length[df_segment_length['OutTapenumber']!=df_segment_length['InTapenumber']])



df_segment_length = df_segment_length[df_segment_length['OutTapenumber']==df_segment_length['InTapenumber']]



# Calculate the segment length (not the ones above)
df_segment_length['segment_lenght'] = df_segment_length['OutTimeCode'] - df_segment_length['InTimeCode']


# Calculate how much time an interview speaks about Auschwitzy
df_interview_segment_length = df_segment_length.groupby(['IntCode'])['segment_lenght'].agg('sum')

df_interview_segment_length = pd.DataFrame({'IntCode':df_interview_segment_length.index, 'length':df_interview_segment_length.values})

df_interview_segment_length = df_interview_segment_length.sort_values('length')
df_interview_segment_length.to_csv('test_2.csv')

pdb.set_trace()

df_segment_length[df_segment_length['\ufeffIntCode']=='10']

df[df['SegmentID']=='198988'].to_csv('test.csv')

df[df['OutTapenumber']==df['InTapenumber']]

pd.DataFrame(df_interview_segment_length)


"""FMT = '%H:%M:%S:%f'
(Pdb) tdelta = datetime.strptime('00:15:00:00', FMT) - datetime.strptime('00:16:00:00', FMT)
(Pdb) tdelta
datetime.timedelta(days=-1, seconds=86340)"""
