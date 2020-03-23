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
for el in input_files[0:1]:
    f = codecs.open(el,"rb","utf-8")
    csvread = csv.reader(f,delimiter=',')
    csv_data_temp = list(csvread)
    columns = csv_data_temp[0]
    csv_data.extend(csv_data_temp[1:10])



df = pd.DataFrame(csv_data,columns=columns)

# Initiate an empty string to hold the report data

report = 'This is a statistical description of Auschwitz segments\n\n'

# Add basic information about the input data

number_data_points = len(df)

report += "Total number of datapoints: "+str(number_data_points)+".\n" 

# Analyze segment data

# Calculate the length of time a person speaks about Auschwitz

begining = df['InTimeCode'].iloc[0]
end = df['OutTimeCode'].iloc[0]

df['InTimeCode'] = pd.to_datetime(df['InTimeCode'], format = "%H:%M:%S:%f")
df['OutTimeCode'] = pd.to_datetime(df['OutTimeCode'], format = "%H:%M:%S:%f")

df['segment_lenght'] = df['OutTimeCode'] - df['InTimeCode']

pd.to_datetime('00:15:00:00', format = "%H:%M:%S:%f")

pd.to_datetime('00:15:03:00', format = "%H:%M:%S:%f").time()

"""FMT = '%H:%M:%S:%f'
(Pdb) tdelta = datetime.strptime('00:15:00:00', FMT) - datetime.strptime('00:16:00:00', FMT)
(Pdb) tdelta
datetime.timedelta(days=-1, seconds=86340)"""
