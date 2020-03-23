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



df = pd.DataFrame(csv_data,columns=columns)

pdb.set_trace()

