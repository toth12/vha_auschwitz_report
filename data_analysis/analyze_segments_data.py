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
import datetime
from scipy import stats, integrate




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

#From here

df_biodata['DateOfBirth'] = pd.to_datetime(df_biodata['DateOfBirth'])
df = df_biodata[['DateOfBirth']]
df = df.sort_values('DateOfBirth').dropna()
df = df.set_index(['DateOfBirth'])
df['value']=range(0,df.shape[0])


df['tvalue'] = df.index
df['delta'] = (df['tvalue']-df['tvalue'].shift()).fillna(0)

container = []
for i,element in enumerate(df['delta'].items()):
    
    if i==0:
        container.append(element[1])

    else:
        duration = element[1]+container[-1]
        container.append(duration)

container = [element.days for element in container]
ser = pd.Series(container)

pp= ser.plot.kde()

pdb.set_trace()

plt.axes[0,1].set_xticks([0,1,2,4.5])

plt.show()
plt.savefig(output_directory+'plots/age_probability.png')

plt.clf()

pdb.set_trace()


#to here
# Make time sequence from interview dates in the biodata

df_biodata['InterviewDate'] = pd.to_datetime(df_biodata['InterviewDate'])

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

# Render a boxplot
sns.set_style("whitegrid")
ax = sns.boxplot(data=df_interview_segment_length['length_in_minutes'])
plt.savefig(output_directory+'plots/interviewee_total_segment_boxplot.png')
plt.clf()
# Render the histogram
ax = sns.distplot(df_interview_segment_length['length_in_minutes'])
plt.savefig(output_directory+'plots/interviewee_total_segment_histogram.png')
plt.clf()


# Keyword analysis



report += "Analysis of keywords:\n\n"

total_number_of_keywords = len(df['KeywordID'].unique())

report += "Total number of keywords: "+str(total_number_of_keywords)+"\n\n"

# Create a keywordframe
df_keywords = df.drop_duplicates('KeywordID')[['KeywordID','KeywordLabel','DateKeywordCreated']]




df_keywords["DateKeywordCreated"] = pd.to_datetime(df_keywords['DateKeywordCreated'],errors='coerce')
df_keywords["YearKeywordCreated"] = df_keywords['DateKeywordCreated'].map(lambda x: x.year)


# Calculate how many times a keyword is used
df_keyword_counts = df.groupby(['KeywordID'])['KeywordID'].agg('count')
count = df_keyword_counts.to_frame(name="TotalNumberUsed").reset_index()
df_keywords = df_keywords.merge(count,how='left',on="KeywordID")

# Calculate how many interviewee uses a keyword
number_of_interviewee_using = df.groupby(['KeywordID'])['IntCode'].unique().map(lambda x: len(x))
number_of_interviewee_using = number_of_interviewee_using.to_frame(name="TotalNumberIntervieweeUsing").reset_index()
df_keywords = df_keywords.merge(number_of_interviewee_using,how='left',on="KeywordID")

# Calculate the relative value of interviewee using a keyword
# I.e Find the number of interviews following the creation of a keyword, and use it 
# to rescale the number of a keyword is used

# Create a histogram showing the number
year_number_of_index_terms = df_keywords.groupby('YearKeywordCreated').count()['KeywordID'].to_frame(name="Number_of_terms_introduced").reset_index()
year_number_of_index_terms = year_number_of_index_terms.astype(int)
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x="YearKeywordCreated", y="Number_of_terms_introduced", data=year_number_of_index_terms)
plt.savefig(output_directory+'plots/year_number_of_index_terms_histogram.png')
plt.clf()



# Render the distribution of index term per interview
ax = sns.distplot(df_keywords['TotalNumberIntervieweeUsing'])
plt.savefig(output_directory+'plots/keyword_interviewee_histogram_total.png')
plt.clf()



# Render the distribution of index term per interview
ax = sns.distplot(df_keywords[df_keywords['TotalNumberIntervieweeUsing']<1000]['TotalNumberIntervieweeUsing'])
plt.savefig(output_directory+'plots/keyword_interviewee_histogram_less_than_1000.png')
plt.clf()

report += "Average of the total number of times a keyword is used (at least once) in an interview: "+str(df_keywords['TotalNumberIntervieweeUsing'].mean())+"\n\n"
report += "Median of the total number of times a keyword is used (at least once) in an interview"+str(df_keywords['TotalNumberIntervieweeUsing'].median())+"\n\n"
report += "Analysis of biodata:\n\n"

# Add year of birth

index = df_biodata[df_biodata['IntCode']==55567][['DateOfBirth']].index[0]    

df_biodata.at[index,"DateOfBirth"] = datetime.datetime(1913, 10, 21, 0, 0)

df_biodata["DateOfBirth"] = pd.to_datetime(df_biodata['DateOfBirth'],errors='coerce')
df_biodata["YearOfBirth"] = df_biodata["DateOfBirth"].map(lambda x: x.year)


# Render the histogram
ax = sns.distplot(df_biodata["YearOfBirth"])
plt.savefig(output_directory+'plots/interviewee_year_of_birth_histogram.png')
plt.clf()


# Describe age groups

report += "Average year of birth: "+str(df_biodata["YearOfBirth"].mean())+"\n\n"
report += "Median of year of birth"+str(df_biodata["YearOfBirth"].median())+"\n\n"
report += "Oldest interviewee: "+str(df_biodata["YearOfBirth"].min())+"\n\n"
report += "Youngest interviewee: "+str(df_biodata["YearOfBirth"].max())+"\n\n"

report += "Analysis of biodata:\n\n"

#df_biodata[(df_biodata["DateOfBirth"]>datetime.datetime(1923,1, 1, 0, 0)) & (df_biodata["DateOfBirth"]<datetime.datetime(1924, 1, 1, 0, 0))]



# Date of birth
birth_year_count = df_biodata.groupby('YearOfBirth').count()['IntCode'].to_frame(name="Count").reset_index()
birth_year_count.to_csv(output_directory+'tables/'+'birth_year_count.csv')

# Countries of orign
country_of_orign_count = df_biodata.groupby('CountryOfBirth').count()['IntCode'].to_frame(name="Count").reset_index()
country_of_orign_count = country_of_orign_count.sort_values('Count',ascending=False)


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)

ax = sns.barplot(x="Count", y="CountryOfBirth", data=country_of_orign_count)
plt.savefig(output_directory+'plots/country_of_orign_count.png')
plt.clf()

# Interview Year

df_biodata["InterviewDate"] = pd.to_datetime(df_biodata['InterviewDate'],errors='coerce')
df_biodata["InterviewYear"] = df_biodata["InterviewDate"].map(lambda x: x.year)

df_interview_year= df_biodata["InterviewYear"]

df_interview_year = pd.to_numeric(df_interview_year, errors='coerce')
df_interview_year  = df_interview_year.dropna()
df_interview_year = df_interview_year.to_frame('count')

df_interview_year['count'] = df_interview_year['count'].map(lambda x: int(x))

df_interview_year_zoomed = df_interview_year[(df_interview_year['count']>1993)& (df_interview_year['count']<2002)]

# Render the histogram
ax = sns.distplot(df_interview_year_zoomed)
plt.savefig(output_directory+'plots/interview_year_histogram_zoomed.png')
plt.clf()

# Render the histogram, set the bins to the number of categories you have
x = sns.distplot(df_interview_year['count'],kde=False,bins=37)
plt.savefig(output_directory+'plots/interview_year_histogram.png')
plt.clf()

#df_interview_year.to_frame(name="InterviewYear").groupby('InterviewYear')['InterviewYear'].count()
interview_year_count = df_biodata.groupby('InterviewYear').count()['IntCode'].to_frame(name="Count").reset_index()
a4_dims = (11.7, 8.27)
interview_year_count['InterviewYearTruncated'] = interview_year_count['InterviewYear'].map(lambda x: str(int(x))[2:])
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(y="Count", x="InterviewYearTruncated", data=interview_year_count)
plt.savefig(output_directory+'plots/interview_year_count_bar_plot.png')
plt.clf()

pdb.set_trace()
df[df['KeywordID']=='12044']
#df_keywords[df_keywords.KeywordLabel.str.contains(pat = 'suicide')][['KeywordLabel','YearKeywordCreated','TotalNumberIntervieweeUsing']]
# df_interview_year.to_frame()['InterviewYear']
