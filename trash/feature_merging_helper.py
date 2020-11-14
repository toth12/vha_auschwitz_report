import pandas as pd 
import pdb

input_file_1 = "feature_merging_original.csv"
input_file_2 = "feature_merging_with_cover_terms.csv"


input_original = pd.read_csv(input_file_1,encoding='utf-8')
input_original['total'] = input_original['count_w'] +input_original['count_m']
input_original = input_original[input_original.total>0]
input_with_cover_terms = pd.read_csv(input_file_2,encoding='utf-8')
input_original['cover_term']=''

for row in input_original.iterrows():
    num = row[1]['Unnamed: 0']
    if input_with_cover_terms[input_with_cover_terms['Unnamed: 0']==num].empty:
        input_with_cover_terms = input_with_cover_terms.append(row[1])
input_with_cover_terms.to_csv('feature_merging_with_cover_terms_2.csv')
pdb.set_trace()