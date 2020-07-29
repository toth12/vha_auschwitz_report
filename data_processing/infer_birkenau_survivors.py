"""Infers those survivors who were in Birkenau from the preprocessed biodata

Specificially, infers those survivors who were in Birkenau, arrived after 1942,
were not on transfer route and in most of their discussion they speak about Birkenau.
Input: biodata_with_inferred_fields.csv
Output: biodata_with_inferred_fields.csv

"""

import pandas as pd
import constants
import pdb

output_folder = constants.input_data
input_folder = constants.input_data
input_file = constants.input_files_biodata_with_inferred_fields
output_file = constants.input_files_biodata_birkenau

df_biodata = pd.read_csv(input_folder+input_file)
df_biodata = df_biodata.fillna(0)

df_biodata = df_biodata[((df_biodata.Birkenau_segment_percentage>0.7)&(df_biodata.earliest_year>1942)&(df_biodata.is_transfer_route==False))]

df_biodata.to_csv(output_folder+output_file)
