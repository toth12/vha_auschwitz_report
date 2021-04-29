"""Replaces those keywords that occur in less than 25 interviews with their respective parent node

As output it constructs a new segment data containing only Jewish survivors and the simplified keywords.
"""


import json
import constants
import pdb
import codecs
from anytree.importer import DictImporter
from anytree import search
import csv
import pandas as pd
from tqdm.auto import tqdm


def find_container_subnodes(container_node_names):
    node_ids = []
    for element in container_node_names:
        container_node = search.findall_by_attr(root,element,name='name')
        if len(container_node) ==0:
            pdb.set_trace()
        for node in container_node:
            node_ids.append(node.id)
            descendants = node.descendants
            for descendant in descendants:
                node_ids.append(descendant.id)
    node_ids = set(node_ids)
    node_ids = [i for i in node_ids]
    return node_ids


def find_nodes_by_name_pattern(name_pattern):
    node_ids = []
    
    nodes = search.findall(root,filter_=lambda node: name_pattern in node.name)
    if len(nodes) ==0:
        pdb.set_trace()
    for node in nodes:
            node_ids.append(node.id)
            descendants = node.descendants
            for descendant in descendants:
                node_ids.append(descendant.id)
    node_ids = set(node_ids)
    node_ids = [i for i in node_ids]
    return node_ids

def check_if_node_leaf(node_id):
    if int(node_id) in leaf_node_ids:
        return True
    else:
        return False

def check_if_place_and_time(node_id):
    return node_id in generic_ids

def check_if_in_term_hierarchy(node_id):

    if node_id in all_node_ids:
        return True
    else:
        return False

def has_multiple_parents(node_id):

    if node_id in double_nodes:
        return True
    else:
        return False

def identify_parents(node_id):
    res = search.findall_by_attr(root,node_id,name="id")
    result_ids = []
    result_labels = []
    for re in res:
        result_ids.append(re.parent.id)
        result_labels.append(re.parent.name)
    return result_ids,result_labels




if __name__ == '__main__':

    print ("Be patient running this script takes several minute (5 - 10)")
    input_directory = constants.input_data
    output_directory = input_directory
    input_files_term_hierarchy = constants.input_files_term_hierarchy
    output_file = constants.input_segments_with_simplified_keywords

    with codecs.open(input_directory+input_files_term_hierarchy,encoding = "utf-8-sig") as json_file:
        treedict = json.load(json_file)

    input_file = constants.input_files_segments_story_end_beginning_distinguished[0]

    # import the term hieararchy
    importer = DictImporter()
    # create a tree object
    root = importer.import_(treedict)

    # Read the segments data
    df = pd.read_csv(input_directory+input_file)

    # Get the bio data
    bio_data = constants.input_files_biodata
    df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']

    
    ### Cast datatypes

    ## Start with segments data

    df_numeric = ['IntCode','SegmentID','InTapenumber','OutTapenumber','KeywordID']
    df_string = ['IntervieweeName','KeywordLabel']
    df_time = ['InTimeCode','OutTimeCode']


    # Cast to numeric
    for col in df_numeric:
        df[col] = pd.to_numeric(df[col])

    # Cast to string
    for col in df_string:
        df[col] = df[col].astype('string')




    ## Type cast on biodata
    df_biodata_string = ['InterviewTitle','IntervieweeName','Gender','ExperienceGroup','CityOfBirth','CountryOfBirth','InterviewCity','InterviewCountry','InterviewLanguage','HistoricEvent','OrganizationName']
    df_biodata_numeric = ['IntCode']
    df_biodata_time = ['DateOfBirth','InterviewDate']

    # Cast to numeric
    for col in df_biodata_numeric:
        df_biodata[col] = pd.to_numeric(df_biodata[col])

    # Cast to string
    for col in df_biodata_string:
        df_biodata[col] = df_biodata[col].astype('string')

    # Cast to temporal

    for col in df_biodata_time:
        df_biodata[col] = pd.to_datetime(df_biodata[col],errors='coerce')



    # Get the IntCode of Jewish survivors
    IntCode = df_biodata[df_biodata['ExperienceGroup']=='Jewish Survivor']['IntCode'].to_list()

    # Leave only Jewish survivors
    df = df[df['IntCode'].isin(IntCode)]
    df_biodata = df_biodata[df_biodata['IntCode'].isin(IntCode)]


    # Count how many times a keyword is used and make a new dataframe that holds all keywords
    df_keywords = df.drop_duplicates('KeywordID')[['KeywordID','KeywordLabel','DateKeywordCreated']]


    df_keyword_counts = df.groupby(['KeywordID'])['KeywordID'].agg('count')
    count = df_keyword_counts.to_frame(name="TotalNumberUsed").reset_index()
    df_keywords = df_keywords.merge(count,how='left',on="KeywordID")

    # Calculate how many interviewee uses a keyword
    number_of_interviewee_using = df.groupby(['KeywordID'])['IntCode'].unique().map(lambda x: len(x))
    number_of_interviewee_using = number_of_interviewee_using.to_frame(name="TotalNumberIntervieweeUsing").reset_index()
    df_keywords = df_keywords.merge(number_of_interviewee_using,how='left',on="KeywordID")

    # Find those keywords that describe times and places
    time_and_place_nodes = search.findall_by_attr(root,78882,name="id")[0].descendants
    time_and_place_node_ids = [element.id for element in time_and_place_nodes]

    # Find those keywords that describe events of world history
    world_history_nodes = search.findall_by_attr(root,43485,name="id")[0].descendants
    world_history_node_ids = [element.id for element in world_history_nodes]

    # Make a joint list of them
    generic_ids = time_and_place_node_ids+world_history_node_ids

    # Get the node id of all nodes
    all_nodes = search.findall_by_attr(root,-1,name="id")[0].descendants
    all_node_ids = [element.id for element in all_nodes]

    # Find all leaf nodes
    leaf_nodes = search.findall_by_attr(root,-1,name="id")[0].leaves
    leaf_node_ids = [element.id for element in leaf_nodes]

    # Some nodes can have multiple parents below is a convenience method to identify them
    #double_nodes = set([x for x in leaf_node_ids if leaf_node_ids.count(x) > 1])

    #Identify generic keywords
    df_keywords["is_generic"] = df_keywords.KeywordID.apply(check_if_place_and_time)
    df_keywords = df_keywords[df_keywords["is_generic"]==False]

    # Some key are the combinations of other keywords and they are not in the term hierarchy
    df_keywords['is_in_term_hierarchy'] = df_keywords.KeywordID.apply(check_if_in_term_hierarchy)

    # Eliminate these keywords
    #df_keywords = df_keywords[df_keywords["is_in_term_hierarchy"]==True]

    # Identify if a keyword is a leaf
    df_keywords['is_leaf'] = df_keywords.KeywordID.apply(check_if_node_leaf)
    
    # Convenience method to add information about whether a keyword has multiple nodeds
    #df_keywords['number_of_parents'] = df_keywords.KeywordID.apply(has_multiple_parents)
    
    # Divide the data into parents and leafes
    df_leafes = df_keywords[(df_keywords.is_leaf==True)]
    df_parents = df_keywords[(df_keywords.is_leaf==False)]

    # Identify the parents of leaves
    df_leafes['parent_ids'], df_leafes['parent_labels'] = zip(*df_leafes.KeywordID.apply(identify_parents))
  

    #df_parents =pd.read_csv('parents.csv')
    #df_leafes =pd.read_csv('leafes.csv')

    #Divide the leaves into two groups: 
    #1. occurring in less than 25 interviews and not refering to transfer from / to or deportation from / to a given location
    #2. occuring in more than 25 interviews and refering to transfer from / to or deportation from / to a given location
    df_leafes_keep_unchanged = df_leafes[~(((df_leafes.KeywordLabel.str.contains('transfer to'))|(df_leafes.KeywordLabel.str.contains('transfer from'))|(df_leafes.KeywordLabel.str.contains('deportation to'))|(df_leafes.KeywordLabel.str.contains('deportation from')))|(df_leafes.TotalNumberIntervieweeUsing<25))]
    

    df_leafes_keep_unchanged = df_leafes_keep_unchanged.assign(parent_labels=0)
    df_leafes_keep_unchanged = df_leafes_keep_unchanged.assign(parent_ids=0)
    


    df_leafes_to_change = df_leafes[(((df_leafes.KeywordLabel.str.contains('transfer to'))|(df_leafes.KeywordLabel.str.contains('transfer from'))|(df_leafes.KeywordLabel.str.contains('deportation to'))|(df_leafes.KeywordLabel.str.contains('deportation from')))|(df_leafes.TotalNumberIntervieweeUsing<25))]
    df_leafes_to_changed = pd.DataFrame(columns=df_leafes_to_change.columns)

    # Update the group 2 with parent node name and id

    for row in tqdm(df_leafes_to_change.iterrows(), total=len(df_leafes_to_change)):
        parents = row[1]['parent_labels']
        parent_ids = row[1]['parent_ids']
        parents = parents
        if len(parents) ==1:
            row[1]['parent_labels']=parents[0]
            row[1]['parent_ids']=parent_ids[0]
            df_leafes_to_changed = df_leafes_to_changed.append(row[1])
        else:
            for i,element in enumerate(parents):
                temporary_row = row[1].copy()
                temporary_row['parent_labels']=element
                temporary_row['parent_ids'] = parent_ids[i]

                df_leafes_to_changed = df_leafes_to_changed.append(temporary_row)
    
    # Put together the three groups (parent nodes, leaves used in more than 25 interviews, leaves used in less than 25 interviews) above
    keyword_ids_to_keep = df_parents.KeywordID.unique().tolist() + df_leafes_keep_unchanged.KeywordID.unique().tolist()+df_leafes_to_changed.KeywordID.unique().tolist()
    
    # Filter the original segment data
    new_df = df[df.KeywordID.isin(keyword_ids_to_keep)]

    # In case of keyword occurring in less than 25 interviews update them with the parent node

    temporary_df = new_df[~new_df.KeywordID.isin(df_leafes_to_changed.KeywordID.values)]
    new_df = new_df[new_df.KeywordID.isin(df_leafes_to_changed.KeywordID.values)]

    # Find the parent labels
    df_leafes_to_change_parent_labels = df_leafes_to_changed.groupby(['KeywordID'])['parent_labels'].apply(lambda group_series: group_series.tolist()).reset_index()
    

    # Find the parent ids
    df_leafes_to_change_parent_ids = df_leafes_to_changed.groupby(['KeywordID'])['parent_ids'].apply(lambda group_series: group_series.tolist()).reset_index()
    df_leafes_to_change_final = df_leafes_to_change_parent_ids.merge(df_leafes_to_change_parent_labels)
    df_leafes_to_change_final.KeywordID = pd.to_numeric(df_leafes_to_change_final.KeywordID)


    new_df = pd.merge(new_df,df_leafes_to_change_final,how='left')

    # Make a dataframe that will hold the results
    updated_new_df = pd.DataFrame(columns=new_df.columns)
    

    # Loop over and include only parents nodes in case of group 1 from above
   
    int_codes=new_df.IntCode.unique().tolist()
    for f,int_code in tqdm(enumerate(int_codes), total=len(int_codes)):
        #print (f)
        #print (len(int_codes))
        temp_df = new_df[new_df.IntCode.isin([int_code])]
        for row in temp_df.iterrows():
            parents = row[1]['parent_labels']
            parent_ids = row[1]['parent_ids']
            if len(parents) == 1:
                row[1]['KeywordLabel']=parents[0]
                row[1]['KeywordID']=parent_ids[0]
                updated_new_df = updated_new_df.append(row[1])
            else:
                for i,element in enumerate(parents):
                    temporary_row = row[1].copy()
                    temporary_row['KeywordLabel']=element
                    temporary_row['KeywordID'] = parent_ids[i]
                    updated_new_df = updated_new_df.append(temporary_row)


    # Delete unneccessary fields

    del updated_new_df['parent_ids']
    del updated_new_df['parent_labels']

    final_df = updated_new_df.append(temporary_df)
    final_df = final_df.sort_values(by=['IntCode','SegmentNumber'])
    final_df.to_csv(output_directory + output_file)
