import json
import constants
import pdb
import codecs
import jsontree as jtree
from anytree.importer import DictImporter
from anytree import RenderTree
importer = DictImporter()
from anytree import search
import csv
import pandas as pd
from ast import literal_eval








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
    return int(node_id) in generic_ids

def check_if_in_term_hierarchy(node_id):

    if int(node_id) in all_node_ids:
        return True
    else:
        return False

def has_multiple_parents(node_id):

    if int(node_id) in double_nodes:
        return True
    else:
        return False

def identify_parents(node_id):
    res = search.findall_by_attr(root,int(node_id),name="id")
    result_ids = []
    result_labels = []
    for re in res:
        result_ids.append(re.parent.id)
        result_labels.append(re.parent.name)
    return result_ids,result_labels




if __name__ == '__main__':
    input_directory = constants.input_data
    output_directory = constants.output_data
    input_files_term_hierarchy = constants.input_files_term_hierarchy
    node_filter = output_directory + 'filtered_nodes/'+'node_filter_1_input.txt' 
    node_filter_output = output_directory + 'filtered_nodes/'+'node_filter_1_output.json' 
    with codecs.open(input_directory+input_files_term_hierarchy,encoding = "utf-8-sig") as json_file:
        treedict = json.load(json_file)

    input_files = constants.input_files_segments
    input_files = [input_directory+i for i in input_files]

    # preprocess data with this pattern ,\n^\s*"children": null

    # create a tree object
    root = importer.import_(treedict)

    # Read the segments data
    csv_data = []
    for el in input_files:

        f = codecs.open(el,"rb","utf-8")
        csvread = csv.reader(f,delimiter=',')
        csv_data_temp = list(csvread)
        columns = csv_data_temp[0]
        #Drop the first line as that is the column
        del csv_data_temp[0:1]
        csv_data.extend(csv_data_temp)



    columns[0] = "IntCode"
    df = pd.DataFrame(csv_data,columns=columns)

    # Read the biodata

    # Get the bio data
    bio_data = constants.input_files_biodata
    df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']

    # Get the IntCode of Jewish survivors
    IntCode = df_biodata[df_biodata['ExperienceGroup']=='Jewish Survivor']['IntCode'].to_list()
    IntCode = [str(el) for el in IntCode]

    # Leave only Jewish survivors
    df = df[df['IntCode'].isin(IntCode)]

    df_biodata = df_biodata[df_biodata['IntCode'].isin(IntCode)]


    # Count how many times a keyword is used

    df_keywords = df.drop_duplicates('KeywordID')[['KeywordID','KeywordLabel','DateKeywordCreated']]


    df_keyword_counts = df.groupby(['KeywordID'])['KeywordID'].agg('count')
    count = df_keyword_counts.to_frame(name="TotalNumberUsed").reset_index()
    df_keywords = df_keywords.merge(count,how='left',on="KeywordID")

    # Calculate how many interviewee uses a keyword
    number_of_interviewee_using = df.groupby(['KeywordID'])['IntCode'].unique().map(lambda x: len(x))
    number_of_interviewee_using = number_of_interviewee_using.to_frame(name="TotalNumberIntervieweeUsing").reset_index()
    df_keywords = df_keywords.merge(number_of_interviewee_using,how='left',on="KeywordID")

    '''

    #search.findall_by_attr(root,7601,name="id")[0].children
    #search.findall_by_attr(root,7882,name="id")[0].is_leaf
    
    time_and_place_nodes = search.findall_by_attr(root,78882,name="id")[0].descendants
    time_and_place_node_ids = [element.id for element in time_and_place_nodes]

    world_history_nodes = search.findall_by_attr(root,43485,name="id")[0].descendants
    world_history_node_ids = [element.id for element in world_history_nodes]
    generic_ids = time_and_place_node_ids+world_history_node_ids

    all_nodes = search.findall_by_attr(root,-1,name="id")[0].descendants
    all_node_ids = [element.id for element in all_nodes]

    leaf_nodes = search.findall_by_attr(root,-1,name="id")[0].leaves
    leaf_node_ids = [element.id for element in leaf_nodes]

    #double_nodes = set([x for x in leaf_node_ids if leaf_node_ids.count(x) > 1])
    double_nodes = []


    df_keywords["is_generic"] = df_keywords.KeywordID.apply(check_if_place_and_time)
    df_keywords = df_keywords[df_keywords["is_generic"]==False]

    df_keywords['is_in_term_hierarchy'] = df_keywords.KeywordID.apply(check_if_in_term_hierarchy)
    df_keywords = df_keywords[df_keywords["is_in_term_hierarchy"]==True]

    
    df_keywords['is_leaf'] = df_keywords.KeywordID.apply(check_if_node_leaf)
    
    #df_keywords['number_of_parents'] = df_keywords.KeywordID.apply(has_multiple_parents)
    
    df_leafes = df_keywords[(df_keywords.is_leaf==True)]
    df_parents = df_keywords[(df_keywords.is_leaf==False)]

    df_leafes['parent_ids'], df_leafes['parent_labels'] = zip(*df_leafes.KeywordID.apply(identify_parents))
    '''

    df_parents =pd.read_csv('parents.csv')
    df_leafes =pd.read_csv('leafes.csv')
    transfer_deportation_parent_ids = [15474,15832,15836,15450,15449,15834,15835,26240,15475,15836,15833,26184,26240]
    df_leafes_keep_unchanged = df_leafes[~(((df_leafes.KeywordLabel.str.contains('transfer to'))|(df_leafes.KeywordLabel.str.contains('transfer from'))|(df_leafes.KeywordLabel.str.contains('deportation to'))|(df_leafes.KeywordLabel.str.contains('deportation from')))|(df_leafes.TotalNumberIntervieweeUsing<25))]
    
    df_leafes_keep_unchanged= df_leafes_keep_unchanged.assign(parent_labels=0)
    df_leafes_keep_unchanged = df_leafes_keep_unchanged.assign(parent_ids=0)
    

    df_leafes_to_change = df_leafes[(((df_leafes.KeywordLabel.str.contains('transfer to'))|(df_leafes.KeywordLabel.str.contains('transfer from'))|(df_leafes.KeywordLabel.str.contains('deportation to'))|(df_leafes.KeywordLabel.str.contains('deportation from')))|(df_leafes.TotalNumberIntervieweeUsing<25))]
    df_leafes_to_changed = pd.DataFrame(columns=df_leafes_to_change.columns)



    for row in df_leafes_to_change.iterrows():
        parents = row[1]['parent_labels']
        parent_ids = literal_eval(row[1]['parent_ids'])
        parents = literal_eval(parents)
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
    
    keyword_ids_to_keep = df_parents.KeywordID.unique().tolist() + df_leafes_keep_unchanged.KeywordID.unique().tolist()+df_leafes_to_changed.KeywordID.unique().tolist()
    keyword_ids_to_keep = [str(b) for b in keyword_ids_to_keep ]
    
    new_df = df[df.KeywordID.isin(keyword_ids_to_keep)]

    new_df['KeywordID']=pd.to_numeric(new_df['KeywordID'])
    temporary_df = new_df[~new_df.KeywordID.isin(df_leafes_to_changed.KeywordID.values)]
    new_df = new_df[new_df.KeywordID.isin(df_leafes_to_changed.KeywordID.values)]

    df_leafes_to_change_parent_labels = df_leafes_to_changed.groupby(['KeywordID'])['parent_labels'].apply(lambda group_series: group_series.tolist()).reset_index()
    df_leafes_to_change_parent_ids = df_leafes_to_changed.groupby(['KeywordID'])['parent_ids'].apply(lambda group_series: group_series.tolist()).reset_index()
    df_leafes_to_change_final =  df_leafes_to_change_parent_ids.merge(df_leafes_to_change_parent_labels)
    
    new_df = pd.merge(new_df,df_leafes_to_change_final,how='left')
    updated_new_df = pd.DataFrame(columns=new_df.columns)
    


   
    int_codes=new_df.IntCode.unique().tolist()
    for f,int_code in enumerate(int_codes):
        print (f)
        print (len(int_codes))
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




    del updated_new_df['parent_ids']
    del updated_new_df['parent_labels']

    final_df = updated_new_df.append(temporary_df)
    final_df = final_df.sort_values(by=['IntCode','SegmentNumber'])

    final_df.to_csv('reorganized_segments.csv')


    pdb.set_trace()
    #df_leafes[['parent_ids','parent_labels']] = df_leafes.KeywordID.apply(identify_parents)
    df_keywords[(df_keywords.is_leaf==False)&(df_keywords.TotalNumberIntervieweeUsing<25)]
    df_keywords[df_keywords["is_leaf"]==True]
    #df_leafes[['parent_labels','KeywordLabel']]
    #15474    transfer from camps
    #15832      transfer to camps
    # 15836  transfer from prisons
    #15450    deportation from ghettos
    # 15449     deportation from cities
    #15834      deportation to camps
    #15835    deportation to ghettos
    #26240  deportation to countries




