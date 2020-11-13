import pandas as pd 
import constants
from anytree.importer import DictImporter
from anytree import search
import codecs
import pdb
import json

def find_nodes_by_name_pattern(name_pattern):
    node_ids = []
    
    nodes = search.findall(root,filter_=lambda node: name_pattern in node.name)
    if len(nodes) ==0:
        return ''
    else:
        for node in nodes:
                node_ids.append(node.id)
                descendants = node.descendants
                for descendant in descendants:
                    node_ids.append(descendant.id)
        node_ids = set(node_ids)
        node_ids = [i for i in node_ids]
        return node_ids



if __name__ == '__main__':
  
    #Load the feature map
    feature_cover_term_map= pd.read_csv('feature_map.csv')


    #Load the term hiearchy 
    input_directory = constants.input_data
    input_files_term_hierarchy = constants.input_files_term_hierarchy

    with codecs.open(input_directory+input_files_term_hierarchy,encoding = "utf-8-sig") as json_file:
            treedict = json.load(json_file)



    # import the term hieararchy
    importer = DictImporter()
    # create a tree object
    root = importer.import_(treedict)

    for label in feature_cover_term_map['KeywordLabel'].to_list():
        if (label == "hiding valuables") or (label=="mass execution coverups"):
            continue
        try:
            node = find_nodes_by_name_pattern(label)
        except:
            pdb.set_trace()
        if (len(node)<1):
            print (label)
            pdb.set_trace()