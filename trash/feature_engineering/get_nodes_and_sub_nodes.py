import json
import constants
import pdb
import codecs
import jsontree as jtree
from anytree.importer import DictImporter
from anytree import RenderTree
importer = DictImporter()
from anytree import search







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

if __name__ == '__main__':
    input_directory = constants.input_data
    output_directory = constants.output_data
    input_files_term_hierarchy = constants.input_files_term_hierarchy
    node_filter = output_directory + 'filtered_nodes/'+'node_filter_1_input.txt' 
    node_filter_output = output_directory + 'filtered_nodes/'+'node_filter_1_output.json' 
    with codecs.open(input_directory+input_files_term_hierarchy,encoding = "utf-8-sig") as json_file:
        treedict = json.load(json_file)

    # preprocess data with this pattern ,\n^\s*"children": null

    # create a tree object
    root = importer.import_(treedict)

    #open the nodes file
    main_nodes = open(node_filter).readlines()
    complete_results = []
    for node_name in main_nodes:
    # Check if a combination of nodes
        if not '+' in node_name:
            if '[' not in node_name:
                result = find_container_subnodes([node_name.strip()])
                complete_results.append({node_name.strip():result})
            else:
                result = find_nodes_by_name_pattern(node_name.strip().split('[')[0])
                complete_results.append({node_name.strip().split('[')[0]:result})
        else:
            partial_result = []
            covering_term = node_name.split('=')[0].strip()
            nodes = node_name.split('=')[1:][0].split('+')
            for element in nodes:
                if '[' not in element:
                    result = find_container_subnodes([element.strip()])
                    partial_result.extend(result)
                else:
                    result = find_nodes_by_name_pattern(element.split('[')[0])
                    partial_result.extend(result)      
            complete_results.append({covering_term:partial_result})
    
    with open(node_filter_output, 'w') as f:
        json.dump(complete_results, f)
    pdb.set_trace()



