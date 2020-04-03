import json
import constants
import pdb
import codecs
import jsontree as jtree
from anytree.importer import DictImporter
from anytree import RenderTree
importer = DictImporter()
from anytree import search



input_directory = constants.input_data
input_files_term_hierarchy = constants.input_files_term_hierarchy 
with codecs.open(input_directory+input_files_term_hierarchy,encoding = "utf-8-sig") as json_file:
    treedict = json.load(json_file)

# ,\n^\s*"children": null


root = importer.import_(treedict)


conditions_container_nodes = ['camp living conditions','housing conditions','environmental conditions']
clothing_container_nodes =['clothing']
def find_container_subnodes(container_node_names):
    node_ids = []
    for element in conditions_container_nodes:
        container_node = search.findall_by_attr(root,element,name='name')
        if len(container_node) ==0:
            print (element)
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
    for node in nodes:
            node_ids.append(node.id)
            descendants = node.descendants
            for descendant in descendants:
                node_ids.append(descendant.id)
    node_ids = set(node_ids)
    node_ids = [i for i in node_ids]
    return node_ids

conditions = find_container_subnodes(conditions_container_nodes)
clothing = find_container_subnodes(clothing_container_nodes)
shoes = find_nodes_by_name_pattern('shoes')
loved_ones = find_nodes_by_name_pattern('loved ones')
latrines = find_nodes_by_name_pattern('latrines')
pdb.set_trace()

#7. Wartime family interactions + family members
# 10. Social relations + friendship + food sharing
#11. Barter + covert economic activities
#22. brutal treatment + executions and killings + violent attacks -> Violence 



camp_living_conditions = search.findall_by_attr(root,'camp living conditions',name='name')[0].descendants
housing_conditions = search.findall_by_attr(root,'housing conditions',name='name')[0].descendants
environment_conditions = search.findall_by_attr(root,'environment conditions',name='name')[0].descendants


clothing = search.findall_by_attr(root,'clothing',name='name')[0].descendants
loved_ones = search.findall(root,filter_=lambda node: 'loved ones' in node.name)

pdb.set_trace()

name = next(iter(treedict.keys()))

edges = []
def get_edges(treedict, parent=None):
    name = next(iter(treedict.keys()))
    if parent is not None:
        edges.append((parent, name))
    for item in treedict["children"]:
        pdb.set_trace
        print(item)
        if isinstance(item, dict):
            get_edges(item, parent=name)
        else:
            edges.append((name, item))

edges = get_edges(treedict)

