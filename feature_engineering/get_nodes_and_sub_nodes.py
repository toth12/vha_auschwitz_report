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

camp_living_conditions = search.findall_by_attr(root,'camp living conditions',name='name')[0].descendants
housing_conditions = search.findall_by_attr(root,'housing Conditions',name='name')[0].descendants
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

