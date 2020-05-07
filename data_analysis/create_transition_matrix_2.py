import pandas as pd 
import pdb
from itertools import islice
import numpy as np
from pyemma import msm,plots
import msmtools
from msmtools.flux import tpt,ReactiveFlux
from pyemma import plots as mplt
import constants

from scipy import sparse

input_directory = constants.input_data
bio_data = constants.input_files_biodata

np.set_printoptions(suppress=True)


topic_doc = open('data_analysis/topics_enumerated.txt').read()

topic_labels = [element.split('\n')[0].strip() for element in topic_doc.split('\n\n')]




def window(seq, n=2):
    "Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


data = pd.read_csv('data/output/topic_sequencing/segment_topics.csv')
data['interview_code'] = data['updated_id'].apply(lambda x: x.split('_')[0])

df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']

IntCodeM = df_biodata[(df_biodata.ExperienceGroup=='Jewish Survivor')&(df_biodata.Gender=='M')]['IntCode'].to_list()
IntCodeW = df_biodata[(df_biodata.ExperienceGroup=='Jewish Survivor')&(df_biodata.Gender=='F')]['IntCode'].to_list()

groups = [IntCodeW,IntCodeM]

for group in groups:

    #filter the data

    group = [str(element) for element in group]
    data_filtered = data[data['interview_code'].isin(group)]



    document_topic_sequences  = data_filtered.groupby('interview_code')['topic'].apply(list)

    transitions = []
    for element in document_topic_sequences:
       #if "unknown_topic" not in element: 
        transition = [i for i in window(element)]
        if len(transition)>1:
            transitions.extend(transition)

    topic_list=data.topic.unique().tolist()
    topic_list.remove('unknown_topic')
    topic_list = sorted(topic_list)



    transition_matrix = np.zeros([len(topic_list),len(topic_list)]).astype(int)

    for element in transitions:
        if 'unknown_topic' not in element:
            state1 = element[0]
            state2 = element[1]
            state1_index= topic_list.index(state1)
            state2_index= topic_list.index(state2)
            transition_matrix[state1_index,state2_index] = transition_matrix[state1_index,state2_index] + 1



    to_be_removed=np.sort(np.where(np.all(transition_matrix == 0, axis=1)==True))[0][::-1]
    for element in to_be_removed:
        transition_matrix=np.delete(transition_matrix,element,axis=1)
        transition_matrix=np.delete(transition_matrix,element,axis=0)
        del topic_list[element]


    






    transition_matrix_scaled = (transition_matrix.T/transition_matrix.sum(axis=1)).T

    # Check null columns

    null_columns = np.sort(np.where(transition_matrix_scaled.sum(0)==0)[0])[::-1]

    for null in null_columns:
        transition_matrix_scaled = np.delete(transition_matrix_scaled,null, 0)  
        transition_matrix_scaled = np.delete(transition_matrix_scaled,null, 1)
        del topic_list[null] 

    
    topic_list_with_labels=[]
    for element in topic_list:
        topic_n = element.split('_')[1:]
        labels = '_'.join([topic_labels[int(l)] for l in topic_n])
        topic_list_with_labels.append(labels)


    #transition_matrix_scaled  = sparse.csr_matrix(transition_matrix_scaled)

    mm = msm.markov_model(transition_matrix_scaled)

    #print the stationary distribution

    for i,element in enumerate(mm.pi.argsort()[::-1]):
        print (i)
        print (topic_list_with_labels[element])
        print (topic_list[element])
        print ('\n')
        if i ==10:
            break

    #topic_2 -> 139 25
    #topic_8 -> 277 43


    A = [topic_list.index('topic_2')]
    B = [topic_list.index('topic_8_14')]
    tpt = msm.tpt(mm, A, B)

    nCut = 1
    (bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.95)
    cumflux = 0

    all_stations = [element for item in bestpaths for element in item]

    all_stations.sort()

    all_stations=set(all_stations)

    all_connections = np.zeros((len(all_stations),len(all_stations)))

    social_relations_indices = [topic_list_with_labels.index(element) for element in topic_list_with_labels if "selection" == element]
    adaptation_relations_indices = [topic_list_with_labels.index(element) for element in topic_list_with_labels if "campdeath" == element]

    #intersection = [element for element in social_relations_indices if element in adaptation_relations_indices]

    #social_relations_indices.remove(intersection[0])

    #intersection = [element for element in adaptation_relations_indices if element in social_relations_indices]

    #adaptation_relations_indices.remove(intersection[0])
       

    tpt_social_relations = msm.tpt(mm, social_relations_indices, adaptation_relations_indices)

    print('socialrelations->B adaptation= ', 1.0/tpt_social_relations.rate)
    print (tpt_social_relations.rate)


    tpt_adaptation = msm.tpt(mm, adaptation_relations_indices,social_relations_indices)


    print('adaptation->socialrelations-> ', 1.0/tpt_adaptation.rate)
    print(tpt_adaptation.rate)
    

    pdb.set_trace()

    print("Path flux\t\t%path\t%of total\tpath")
    for i in range(len(bestpaths)):
        cumflux += bestpathfluxes[i]

        print(bestpathfluxes[i],'\t','%3.1f'%(100.0*bestpathfluxes[i]/tpt.total_flux),'%\t','%3.1f'%(100.0*cumflux/tpt.total_flux),'%\t\t',bestpaths[i])
        
        for element in bestpaths[i]:
            print (topic_list_with_labels[element])
        #get the path labels



        #inbetween= bestpaths[i][nCut:len(bestpaths[i])-nCut]



    '''

    print('Total TPT flux = ', tpt.total_flux)
    print('Rate from TPT flux = ', tpt.rate)
    print('A->B transition time = ', 1.0/tpt.rate)

    print('mfpt(0,4) = ', mm.mfpt(0, 4))

    Fsub = tpt.major_flux(fraction=0.1)
    print(Fsub)
    Fsubpercent = 100.0 * Fsub / tpt.total_flux
    plt = mplt.plot_network(Fsubpercent, state_sizes=tpt.stationary_distribution, arrow_label_format="%3.1f")


    pdb.set_trace()



    # Find the eigenvalues


    
    for element in mm.eigenvalues().argsort()[::-1]:
        print (topic_list[element])
    transition_matrix_scaled  = sparse.csr_matrix(transition_matrix_scaled)
    flux = tpt(transition_matrix_scaled,[1],[2])
    '''





