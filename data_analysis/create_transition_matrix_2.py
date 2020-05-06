import pandas as pd 
import pdb
from itertools import islice
import numpy as np
from pyemma import msm,plots
import msmtools
from msmtools.flux import tpt,ReactiveFlux
from pyemma import plots as mplt

from scipy import sparse

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
document_topic_sequences  = data.groupby('interview_code')['topic'].apply(list)

transitions = []
for element in document_topic_sequences:
    if "unknown_topic" not in element: 
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


topic_list_with_labels=[]
for element in topic_list:
    topic_n = element.split('_')[1:]
    labels = '_'.join([topic_labels[int(l)] for l in topic_n])
    topic_list_with_labels.append(labels)






transition_matrix_scaled = (transition_matrix.T/transition_matrix.sum(axis=1)).T


#transition_matrix_scaled  = sparse.csr_matrix(transition_matrix_scaled)

mm = msm.markov_model(transition_matrix_scaled)

#print the stationary distribution

for i,element in enumerate(mm.pi.argsort()[::-1]):
    print (i)
    print (topic_list[element])
    print ('\n')
    if i ==50:
        continue

#topic_2 -> 139 25
#topic_8 -> 277 43


A = [topic_list.index('topic_2')]
B = [topic_list.index('topic_8_15')]
tpt = msm.tpt(mm, A, B)

nCut = 1
(bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.3)
cumflux = 0

all_stations = [element for item in bestpaths for element in item]

all_stations.sort()

all_stations=set(all_stations)

all_connections = np.zeros((len(all_stations),len(all_stations)))



print("Path flux\t\t%path\t%of total\tpath")
for i in range(len(bestpaths)):
    cumflux += bestpathfluxes[i]

    print(bestpathfluxes[i],'\t','%3.1f'%(100.0*bestpathfluxes[i]/tpt.total_flux),'%\t','%3.1f'%(100.0*cumflux/tpt.total_flux),'%\t\t',bestpaths[i])
    
    for element in bestpaths[i]:
        print (topic_list_with_labels[element])
    #get the path labels



    #inbetween= bestpaths[i][nCut:len(bestpaths[i])-nCut]
pdb.set_trace()






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



pdb.set_trace()


