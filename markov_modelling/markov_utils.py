import pandas as pd 
import pdb
from itertools import islice
import numpy as np
from pyemma import msm
import msmtools
from msmtools.estimation import connected_sets


def window(seq, n=2):
    "Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def cg_transition_matrix(T, chi):
    """
    Map a transition matrix T to coarse states via membership
    matrix chi. Implements Eq. 14 of
    Roeblitz & Weber, Adv Data Anal Classif (2013) 7:147–179
    DOI 10.1007/s11634-013-0134-6
    
    :params:
    T: np.ndarray; transition matrix in microstate space
    chi: np.ndarray membership matrix
    """
    assert msmtools.analysis.is_connected(T)

    pi = msmtools.analysis.stationary_distribution(T)
    D2 = np.diag(pi)
    D_c2_inv = np.diag(1/np.dot(chi.T, pi))
    return D_c2_inv @ chi.T @ D2 @ T @ chi

def transform_transition_matrix_connected(transition_matrix):
    
    connected_nodes = connected_sets(transition_matrix)[0]
    connected_matrix = np.take(transition_matrix,connected_nodes,axis=0)
    connected_matrix = np.take(connected_matrix,connected_nodes,axis=1)

    
    #removed_nodes = [element[0] for element in connected_sets(transition_matrix)[1:]]
    removed_nodes = [element for element in range(0,transition_matrix.shape[0]) if element not in connected_nodes]
    removed_nodes.sort()
    removed_nodes.reverse()
    return connected_matrix,removed_nodes

def round_to_one(m):
    indices=np.where(m.sum(axis=1)>1)
    for element in indices: 
        sums = m[element].sum()
        dif = sums-1
        x = np.nonzero(m[element])[0][0]
        m[element][x]=m[element][x]-dif

    indices=np.where(m.sum(axis=1)<1)
    for element in indices:
        sums = m[element].sum()
        dif = 1-sums
        x = np.nonzero(m[element])[0][0]
        m[element][x]=m[element][x]+dif
    return m


def train_markov_chain(transition_matrix):
    mm = msm.markov_model(transition_matrix)
    #mm = msm.MSM(transition_matrix,neig=13)
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6910584/

    # Use this to check if reversible
    # TODO: not sure if this attribute is updated internally; if so: assert
    mm.is_reversible

    

    return mm

def print_stationary_distributions(mm, topic_labels):
    #Print the stationary distributions of the top states
    results = []
    for i, element in enumerate(mm.pi.argsort()[::-1]):
        #print (i)
        #print (topic_labels[element])
        #print (mm.pi[element])
        #print ('\n')
        results.append({'topic_name':topic_labels[mm.active_set[element]],'stationary_prob':mm.pi[element]})
    return results


    # Print the eigenvalues of states

    '''for element in mm.eigenvalues().argsort()[::-1]:
        print (topic_list[element])
    '''
def calculate_mean_passage_time_between_states(mm,topic_labels):
    #Create a matrix that will hold the data

    passage_times = np.zeros(shape=(len(topic_labels),len(topic_labels)))
    df_passage_times = pd.DataFrame(passage_times)  
    for f,row in enumerate(topic_labels):

        for l,column in enumerate(topic_labels):
            try:
                df_passage_times.iloc()[f][l]=msm.tpt(mm,[f],[l]).mfpt
            except:
                df_passage_times.iloc()[f][l]=0
    column_names = {v: k for v, k in enumerate(topic_labels)}
    df_passage_times = df_passage_times.rename(columns=column_names,index=column_names)
    return df_passage_times

def calculate_flux(mm,topic_labels,source,target):
    #A=[8],B=[2,13],
    # Calculate the flux between two states camp arrival and camp liquidiation / camp transfer )
    np.set_printoptions(suppress=True) 
    A = []
    B = []
    for element in source:
        A.append(topic_labels.index(element))

    for element in target:
         B.append(topic_labels.index(element))
   
    tpt = msm.tpt(mm, A, B)

    nCut = 1
    (bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.3)
    cumflux = 0

    # Print the best path between the two states

    print("Path flux\t\t%path\t%of total\tpath")

    topic_sequences = {}
    for i in range(len(bestpaths)):
        cumflux += bestpathfluxes[i]
        flux = 100.0*bestpathfluxes[i]/tpt.total_flux
        if flux > 0:
            #print(bestpathfluxes[i],'\t','%3.1f'%(100.0*bestpathfluxes[i]/tpt.total_flux),'%\t','%3.1f'%(100.0*cumflux/tpt.total_flux),'%\t\t',bestpaths[i])
            
            topic_sequence = []
            for element in bestpaths[i]:
                #print (topic_labels[element])
                topic_sequence.append(topic_labels[element])
            topic_sequence = '-'.join(topic_sequence)
            topic_sequences[topic_sequence]=100.0*bestpathfluxes[i]/tpt.total_flux
   
    return topic_sequences
    
  

def create_dataframe_with_paths(paths_w,paths_m,filter_stat=None):
    women_topic_sequences = paths_w
    men_topic_sequences = paths_m

    pathes = list(set(list(women_topic_sequences.keys())+list(men_topic_sequences.keys())))

    result = []
    for element in pathes:
        try:
            wflux = women_topic_sequences[element]
        except:
            wflux = 0
        try:
            mflux = men_topic_sequences[element]
        except:
            mflux = 0
        result.append({'path':element,'wflux':wflux,'mflux':mflux})
    if filter_stat == None:
        return pd.DataFrame(result)
    else:
        filter = "|".join(filter_stat)
        df = pd.DataFrame(result)
        df = df[df.path.str.contains("social|aid")]
        return df

def post_process_topic_sequences(sequences,features_df):
    print ('ss')
    final_result = []
    for element in sequences:
        pdb.set_trace()

    return final_result

if __name__ == '__main__':
    pass

