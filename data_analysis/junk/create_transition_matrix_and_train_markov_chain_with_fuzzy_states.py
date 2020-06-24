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
from sklearn import preprocessing

np.set_printoptions(suppress=True)

# Read the input data
input_directory = constants.input_data

# Read the biodata
bio_data = constants.input_files_biodata

# Read the topic labels
topic_doc = open('data_analysis/topic_anchors_Birkenau.txt').read()
topic_labels = [element.split('\n')[0].strip() for element in topic_doc.split('\n\n')]
transcripts = pd.read_csv('data_analysis/transcripts_available.csv').IntCode.to_list()

# Read the segments dataframe
data = pd.read_csv('data/output/topic_sequencing/document_index_with_topic_labels.csv')

# Eliminate those interview segments where the topic label us unknown_topic
data = data[~data.new_segment_id.isin(data[data.topic =="unknown_topic"].new_segment_id.tolist())]

# Compute the interview code of each segment (the unique id of each segment contains the interview code as prefix)
data['interview_code'] = data['new_segment_id'].apply(lambda x: x.split('_')[0])

# Read the biodata and identify the interview code of male and female survivors
df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']
IntCodeM = df_biodata[(df_biodata.ExperienceGroup=='Jewish Survivor')&(df_biodata.Gender=='M')]['IntCode'].to_list()
IntCodeW = df_biodata[(df_biodata.ExperienceGroup=='Jewish Survivor')&(df_biodata.Gender=='F')]['IntCode'].to_list()


groups = [IntCodeW,IntCodeM]


def window(seq, n=2):
    "Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# Create the transition matrix for men and women
women_topic_sequences = {}
men_topic_sequences = {}
women_topics_stationary_prob = {}
men_topics_stationary_prob = {}

for f,group in enumerate(groups):

    #Filter the segment data so that only men or women segments remain
    group = [str(element) for element in group]
    data_filtered = data[data['interview_code'].isin(group)]


    # Create a new dataframe in which every interview is a sequence of topics
    # For instance

    #interview_code
    #110006                    [topic_2, topic_1]
    document_topic_sequences  = data_filtered.groupby('interview_code')['topic'].apply(list)

    # Helper code to print topic sequenqes
    '''
    if f ==0:

        w_tpic_sequences = data_filtered.groupby('interview_code')['topic'].apply(list).to_frame(name="sequences").reset_index()
        int_codes = w_tpic_sequences.interview_code.to_list()
        transcripts_available = [True if int(item) in transcripts else False for item in int_codes]
        w_tpic_sequences['transcripts_available'] = transcripts_available
        w_tpic_sequences.to_csv('w_tpic_sequences.csv')
    else:
        m_tpic_sequences = data_filtered.groupby('interview_code')['topic'].apply(list).to_frame(name="sequences").reset_index()
        int_codes = m_tpic_sequences.interview_code.to_list()
        transcripts_available = [True if int(item) in transcripts else False for item in int_codes]
        m_tpic_sequences['transcripts_available'] = transcripts_available
        m_tpic_sequences.to_csv('m_tpic_sequences.csv')
    '''



    # Make a list of trajectories

    # First make a list of every transition pairs in the data

    transitions = []

    for element in document_topic_sequences:
        transition = [i for i in window(element)]
        if len(transition)>1:
            transitions.extend(transition)



    trajectories = []

    # Iterate through all transitions
    for element in transitions:
        if 'unknown_topic' not in element:
            #create an empty matrix holding the fuzzy state for the first element of the transition pair
            fuzzy_state=np.zeros([len(topic_labels)])

            # Count how many topics there are 
            state_1_number_of_topics = len(element[0].split("_"))-1
            prob_each_topic_state_1 = 1 / state_1_number_of_topics 
            topic_indices = element[0].split("_")[1:]
            
            for index in topic_indices:
                np.put(fuzzy_state,int(index), prob_each_topic_state_1)


            #create an empty matrix holding the fuzzy state for the second element of the transition pair
            fuzzy_state=np.zeros([len(topic_labels)]).astype(np.float32)
            trajectories.append(fuzzy_state)

            # Count how many topics there are 
            state_2_number_of_topics = len(element[1].split("_"))-1
            prob_each_topic_state_1 = 1 / state_1_number_of_topics 
            topic_indices = element[0].split("_")[1:]
            for index in topic_indices:
                np.put(fuzzy_state,int(index), prob_each_topic_state_1)
                trajectories.append(fuzzy_state)
            pdb.set_trace()



    dtraj_fuzzy = np.vstack(trajectories)

    dtraj_fuzzy = dtraj_fuzzy.astype(np.float64)

    assert np.allclose(dtraj_fuzzy.sum(axis=1), 1)


    #   we estimate a count matrix and transition matrix as we did before with the one-hot encoding
    # in comparison, here the np.linalg.inv() term is not a diagonal matrix, which 
    # is the reason why we have to do the inverse instead of the simpler operation here
    count_matrix_fuzzy = dtraj_fuzzy[:-1].T @ dtraj_fuzzy[1:]
    transition_matrix_fuzzy = np.linalg.inv(dtraj_fuzzy[:-1].T @ dtraj_fuzzy[:-1]) @ count_matrix_fuzzy

    

            
    
    transition_matrix_fuzzy[transition_matrix_fuzzy<0]=0
    transition_matrix_fuzzy = preprocessing.normalize(transition_matrix_fuzzy,axis=1,norm="l1")
    
    #transition_matrix_fuzzy = transition_matrix_fuzzy / transition_matrix_fuzzy.sum(axis=1)
    #transition_matrix_fuzzy = transition_matrix_fuzzy.astype(np.float16)


    


    assert np.allclose(transition_matrix_fuzzy.sum(axis=1), 1)

    
    mm = msm.markov_model(transition_matrix_fuzzy)


    mm.is_reversible

    #Print the stationary distributions of the top states
    results = []
    for i,element in enumerate(mm.pi.argsort()[::-1]):
        print (i)
        print (topic_labels[element])
        print (mm.pi[element])
        print ('\n')
        results.append({'topic_name':topic_labels[element],'stationary_prob':mm.pi[element]})
        if i ==12:
            break

    # Print the eigenvalues of states

    '''for element in mm.eigenvalues().argsort()[::-1]:
        print (topic_list[element])
    '''


    # Calculate the flux between two states (topic_2, selection and topic_8_14 camp liquidiation / camp transfer )


    A = [8]
    B = [2,13]
    tpt = msm.tpt(mm, A, B)

    nCut = 1
    (bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.5)
    cumflux = 0

    # Print the best path between the two states

    print("Path flux\t\t%path\t%of total\tpath")


    for i in range(len(bestpaths)):
        cumflux += bestpathfluxes[i]

        print(bestpathfluxes[i],'\t','%3.1f'%(100.0*bestpathfluxes[i]/tpt.total_flux),'%\t','%3.1f'%(100.0*cumflux/tpt.total_flux),'%\t\t',bestpaths[i])
        
        topic_sequence = []
        for element in bestpaths[i]:
            print (topic_labels[element])
            topic_sequence.append(topic_labels[element])
           #print (topic_labels[element])
        topic_sequence = '-'.join(topic_sequence)
        if (f==0):
            women_topic_sequences[topic_sequence]=100.0*bestpathfluxes[i]/tpt.total_flux
        else:
            men_topic_sequences[topic_sequence]=100.0*bestpathfluxes[i]/tpt.total_flux        #get the path labels


    if (f ==0):
        pd.DataFrame(results).to_csv("women_topics_stationary_prob.csv")
    else:
        pd.DataFrame(results).to_csv("men_topics_stationary_prob.csv")

    # Additionaly calculate the mean time passage between social relations and adaptation


    '''social_relations_indices = [topic_list_with_labels.index(element) for element in topic_list_with_labels if "socialrelations" == element]
    adaptation_relations_indices = [topic_list_with_labels.index(element) for element in topic_list_with_labels if "adaptation" == element]

       

    tpt_social_relations = msm.tpt(mm, social_relations_indices, adaptation_relations_indices)

    print('socialrelations->B adaptation= ', 1.0/tpt_social_relations.rate)
    print (tpt_social_relations.rate)


    tpt_adaptation = msm.tpt(mm, adaptation_relations_indices,social_relations_indices)


    print('adaptation->socialrelations-> ', 1.0/tpt_adaptation.rate)
    print(tpt_adaptation.rate)
    '''
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

pd.DataFrame(result).to_csv('man_women_paths.csv')
pdb.set_trace()







