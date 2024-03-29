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


# Read the input data

input_directory = constants.input_data+'/filtered/'

# Read the biodata
bio_data = constants.input_files_biodata_birkenau

np.set_printoptions(suppress=True)

# Read the topic labels
topic_doc = open('data_analysis/topic_anchors_Birkenau.txt').read()

topic_labels = [element.split('\n')[0].strip() for element in topic_doc.split('\n\n')]

transcripts = pd.read_csv('data_analysis/transcripts_available.csv').IntCode.to_list()





# Read the segments dataframe
data = pd.read_csv('data/output/topic_sequencing/segment_topics_Birkenau.csv')

# Compute the interview code of each segment (the unique id of each segment contains the interview code as prefix)
data['interview_code'] = data['updated_id'].apply(lambda x: x.split('_')[0])

# Read the biodata and identify the interview code of male and female survivors

df_biodata = pd.read_csv(input_directory+bio_data)

#df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']

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
    #110006                    [topic_2, unknown_topic, topic_1]
    document_topic_sequences  = data_filtered.groupby('interview_code')['topic'].apply(list)

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
        



    # Make a transition matrix from each topic

    # First make a list of every transition pairs

    transitions = []
    for element in document_topic_sequences:
       # uncomment the line below if you want to add only those interviews where every topic is known
       #if "unknown_topic" not in element:
        transition = [i for i in window(element)]
        if len(transition)>1:
            transitions.extend(transition)

    # Make a definite topic list and remove unknown topic, this will be the index of the transition matrix

    topic_list=data.topic.unique().tolist()
    topic_list.remove('unknown_topic')
    topic_list = sorted(topic_list)


    # Create an empty transition matrix

    transition_matrix = np.zeros([len(topic_list),len(topic_list)]).astype(int)


    # Iterate through all transitions
    for element in transitions:
        if 'unknown_topic' not in element:

            # Get the two states
            state1 = element[0]
            state2 = element[1]

            
            # Get the indices of the two states
            state1_index= topic_list.index(state1)
            state2_index= topic_list.index(state2)

            # Fill in the necessary row - column based on the transition
            transition_matrix[state1_index,state2_index] = transition_matrix[state1_index,state2_index] + 1
    
    topic_list.index('topic_9')
    transition_matrix[topic_list.index('topic_9'),topic_list.index('topic_10')]
    transition_matrix[topic_list.index('topic_9'),topic_list.index('topic_10')]/transition_matrix[topic_list.index('topic_9')].sum()
    



    
    # Remove those states from which transitions begin but to which no transitions goes
    # If this is not done, the transition matrix is singular 
    to_be_removed=np.sort(np.where(np.all(transition_matrix == 0, axis=1)==True))[0][::-1]
    for element in to_be_removed:
        transition_matrix=np.delete(transition_matrix,element,axis=1)
        transition_matrix=np.delete(transition_matrix,element,axis=0)
        # Delete those transitions from the topic list
        del topic_list[element]


    




    # Create the final transition matrix with probability values

    transition_matrix_scaled = (transition_matrix.T/transition_matrix.sum(axis=1)).T


    # Make sure that transition matrix is standard MSMs (reversibly connect), 
    #i.e. that for every state in the transition matrix there must be at least one transition out of it and into it. 
    #tim.hempel[at]fu-berlin.de


    # Check null columns



    null_columns = np.sort(np.where(transition_matrix_scaled.sum(0)==0)[0])[::-1]

    # Remove null columns
    for null in null_columns:
        transition_matrix_scaled = np.delete(transition_matrix_scaled,null, 0)  
        transition_matrix_scaled = np.delete(transition_matrix_scaled,null, 1)
        del topic_list[null] 

    
    # Find the final topic labels (until now for instance, topic_1_2 was used,)
    topic_list_with_labels=[]
    for element in topic_list:
        topic_n = element.split('_')[1:]
        try:
            labels = '_'.join([topic_labels[int(l)] for l in topic_n])
        except:
            pdb.set_trace()
        topic_list_with_labels.append(labels)

    transition_matrix_scaled[topic_list.index('topic_9'),topic_list.index('topic_10')]

    # Train the markov chain

    mm = msm.markov_model(transition_matrix_scaled)


    mm.is_reversible

    #Print the stationary distributions of the top states
    results = []
    for i,element in enumerate(mm.pi.argsort()[::-1]):
        print (i)
        print (topic_list_with_labels[element])
        print (topic_list[element])
        print (mm.pi[element])
        print ('\n')
        results.append({'topic_name':topic_list_with_labels[element],'stationary_prob':mm.pi[element]})
        if i ==30:
            break

    # Print the eigenvalues of states

    '''for element in mm.eigenvalues().argsort()[::-1]:
        print (topic_list[element])
    '''


    # Calculate the flux between two states (topic_2, selection and topic_8_14 camp liquidiation / camp transfer )


    A = [topic_list.index('topic_8')]
    B = [topic_list.index('topic_2')]
    tpt = msm.tpt(mm, A, B)

    nCut = 1
    (bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.99)
    cumflux = 0

    # Print the best path between the two states

    print("Path flux\t\t%path\t%of total\tpath")


    for i in range(len(bestpaths)):
        cumflux += bestpathfluxes[i]

        print(bestpathfluxes[i],'\t','%3.1f'%(100.0*bestpathfluxes[i]/tpt.total_flux),'%\t','%3.1f'%(100.0*cumflux/tpt.total_flux),'%\t\t',bestpaths[i])
        
        topic_sequence = []
        for element in bestpaths[i]:
            print (topic_list_with_labels[element])
            topic_sequence.append(topic_list_with_labels[element])
            print (topic_list[element])
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







