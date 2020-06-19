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
import os




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
joint_topic_sequences = {}
women_topics_stationary_prob = {}
men_topics_stationary_prob = {}
joint_topics_stationary_prob = {}

def create_transitions(data):
    # Create a new dataframe in which every interview is a sequence of topics
    # For instance

    #interview_code
    #110006                    [topic_2, topic_1]
    document_topic_sequences  = data.groupby('interview_code')['topic'].apply(list)

    # First make a list of every transition pairs in the data

    transitions = []

    for element in document_topic_sequences:
        transition = [i for i in window(element)]
        if len(transition)>1:
            transitions.extend(transition)

    return transitions

def calculate_transition_matrix(transitions,topic_labels):

    trajectories = []

    # Iterate through all transitions
    for element in transitions:

        #create an empty matrix holding the fuzzy state for the first element of the transition pair
        fuzzy_state=np.zeros([len(topic_labels)])

        # Count how many topics there are 
        state_1_number_of_topics = len(element[0].split("_"))-1
        prob_each_topic_state_1 = 1 / state_1_number_of_topics 
        topic_indices = element[0].split("_")[1:]
        
        for index in topic_indices:
            np.put(fuzzy_state,int(index), prob_each_topic_state_1)

        trajectories.append(fuzzy_state)
        

        #create an empty matrix holding the fuzzy state for the second element of the transition pair
        fuzzy_state=np.zeros([len(topic_labels)]).astype(np.float32)
        

        # Count how many topics there are 
        state_2_number_of_topics = len(element[1].split("_"))-1
        prob_each_topic_state_2 = 1 / state_2_number_of_topics 
        topic_indices = element[1].split("_")[1:]
        for index in topic_indices:
            try:
                np.put(fuzzy_state,int(index), prob_each_topic_state_2)
            except:
                pdb.set_trace()
        trajectories.append(fuzzy_state)
       
    print(len(transitions))        
    print(len(trajectories))     
    assert len(transitions)*2==len(trajectories)
    dtraj_fuzzy = np.vstack(trajectories)

    dtraj_fuzzy = dtraj_fuzzy.astype(np.float64)


    assert np.allclose(dtraj_fuzzy.sum(axis=1), 1)

    # Check for null columns

    null_columns = np.sort(np.where(dtraj_fuzzy.sum(0)==0)[0])[::-1]

     # Remove null columns
    for null in null_columns:
        dtraj_fuzzy= np.delete(dtraj_fuzzy,null, 0)  
        dtraj_fuzzy = np.delete(dtraj_fuzzy,null, 1)
        del topic_labels[null]



    #   we estimate a count matrix and transition matrix as we did before with the one-hot encoding
    # in comparison, here the np.linalg.inv() term is not a diagonal matrix, which 
    # is the reason why we have to do the inverse instead of the simpler operation here
    count_matrix_fuzzy = dtraj_fuzzy[:-1].T @ dtraj_fuzzy[1:]
    try:
        transition_matrix_fuzzy = np.linalg.inv(dtraj_fuzzy[:-1].T @ dtraj_fuzzy[:-1]) @ count_matrix_fuzzy
    except:
        pdb.set_trace()

    

            
    
    transition_matrix_fuzzy[transition_matrix_fuzzy<0]=0
    transition_matrix_fuzzy = preprocessing.normalize(transition_matrix_fuzzy,axis=1,norm="l1")


    assert np.allclose(transition_matrix_fuzzy.sum(axis=1), 1)

    return (transition_matrix_fuzzy,topic_labels)
    
def train_markov_chain(transition_matrix):
    mm = msm.markov_model(transition_matrix)
    #mm = msm.MSM(transition_matrix,neig=13)
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6910584/

    # Use this to check if reversible
    mm.is_reversible

    

    return mm

def print_stationary_distributions(mm,topic_labels):
    #Print the stationary distributions of the top states
    results = []
    for i,element in enumerate(mm.pi.argsort()[::-1]):
        #print (i)
        #print (topic_labels[element])
        #print (mm.pi[element])
        #print ('\n')
        results.append({'topic_name':topic_labels[element],'stationary_prob':mm.pi[element]})
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

def calculate_flux(mm,topic_labels,A=[8],B=[2,13],):
    # Calculate the flux between two states camp arrival and camp liquidiation / camp transfer )


    
    tpt = msm.tpt(mm, A, B)

    nCut = 1
    (bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.8)
    cumflux = 0

    # Print the best path between the two states

    print("Path flux\t\t%path\t%of total\tpath")

    topic_sequences = {}
    for i in range(len(bestpaths)):
        cumflux += bestpathfluxes[i]
        flux = 100.0*bestpathfluxes[i]/tpt.total_flux
        if flux > 1:
            #print(bestpathfluxes[i],'\t','%3.1f'%(100.0*bestpathfluxes[i]/tpt.total_flux),'%\t','%3.1f'%(100.0*cumflux/tpt.total_flux),'%\t\t',bestpaths[i])
            
            topic_sequence = []
            for element in bestpaths[i]:
                #print (topic_labels[element])
                topic_sequence.append(topic_labels[element])
            topic_sequence = '-'.join(topic_sequence)
            topic_sequences[topic_sequence]=100.0*bestpathfluxes[i]/tpt.total_flux
        
    return topic_sequences
    
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



if __name__ == '__main__':
    metadata_fields = ['complete','CountryOfBirth',]



    np.set_printoptions(suppress=True)

    # Read the input data
    input_directory = constants.input_data

    output_directory = constants.output_data_topic_sequences

    # Current work directory

    path = os.getcwd()

    # Read the biodata
    bio_data = constants.input_files_biodata_with_inferred_fields

    # Read the topic labels
    topic_doc = open('data_analysis/topic_anchors_Birkenau.txt').read()
    topic_labels_originals = [element.split('\n')[0].strip() for element in topic_doc.split('\n\n')]
    transcripts = pd.read_csv('data_analysis/transcripts_available.csv').IntCode.to_list()

    # Read the segments dataframe
    data = pd.read_csv('data/output/topic_sequencing/document_index_with_topic_labels.csv')

    # Eliminate those interview segments where the topic label us unknown_topic
    data = data[~data.new_segment_id.isin(data[data.topic =="unknown_topic"].new_segment_id.tolist())]

    # Compute the interview code of each segment (the unique id of each segment contains the interview code as prefix)
    data['interview_code'] = data['new_segment_id'].apply(lambda x: int(x.split('_')[0]))

    # Read the biodata and identify the interview code of male and female survivors
    #df_biodata = pd.read_excel(input_directory+bio_data, sheet_name=None)['Sheet1']
    df_biodata = pd.from_csv(input_directory+bio_data)

    pdb.set_trace()


    country_of_origins = df_biodata.groupby('CountryOfBirth')['CountryOfBirth'].count().to_frame('Count').reset_index()
    country_of_origins= country_of_origins[country_of_origins.Count>50]

    df_biodata = df_biodata[(df_biodata.ExperienceGroup=='Jewish Survivor')]

    IntCodeM = df_biodata[df_biodata.Gender=='M']['IntCode'].to_list()
    IntCodeW = df_biodata[df_biodata.Gender=='F']['IntCode'].to_list()


    for element in metadata_fields:
        if element == "complete":


            transitions = create_transitions(data)
            transition_matrix,topic_list_result = calculate_transition_matrix(transitions,topic_labels_originals[:])
            mm= train_markov_chain(transition_matrix)
            complete_stationary_probs = print_stationary_distributions(mm,topic_list_result)
            paths = calculate_flux(mm,topic_list_result)


            woman_data = data[data.interview_code.isin(IntCodeW)]
            transitions = create_transitions(woman_data)
            transition_matrix,topic_list_result = calculate_transition_matrix(transitions,topic_labels_originals[:])
            mm= train_markov_chain(transition_matrix)
            woman_stationary_probs = print_stationary_distributions(mm,topic_list_result)
            woman_paths = calculate_flux(mm,topic_list_result)
            df_woman_stationary_probs=pd.DataFrame(woman_stationary_probs)

            women_mean_passage_time=calculate_mean_passage_time_between_states(mm,topic_list_result)


            man_data = data[data.interview_code.isin(IntCodeM)]
            transitions = create_transitions(man_data)
            transition_matrix,topic_list_result = calculate_transition_matrix(transitions,topic_labels_originals[:])
            mm= train_markov_chain(transition_matrix)
            man_stationary_probs = print_stationary_distributions(mm,topic_list_result)
            df_man_stationary_probs=pd.DataFrame(man_stationary_probs)
            man_paths = calculate_flux(mm,topic_list_result)
            
            men_mean_passage_time=calculate_mean_passage_time_between_states(mm,topic_list_result)

            df_complete = create_dataframe_with_paths(woman_paths,man_paths,filter_stat=None)   
            df_complete_filtered = create_dataframe_with_paths(woman_paths,man_paths,filter_stat=['social','aid'])
            df_complete_stationary_probs = pd.merge(df_man_stationary_probs,df_woman_stationary_probs,how="outer", on=['topic_name'],suffixes=("_complete_man", "_complete_woman"))

            
            # check if the output folder exists 

            try:
                os.mkdir(path+'/'+output_directory+element)
            except:
                print("output folder exists")

            df_complete_filtered.to_csv(path+'/'+output_directory+element+'/'+element+'_filtered.csv')
            df_complete_filtered.to_csv(path+'/'+output_directory+element+'/'+element+'.csv')
            men_mean_passage_time.to_csv(path+'/'+output_directory+element+'/men_mean_passage_time.csv')
            women_mean_passage_time.to_csv(path+'/'+output_directory+element+'/women_mean_passage_time.csv')
        if element == 'CountryOfBirth':
            stationary_probs = df_complete_stationary_probs.copy()
            for country in country_of_origins.CountryOfBirth.tolist():
                print (country)
                int_codes = df_biodata[df_biodata.CountryOfBirth==country].IntCode.tolist()
                complete_data = data[data.interview_code.isin(int_codes)]
                woman_data = complete_data[complete_data.interview_code.isin(IntCodeW)]
                man_data = complete_data[complete_data.interview_code.isin(IntCodeM)]



                transitions = create_transitions(woman_data)
                transition_matrix,topic_list = calculate_transition_matrix(transitions,topic_labels_originals[:])
                mm= train_markov_chain(transition_matrix)
                woman_stationary_probs = print_stationary_distributions(mm,topic_list)
                woman_paths = calculate_flux(mm,topic_list)
                women_mean_passage_time=calculate_mean_passage_time_between_states(mm,topic_list)


                transitions = create_transitions(man_data)
                transition_matrix,topic_list = calculate_transition_matrix(transitions,topic_labels_originals[:])
                mm= train_markov_chain(transition_matrix)
                man_stationary_probs = print_stationary_distributions(mm,topic_list)
                man_paths = calculate_flux(mm,topic_list)
                men_mean_passage_time=calculate_mean_passage_time_between_states(mm,topic_list_result)

                df_metadata = create_dataframe_with_paths(woman_paths,man_paths,filter_stat=None)
                df_metadata_filtered = create_dataframe_with_paths(woman_paths,man_paths,filter_stat=['social','aid'])

                paths_complete_metadata=pd.merge(df_complete_filtered,df_metadata_filtered,how="outer", on=['path'],suffixes=("_complete", "_meta"))

                try:
                    os.mkdir(path+'/'+output_directory+element)
                except:
                    print("output folder exists")

                df_metadata.to_csv(path+'/'+output_directory+element+'/'+country+'.csv')
                df_metadata_filtered.to_csv(path+'/'+output_directory+element+'/'+country+'_filtered.csv')
                paths_complete_metadata.to_csv(path+'/'+output_directory+element+'/'+country+'_complete_filtered.csv')

                women_mean_passage_time.to_csv(path+'/'+output_directory+element+'/'+country+'_women_mean_passage_time.csv')
                men_mean_passage_time.to_csv(path+'/'+output_directory+element+'/'+country+'_men_mean_passage_time.csv')
                #Save the stationary probs
                df_woman_stationary_probs = pd.DataFrame(woman_stationary_probs)
                df_man_stationary_probs = pd.DataFrame(man_stationary_probs)  
                

                df_woman_stationary_probs = df_woman_stationary_probs.rename(columns={"stationary_prob":"stationary_prob_women_"+country})
                df_man_stationary_probs = df_man_stationary_probs.rename(columns={"stationary_prob":"stationary_prob_men_"+country})

                stationary_probs = pd.merge(stationary_probs,df_woman_stationary_probs,on="topic_name")
                stationary_probs = pd.merge(stationary_probs,df_man_stationary_probs,on="topic_name")

    stationary_probs.set_index('topic_name').T.to_csv(path+'/'+output_directory+element+'/stationary_probs.csv')
           




