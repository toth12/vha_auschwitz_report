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
import argparse




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

def calculate_flux(mm,topic_labels,A=[8],B=[2,13]):
    #A=[8],B=[2,13],
    # Calculate the flux between two states camp arrival and camp liquidiation / camp transfer )


    
    tpt = msm.tpt(mm, A, B)

    nCut = 1
    (bestpaths,bestpathfluxes) = tpt.pathways(fraction=0.7)
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

def process_data(data_set):

    transitions = create_transitions(data_set)
    transition_matrix,topic_list_result = calculate_transition_matrix(transitions,topic_labels_originals[:])
    mm= train_markov_chain(transition_matrix)
    stationary_probs = print_stationary_distributions(mm,topic_list_result)
    try:

        paths = calculate_flux(mm,topic_list_result,path_start,path_end)
    except:
        paths ={"nan":0}

    df_stationary_probs=pd.DataFrame(stationary_probs)
    mean_passage_time=calculate_mean_passage_time_between_states(mm,topic_list_result)

    return {"stationary_probs":stationary_probs,"paths":paths,"mean_passage_time":mean_passage_time}

def post_process_topic_sequences(sequence):
    result = []
    for element in sequence:
        topics=element.split('_')[1:]
        partial_result = []
        for topic_n in topics:
            partial_result.append(topic_labels_originals[int(topic_n)])
        partial_result = '-'.join(partial_result)
        result.append(partial_result)
    return result



def print_topic_sequences(data,filename):
    df = data.groupby('interview_code')['topic'].apply(list).to_frame(name="sequences").reset_index()
    df['topic_sequences'] = df['sequences'].apply(post_process_topic_sequences)
    df.to_csv(constants.output_data_topic_sequences+'instances_of_sequences/'+filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', nargs='+')
    parser.add_argument('--to', nargs='+')
    for key, value in parser.parse_args()._get_kwargs():
        if (key == "from"):
            if (value is not None):
                path_start = value
            else:
                path_start = ['8']
        if (key == "to"):
            if (value is not None):
                path_end = value
            else:
                path_end = ['2','13']


    metadata_fields = ['complete','CountryOfBirth','easy','medium','hard',"not_work","work"]



    np.set_printoptions(suppress=True)

    # Read the input data
    input_directory = constants.input_data
    

    main_output_directory = constants.output_data_topic_sequences+'paths/'+'_'.join(path_start)+'|'+'_'.join(path_end)+'/'

    try:
        os.mkdir(main_output_directory)
    except:
        pass

    path_start = [int(el) for el in path_start]
    path_end = [int(el) for el in path_end]

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
    df_biodata = pd.read_csv(input_directory+bio_data)
    df_biodata = df_biodata.fillna(0)

    df_biodata  = df_biodata[((df_biodata.Birkenau_segment_percentage>0.7)&(df_biodata.earliest_year>1942)&(df_biodata.is_transfer_route==False)&(df_biodata.length_in_minutes>10))]




    country_of_origins = df_biodata.groupby('CountryOfBirth')['CountryOfBirth'].count().to_frame('Count').reset_index()
    country_of_origins= country_of_origins[country_of_origins.Count>50]
    country_of_origins = country_of_origins.CountryOfBirth.tolist()

    df_biodata = df_biodata[(df_biodata.ExperienceGroup=='Jewish Survivor')]

    IntCodeM = df_biodata[df_biodata.Gender=='M']['IntCode'].to_list()
    IntCodeW = df_biodata[df_biodata.Gender=='F']['IntCode'].to_list()


    for element in metadata_fields:
       
        woman_data = []
        man_data =[]
        # Gather data
        if element == "complete":

            woman_data.append(data[data.interview_code.isin(IntCodeW)])
            man_data.append(data[data.interview_code.isin(IntCodeM)])
            print_topic_sequences(data[data.interview_code.isin(IntCodeW)],element+'_woman.csv')
            print_topic_sequences(data[data.interview_code.isin(IntCodeM)],element+'_man.csv')

        elif element == 'CountryOfBirth':


            for country in country_of_origins:
                int_codes = df_biodata[df_biodata.CountryOfBirth==country].IntCode.tolist()
                complete_data = data[data.interview_code.isin(int_codes)]
                print_topic_sequences(complete_data[complete_data.interview_code.isin(IntCodeW)],element+'_'+country+'_woman.csv')
                print_topic_sequences(complete_data[complete_data.interview_code.isin(IntCodeM)],element+'_'+country+'_man.csv')
                woman_data.append(complete_data[complete_data.interview_code.isin(IntCodeW)])
                man_data.append(complete_data[complete_data.interview_code.isin(IntCodeM)])

        elif ((element == 'easy') or (element == 'medium') or (element == 'hard')):
            int_codes = df_biodata[df_biodata[element]==1].IntCode.tolist()
            complete_data = data[data.interview_code.isin(int_codes)]

            print_topic_sequences(complete_data[complete_data.interview_code.isin(IntCodeW)],element+'_'+'_woman.csv')
            print_topic_sequences(complete_data[complete_data.interview_code.isin(IntCodeM)],element+'_'+'_man.csv')


            woman_data.append(complete_data[complete_data.interview_code.isin(IntCodeW)])
            man_data.append(complete_data[complete_data.interview_code.isin(IntCodeM)])
        elif element == "not_work":
            int_codes  = df_biodata[(df_biodata['easy']==0)&(df_biodata['hard']==0)&(df_biodata['medium']==0)].IntCode.tolist()
            complete_data = data[data.interview_code.isin(int_codes)]
            print_topic_sequences(complete_data[complete_data.interview_code.isin(IntCodeW)],element+'_'+'_woman.csv')
            print_topic_sequences(complete_data[complete_data.interview_code.isin(IntCodeM)],element+'_'+'_man.csv')
            woman_data.append(complete_data[complete_data.interview_code.isin(IntCodeW)])
            man_data.append(complete_data[complete_data.interview_code.isin(IntCodeM)])
        elif element == "work":
            int_codes  = df_biodata[(df_biodata['easy']==1)|(df_biodata['hard']==1)|(df_biodata['medium']==1)].IntCode.tolist()
            complete_data = data[data.interview_code.isin(int_codes)]
            print_topic_sequences(complete_data[complete_data.interview_code.isin(IntCodeW)],element+'_'+'_woman.csv')
            print_topic_sequences(complete_data[complete_data.interview_code.isin(IntCodeM)],element+'_'+'_man.csv')
            woman_data.append(complete_data[complete_data.interview_code.isin(IntCodeW)])
            man_data.append(complete_data[complete_data.interview_code.isin(IntCodeM)])

        # Process the data

        # First the women data
        women_output = []

        for data_element in woman_data:
            
            women_output.append(process_data(data_element))

            


        # Second the men data
        men_output = []
        for data_element in man_data:
            men_output.append(process_data(data_element))
    
    
        
        for f in range(0,len(women_output)):

            if element == "complete":
                
                df_complete_stationary_probs = pd.merge(pd.DataFrame(men_output[f]['stationary_probs']),pd.DataFrame(women_output[f]['stationary_probs']),how="outer", on=['topic_name'],suffixes=("_complete_man", "_complete_woman"))
                df_complete_filtered_paths = create_dataframe_with_paths(women_output[f]['paths'],men_output[f]['paths'],filter_stat=['social','aid'])
                df_complete_paths = create_dataframe_with_paths(women_output[f]['paths'],men_output[f]['paths'],filter_stat=None)   
            else:

                df_metadata_paths = create_dataframe_with_paths(women_output[f]['paths'],men_output[f]['paths'],filter_stat=None)   
                df_metadata_filtered_paths = create_dataframe_with_paths(women_output[f]['paths'],men_output[f]['paths'],filter_stat=['social','aid'])
                df_metadata_stationary_probs = pd.merge(pd.DataFrame(men_output[f]['stationary_probs']),pd.DataFrame(women_output[f]['stationary_probs']),how="outer", on=['topic_name'],suffixes=("_man", "_woman"))
                df_metadata_filtered_paths_with_complete_paths =pd.merge(df_complete_filtered_paths,df_metadata_filtered_paths,how="outer", on=['path'],suffixes=("_complete", "_meta"))


            # Print all data to files
            output_directory = main_output_directory + element + '/'
            try:
                os.mkdir(path+'/'+output_directory)
            except:
                print("output folder exists")
            if element == "CountryOfBirth":
                filename = country_of_origins[f]+'_'

            else:
                filename = element+'_'

            output_path = output_directory+filename
            if element == "complete":
                df_complete_paths.to_csv(output_path+'all_paths.csv')
                df_complete_filtered_paths.to_csv(output_path+'filtered_paths.csv')
                women_output[f]['mean_passage_time'].to_csv(output_path+'women_mean_passage_time.csv')
                men_output[f]['mean_passage_time'].to_csv(output_path+'men_mean_passage_time.csv')

            else:
                df_metadata_filtered_paths_with_complete_paths.to_csv(output_path+'filtered_paths_with_complete_paths.csv')
                df_metadata_filtered_paths.to_csv(output_path+'filtered_paths.csv')
                df_metadata_paths.to_csv(output_path+'all_paths.csv')

                women_output[f]['mean_passage_time'].to_csv(output_path+'women_mean_passage_time.csv')
                men_output[f]['mean_passage_time'].to_csv(output_path+'men_mean_passage_time.csv')






            # Merge it with the final dataframe holding all stationary probs
            if element == "CountryOfBirth":
                df_woman_stationary_probs = pd.DataFrame(women_output[f]['stationary_probs']).rename(columns={"stationary_prob":"stationary_prob_women_"+country_of_origins[f]})
                df_man_stationary_probs = pd.DataFrame(men_output[f]['stationary_probs']).rename(columns={"stationary_prob":"stationary_prob_men_"+country_of_origins[f]})
                df_complete_stationary_probs = pd.merge(df_complete_stationary_probs,df_woman_stationary_probs,on="topic_name")
                df_complete_stationary_probs = pd.merge(df_complete_stationary_probs,df_man_stationary_probs,on="topic_name")

            elif((element == 'easy') or (element == 'medium') or (element == 'hard') or (element == 'not_work') or (element == 'work') ):
                df_woman_stationary_probs = pd.DataFrame(women_output[f]['stationary_probs']).rename(columns={"stationary_prob":"stationary_prob_women_"+element})
                df_man_stationary_probs = pd.DataFrame(men_output[f]['stationary_probs']).rename(columns={"stationary_prob":"stationary_prob_men_"+element})
                df_complete_stationary_probs = pd.merge(df_complete_stationary_probs,df_woman_stationary_probs,on="topic_name")
                df_complete_stationary_probs = pd.merge(df_complete_stationary_probs,df_man_stationary_probs,on="topic_name")

    
            
    df_complete_stationary_probs.set_index('topic_name').T.to_csv(main_output_directory + 'complete' + '/stationary_probs.csv')
           




