import pandas as pd 
import pdb
from itertools import islice
import numpy as np
from pyemma import msm,plots
import msmtools
from msmtools.flux import tpt,ReactiveFlux
from scipy import sparse

np.set_printoptions(suppress=True)


def window(seq, n=2):
    "Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class MarkovChain(object):
    def __init__(self, transition_matrix, states):
        """
        Initialize the MarkovChain instance.
 
        Parameters
        ----------
        transition_matrix: 2-D array
            A 2-D array representing the probabilities of change of 
            state in the Markov Chain.
 
        states: 1-D array 
            An array representing the states of the Markov Chain. It
            needs to be in the same order as transition_matrix.
        """
        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.states = states
        self.index_dict = {self.states[index]: index for index in 
                           range(len(self.states))}
        self.state_dict = {index: self.states[index] for index in
                           range(len(self.states))}
 
    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time 
        instance.
 
        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
         self.states, 
         p=self.transition_matrix[self.index_dict[current_state], :]
        )
 
    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.
 
        Parameters
        ----------
        current_state: str
            The state of the current random variable.
 
        no: int
            The number of future states to generate.
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states



data = pd.read_csv('data/output/topic_sequencing/segment_topics.csv')
data['interview_code'] = data['updated_id'].apply(lambda x: x.split('_')[0])
document_topic_sequences  = data.groupby('interview_code')['topic'].apply(list)

transitions = []
for element in document_topic_sequences:
    #if "unknown_topic" not in element: 
    transition = [i for i in window(element)]
    if len(transition)>1:
        transitions.extend(transition)

topic_list=data.topic.unique().tolist()
topic_list = sorted(topic_list)

transition_matrix = np.zeros([len(topic_list),len(topic_list)]).astype(int)

for element in transitions:
    state1 = element[0]
    state2 = element[1]
    state1_index= topic_list.index(state1)
    state2_index= topic_list.index(state2)
    transition_matrix[state1_index,state2_index] = transition_matrix[state1_index,state2_index] + 1

transition_matrix_scaled = (transition_matrix.T/transition_matrix.sum(axis=1)).T

mm = msm.markov_model(transition_matrix_scaled)

for element in mm.eigenvalues().argsort()[::-1]:
    print (topic_list[element])
transition_matrix_scaled  = sparse.csr_matrix(transition_matrix_scaled)
flux = tpt(transition_matrix_scaled,[1],[2])



pdb.set_trace()
chain = MarkovChain(transition_matrix=transition_matrix,states=topic_list)
pdb.set_trace()

