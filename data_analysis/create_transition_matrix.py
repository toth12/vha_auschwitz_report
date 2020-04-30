import pandas as pd 
import pdb
from itertools import islice


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
    #if "unknown_topic" not in element: 
    transition = [i for i in window(element)]
    if len(transition)>1:
        transitions.extend(transition)


pairs = pd.DataFrame(transitions, columns=['state1','state2'])

counts = pairs.groupby('state1')['state2'].value_counts()
probs = (counts / counts.sum()).unstack()

pdb.set_trace()