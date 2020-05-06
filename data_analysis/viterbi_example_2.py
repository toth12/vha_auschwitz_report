import hmmlearn.hmm as hmm
import numpy as np
import pdb

transmat = np.array([[1, 0],
                     [1, 0]])
emitmat = np.array([[0.9, 0.1],
                    [0.2, 0.8]])

startprob = np.array([[0.5, 0.5]])
model = hmm.MultinomialHMM(n_components=2)
#model.emissionprob_ = emitmat

model.startprob=startprob
model.transmat=transmat
# works fine
model.fit([[1, 1],[1, 1],[1, 1]]) 

print (model.decode([[0], [0]]))

print (model.predict([[0]]))

pdb.set_trace()