import numpy as np
import msmtools
import pdb


# in this script I explain at example how to estimate a 
# transition matrix from fuzzy state trajectories.
# I start with a regular transition matrix estimation from a count matrix,
# show that the same result is obtained in a different way
# and show that this alternative route is applicable to fuzzy state trajectories.

# IMPORTANT: I use a lag time of 1 in this example! please adjust accordingly when
# applying to your data

# regular discrete trajectory with 2 states
dtraj = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1])
np.array([[[0.95, 0.05],[0.05, 0.95],[0.5, 0.5]],[[0.9, 0.01],[0, 1],[0.5, 0.5]]])

pdb.set_trace()





# estimation of the count matrix by counting
count_matrix_by_counting = np.zeros((2, 2))
for t in range(len(dtraj)-1):
    index_time_t = dtraj[t]
    index_time_tplus1 = dtraj[t+1]
    count_matrix_by_counting[index_time_t, index_time_tplus1] +=1
    
# transition matrix estimation from count matrix by division by number of counts per state
# (written as a matrix multiplication)
transition_matrix_by_counting = np.diag(1/count_matrix_by_counting.sum(axis=1)) @ count_matrix_by_counting

# note that the first guy on the right side is the inverse of the "instantaneous counts"
# i.e. the counts to be in one state (not going in our out)

print(transition_matrix_by_counting)

# we can use msmtools to check if it's a valid transition matrix
print(msmtools.analysis.is_transition_matrix(transition_matrix_by_counting))

# https://numpy.org/doc/1.18/reference/generated/numpy.matmul.html
# re-write the discrete trajectory to a one hot encoding. That's the same
# as before but written differently. in this formulation, we could also use 
# fuzzy states. here, we have crisp state assignments, so it's always binary.
# the new state assignment is this:
# 0 -> (1, 0)
# 1 -> (0, 1)
# for more states, it would require a larger vector, e.g. 5 -> (0, 0, 0, 0, 0, 1)

dtraj_one_hot_encoded = np.zeros((dtraj.shape[0], len(set(dtraj))))

for step, state in enumerate(dtraj):
    dtraj_one_hot_encoded[step, state] = 1



# estimation of the count matrix from this discrete trajectory
# note that this is a matrix product @ between two arrays which are time shifted
# by the lag time
count_matrix_by_multiplication = dtraj_one_hot_encoded[:-1].T @ dtraj_one_hot_encoded[1:]


# instead of taking the diag as above, which would be 
# np.diag(1/count_matrix_by_multiplication.sum(axis=1))
# we use the more general formulation 
# np.linalg.inv(dtraj_one_hot_encoded[:-1].T @ dtraj_one_hot_encoded[:-1])
# you can check that in this case it's the same


# we estimate the transition matrix from that as before 
# and obtain the same result
transition_matrix_by_multiplication = np.linalg.inv(dtraj_one_hot_encoded[:-1].T @ dtraj_one_hot_encoded[:-1]) @ count_matrix_by_multiplication

print(transition_matrix_by_multiplication)

print(transition_matrix_by_counting == transition_matrix_by_multiplication)


# now we define a trajectory in the same way as we did before
# but instead of crisp assignments, we assign each frame to 
# a state with a certain probability
# note that the probability of each frame to belong to any state 
# must sum up to one! 
dtraj_fuzzy = np.array([[0.95, 0.05],
                       [0.05, 0.95],
                       [0.5, 0.5],
                       [0.05, 0.95],
                       [0.05, 0.95],
                       [.95, 0.05],
                       [1., 0.0],
                       [0.05, 0.95],
                       [.95, 0.05],
                       [.95, 0.05],
                       [.95, 0.05],
                       [0.05, 0.95],
                       [0.0, 1.],
                       [.95, 0.05],
                       [0.05, 0.95]])

# note that the probability of each frame to belong to any state 
# must sum up to one! 
assert np.allclose(dtraj_fuzzy.sum(axis=1), 1)


# we estimate a count matrix and transition matrix as we did before with the one-hot encoding
# in comparison, here the np.linalg.inv() term is not a diagonal matrix, which 
# is the reason why we have to do the inverse instead of the simpler operation here
count_matrix_fuzzy = dtraj_fuzzy[:-1].T @ dtraj_fuzzy[1:]
transition_matrix_fuzzy = np.linalg.inv(dtraj_fuzzy[:-1].T @ dtraj_fuzzy[:-1]) @ count_matrix_fuzzy
transition_matrix_fuzzy

pdb.set_trace()

# this yields a different transition matrix as it's not the same data anymore
print(transition_matrix_fuzzy)

# make sure this is a valid discrete trajectory
print(msmtools.analysis.is_transition_matrix(transition_matrix_fuzzy))