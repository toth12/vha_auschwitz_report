#!/usr/bin/env python
# coding: utf-8

import msmtools
import numpy as np
import pdb
import numpy as np
from msmtools.estimation import connected_sets,is_connected,largest_connected_submatrix

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


# example: 3x3 transition matrix
# this would be the matrix that you estimated on your full set of states
Tmat_full = np.array([[0.45, 0.45, 0.1],
                     [0.45, 0.45, 0.1],
                     [0.1, 0.1, 0.8]])

print(msmtools.analysis.is_transition_matrix(Tmat_full))

# the example transition matrix is between three states,
# but you see that states 0 and 1 exchange with very high probability
# so we want to put those into one single state and construct a 
# matrix with only 2 states, A and B, as follows:

# old state -> new state
# 0 -> A
# 1 -> A
# 2 -> B

# the membership matrix chi defines this map
# each row gives us a binary encoding for a given
# state 0, 1, 2 to a state A, B

# state  goes to A, B
chi = np.array([[1, 0],  # for state 0
               [1, 0],   # for state 1
               [0, 1]])  # for state 2

# this could be a 'fuzzy' state assignment as well.


# the resulting 'coarse' transition matrix is this.
# compare the probabilities to the above one.

print(cg_transition_matrix(Tmat_full, chi))

C = np.array([[10, 1, 0], [2, 0, 3], [0, 0, 4]])
cc_directed = connected_sets(C)

pdb.set_trace()
