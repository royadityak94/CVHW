import numpy as np
from scipy.spatial.distance import cdist

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 4

def computeMatches(f1, f2):
    """ Match two sets of SIFT features f1 and f2 """
    reject_limit = .55
    matches = []
    distance_matrix = cdist(f1, f2)
    
    smallest = f1
    if (f1.shape[0] > f2.shape[0]):
        smallest = f2
    
    for n in range(smallest.shape[0]):
        currd_matrix = distance_matrix[n]
        if (np.min(currd_matrix) / np.partition(currd_matrix, 2)[1]) <= reject_limit:
            matches.append([n])
    matches = np.squeeze(np.array(matches))
    return matches

