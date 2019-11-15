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
    reject_limit = 1 #.95
    matches = []
    distance_matrix = cdist(f1, f2)
    
    for n in range(distance_matrix.shape[0]):
        if n > 
        currd_matrix = distance_matrix[n]
        if (np.min(currd_matrix) / np.partition(currd_matrix, 2)[1]) < reject_limit:
            matches.append([n])
    matches = np.squeeze(np.array(matches))
    return matches