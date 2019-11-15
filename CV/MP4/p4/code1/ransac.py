import numpy as np
import cv2
import time

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 4

def ransac(matches, blobs1, blobs2, param={}):
    pairs = int(matches.shape[0] * .3)
    default_max_itr, default_reject_limit = 1000, 1000
    max_iteration = param.get("max_iteration") if param.get("max_iteration") is not None else default_max_itr
    reject_limit = param.get("reject_limit") if param.get("reject_limit") is not None else default_reject_limit
    
    match_blob1 = blobs1[matches]
    match_blob2 = blobs2[matches]
    total_inliers, best_inlier_model = 0, np.zeros(matches.shape[0])
    for n in range(max_iteration):
        idxs = np.random.choice(matches.shape[0], pairs)
        matching1 = match_blob1[idxs, :]
        matching2 = match_blob2[idxs, :]
        transf = np.linalg.lstsq(matching2, matching1, rcond=None)[0]
        transf[:, 2] = np.array([0, 0, 1])
        curr_inliers = np.power(np.linalg.norm(np.dot(match_blob2, transf) - match_blob1, axis=1), 2) < reject_limit
        if curr_inliers.sum() > total_inliers:
            print ("Uauua")
            total_inliers = curr_inliers.sum()
            best_inlier_model = curr_inliers.copy()     
    transf = np.linalg.lstsq(match_blob2[best_inlier_model], match_blob1[best_inlier_model], rcond=None)[0]
    
    if np.sum(transf - np.eye(3)) == 0:
        best_inlier_model = -1
    
    return best_inlier_model, transf[:2, :3]


