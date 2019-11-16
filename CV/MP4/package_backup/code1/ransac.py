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

def add_extra_dim(matching_blobs):
    return np.hstack([matching_blobs, np.ones((matching_blobs.shape[0], 1))])

def ransac(matches, blobs1, blobs2, param={}):
    default_max_itr, default_reject_limit = 900, 1
    max_iteration = param.get("max_iteration") if param.get("max_iteration") is not None else default_max_itr
    reject_limit = param.get("reject_limit") if param.get("reject_limit") is not None else default_reject_limit
    pairs = max(int(matches.shape[0] * 0.01), 2)
    constant_padding = np.array([0, 0, 1])
    blob1_matches = add_extra_dim(blobs2[:, [0, 1]][matches])
    blob2_matches = add_extra_dim(blobs1[:, [0, 1]][matches])
    total_inliers, best_inlier_model = 0, np.zeros(matches.shape[0])
    for i_ in range(max_iteration):
        idxs = np.random.choice(matches.shape[0], pairs, replace=False)
        selected1 = blob1_matches[idxs, :]
        selected2 = blob2_matches[idxs, :]
        transf = np.linalg.lstsq(selected2, selected1, rcond=None)[0]
        curr_inliers = np.power(np.linalg.norm(np.dot(blob2_matches, transf) - blob1_matches, axis=1), 2) < reject_limit
        if np.sum(curr_inliers) > total_inliers:
            total_inliers = np.sum(curr_inliers)
            best_inlier_model = np.copy(curr_inliers)
    transf, _, _, _ = np.linalg.lstsq(blob2_matches[best_inlier_model], blob1_matches[best_inlier_model], rcond=None)
    return best_inlier_model, transf[:2, :3]