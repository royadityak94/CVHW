# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import numpy as np
import cv2

def detectBlobs(im, param=None):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 5 array with blob in each row in (x, y, radius, angle, score)
    #
    # Dummy - returns a blob at the center of the image
    blobs = np.array([im.shape[1]/2, im.shape[0]/2, 100, 1.0])
    return np.array(blobs)
