import numpy as np

# EXTRACTDIGITFEATURES extracts features from digit images
#   features = extractDigitFeatures(x, featureType) extracts FEATURES from images
#   images X of the provided FEATURETYPE. The images are assumed to the of
#   size [W H 1 N] where the first two dimensions are the width and height.
#   The output is of size [D N] where D is the size of each feature and N
#   is the number of images. 
def extractDigitFeatures(x, featureType):
    
    if featureType == 'pixel':
        features = zeroFeatures(x)  # implement this
    elif featureType == 'hog':
        features = zeroFeatures(x)  # implement this
    elif featureType == 'lbp':
        features = zeroFeatures(x)  # implement this

    return features


def zeroFeatures(x):
    return np.zeros((10, x.shape[2]))

