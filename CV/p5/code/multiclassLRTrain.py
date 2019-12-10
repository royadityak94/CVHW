import numpy as np

def multiclassLRTrain(x, y, param):

    classLabels = np.unique(y)
    numClass = classLabels.shape[0]
    numFeats = x.shape[0]
    numData = x.shape[1]

    # Initialize weights randomly (Implement gradient descent)
    model = {}
    model['w'] = np.random.randn(numClass, numFeats)*0.01
    model['classLabels'] = classLabels

    return model
