import numpy as np

def multiclassLRPredict(model, x):
    numData = x.shape[1]
    
    # Simply predict the first class (Implement this)
    ypred = model['classLabels'][0]*np.ones(numData)
    return ypred
