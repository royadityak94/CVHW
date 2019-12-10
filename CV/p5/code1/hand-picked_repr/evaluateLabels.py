import numpy as np
import matplotlib.pyplot as plt

def evaluateLabels(y, ypred, visualize=False):

    classLabels = np.unique(y)
    conf = np.zeros((len(classLabels), len(classLabels)))
    for tc in range(len(classLabels)):
        for pc in range(len(classLabels)):
            conf[tc, pc] = np.sum(np.logical_and(y==classLabels[tc], 
                ypred==classLabels[pc]).astype(float))
    
    acc = np.sum(np.diag(conf))/y.shape[0]

    if visualize:
        plt.figure()
        plt.imshow(conf, cmap='gray')
        plt.ylabel('true labels')
        plt.xlabel('predicted labels')

    return (acc, conf)
