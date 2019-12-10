import numpy as np

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.array([np.sum(exp, axis=1)]).T

def multiclassLRPredict(model, x):
    ypred = np.argmax(softmax(np.dot(x.T, model['w']) + model['b']), axis=1)
    return ypred