import numpy as np

from multiclassLRTrain import multiclassLRTrain

def trainModel(x, y):
    param = {}
    param['lambda'] = 0.01      # Regularization term
    param['maxiter'] = 1000     # Number of iterations
    param['eta'] = 0.01         # Learning rate

    return multiclassLRTrain(x, y, param)
