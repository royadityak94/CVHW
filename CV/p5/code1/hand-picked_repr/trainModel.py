import numpy as np

from multiclassLRTrain import multiclassLRTrain

def trainModel(x, y, x_val, y_val):
    param = {}
    param['lambda'] = 0.01      # Regularization term
    param['maxiter'] = 1000     # Number of iterations
    param['eta'] = 0.005         # Learning rate
    param['x_val'] = x_val
    param['y_val'] = y_val
    param['debug'] = False
    param['maxTol'] = 15
    
    return multiclassLRTrain(x, y, param)