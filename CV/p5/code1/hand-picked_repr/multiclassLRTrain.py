import numpy as np

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.array([np.sum(exp, axis=1)]).T

def confusion_matrix(model, y):
        confusion_matrix = np.zeros((model['numClass'], model['numClass']))
        for itr in range(model['y_pred'].size):
            confusion_matrix[y[itr], model['y_pred'][itr]] += 1
        return confusion_matrix

def accuracy(model, X, y):
        y_pred = np.argmax(softmax(np.dot(X.T, model['w']) + model['b']), axis=1)
        model['y_pred'] = y_pred
        accuracy = np.sum([y_pred[i] == y[i] for i in range(y.size)]) / y.size
        return model, accuracy

def multiclassLRTrain(x, y, param):

    classLabels = np.unique(y)
    numClass = classLabels.shape[0]
    numFeats = x.shape[0]
    numData = x.shape[1]
    

    # Initialize weights randomly (Implement gradient descent)
    model = {}
    model['classLabels'] = classLabels
    model['numClass'] = numClass
    model['w'] = np.random.uniform(low=-0.01, high=.01, size=(x.shape[0], numClass))
    model['b'] = np.random.uniform(low=-0.01, high=.01)
    model['lastAccuracy'] = -1
    model['debug'] = param['debug']
    model['maxTol'] = param['maxTol']
    y_categorical = np.eye(model['numClass'])[y] 
    curTol = 0
        
    for itr in range(param['maxiter']):
        prediction = softmax(np.dot(x.T, model['w']))
        error = y_categorical - prediction
        gradient = np.dot(x, error)
        model['w'] += param['eta'] * (gradient - (param['lambda'] * model['w']))
        model['b'] -= param['eta'] * np.sum(error)

        model, curr_accuracy = accuracy(model, param['x_val'], param['y_val'])
        if curr_accuracy != model['lastAccuracy']:
            model['lastAccuracy'] = curr_accuracy
        else:
            curTol += 1
            if curTol >= model['maxTol']:
                return model
    return model