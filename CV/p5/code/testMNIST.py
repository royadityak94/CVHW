import numpy as np
import os
import utils
import time

from montageDigits import montageDigits
from extractDigitFeatures import extractDigitFeatures
from trainModel import trainModel
from evaluateLabels import evaluateLabels
from evaluateModel import evaluateModel

# There are three versions of MNIST dataset
dataTypes = ['digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat']

# You have to implement three types of features
featureTypes = ['pixel', 'hog', 'lbp']

# Accuracy placeholder
accuracy = np.zeros((len(dataTypes), len(featureTypes)))
trainSet = 1
testSet = 2

for i in xrange(len(dataTypes)):
    dataType = dataTypes[i]
    #Load data
    path = os.path.join('..', 'data', dataType)
    data = utils.loadmat(path)
    print '+++ Loading digits of dataType: {}'.format(dataType)

    # Optionally montage the digits in the val set
    #montageDigits(data['x'][:, :, data['set']==2])

    for j in xrange(len(featureTypes)):
        featureType = featureTypes[j]

        # Extract features
        tic = time.time()
        features = extractDigitFeatures(data['x'], featureType)
        print '{:.2f}s to extract {} features for {} images'.format(time.time()-tic,
                featureType, features.shape[1])

        # Train model
        tic = time.time()
        model = trainModel(features[:, data['set']==trainSet], data['y'][data['set']==trainSet])
        print '{:.2f}s to train model'.format(time.time()-tic)

        # Test the model
        ypred = evaluateModel(model, features[:, data['set']==testSet])
        y = data['y'][data['set']==testSet]

        # Measure accuracy
        (acc, conf) = evaluateLabels(y, ypred, False)
        print ' Accuracy [testSet={}] {:.2f}\n'.format(testSet, acc*100)
        accuracy[i, j] = acc


# Print the results in a table
print '+++ Accuracy Table [trainSet={}, testSet={}]'.format(trainSet, testSet)
print '--------------------------------------------------'
print 'dataset\t\t\t',
for j in xrange(len(featureTypes)):
    print '{}\t'.format(featureTypes[j]),

print ''
print '--------------------------------------------------'
for i in xrange(len(dataTypes)):
    print '{}\t'.format(dataTypes[i]),
    for j in xrange(len(featureTypes)):
        print '{:.2f}\t'.format(accuracy[i, j]*100),
    print ''

# Once you have optimized the hyperparameters, you can report test accuracy
# by setting testSet=3. You should not optimize your hyperparameters on the
# test set. That would be cheating.
