import numpy as np
import utils
from utils import loadmat
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.utils import to_categorical


def get_random_data(data1, data2, low, high, max_samples=100):
    N, H1, W1, C1 = data1.shape
    suff_data1 = np.zeros((max_samples, H1, W1, C1))
    suff_data2 = np.zeros((max_samples,))
    shuffles = np.random.randint(low, high+1, max_samples)
    for idx in range(shuffles.shape[0]):
        suff_data1[idx] = data1[idx, :, :, :]
        suff_data2[idx] = data2[idx]
    return suff_data1, suff_data2

def reformat(x):
    new_x = np.zeros((x.shape[2], x.shape[0], x.shape[1]))
    for i in range(x.shape[2]):
        new_x[i, :] = x[:, :, i]
    return new_x[:, :, :, np.newaxis]

def return_datasets(filename):
    data = utils.loadmat('../data/{}'.format(filename))
    trainSet, testSet, valSet = 1, 2, 3
    
    x_train = reformat(data['x'][:, :, data['set']==trainSet ])
    y_train = (data['y'][data['set']==trainSet])
    x_val = reformat(data['x'][:, :, data['set']==valSet])
    y_val = (data['y'][data['set']==valSet])
    x_test = reformat(data['x'][:, :, data['set']==testSet])
    y_test = (data['y'][data['set']==testSet])
    
    x_train, y_train = get_random_data(x_train, y_train, 0, x_train.shape[0], x_train.shape[0])
    x_val, y_val = get_random_data(x_val, y_val, 0, x_val.shape[0], x_val.shape[0])
    x_test, y_test = get_random_data(x_test, y_test, 0, x_test.shape[0], x_test.shape[0])
    
    return (x_train, y_train, x_val, y_val, x_test, y_test)


def extract_model(activation_func = 'relu'):
    model = Sequential()
    kernel_size=(5)
    stride_size = (2)

    model.add(Conv2D(32, kernel_size=kernel_size, strides=stride_size, activation=activation_func, padding='same'))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=kernel_size, strides=stride_size, activation=activation_func, padding='same'))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=kernel_size, strides=stride_size, activation=activation_func, padding='same'))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())
   
    model.add(Conv2D(96, kernel_size=kernel_size, strides=stride_size, activation=activation_func, padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation=activation_func)) 
    model.add(Dense(10, activation='softmax'))
    
    return model


if __name__ == '__main__':
    NUM_CLASSES = 10
    MAX_EPOCHS, MAX_BATCH_SIZE = 25, 25
    files = ['digits-jitter.mat', 'digits-normal.mat', 'digits-scaled.mat'] 
    for file in files:
        model = extract_model('relu')

        if file == 'digits-jitter.mat': MAX_EPOCHS, MAX_BATCH_SIZE = 25, 10
        elif file == 'digits-scaled.mat': model = extract_model('selu')
        x_train, y_train, x_val, y_val, x_test, y_test = return_datasets(file)
        model.compile(loss="categorical_crossentropy", optimizer='Adam', validation_data=(x_val, to_categorical(y_val, num_classes=NUM_CLASSES)), metrics=["accuracy"])
        model.fit(x_train, to_categorical(y_train, num_classes=NUM_CLASSES), epochs=MAX_EPOCHS, batch_size=MAX_BATCH_SIZE, verbose=0)
        accuracy = model.evaluate(x_test, to_categorical(y_test, num_classes=10), verbose=0)[1]
        print ("File ={}, Accuracy = {}".format(file, accuracy))