# School project - Trying to establish the best possible neural network from given dataset-items.
# Simo Sjögren

from email.policy import default
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import os
import skimage
from scipy.stats import multivariate_normal
from sklearn.metrics import classification_report
from sqlalchemy import true

import tensorflow as tf
import keras

colors = ["red", "green", "blue"]
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


def load_test_data(CIFAR_PATH):
    test_path = CIFAR_PATH + r'\test_batch'
    datadict = unpickle(test_path)
    X_test = datadict["data"]
    Y_test = datadict["labels"]
    X_test = np.array(X_test).astype('int32')
    Y_test = np.array(Y_test)
    return X_test, Y_test

def compile_to_onehot(Y_vector, amount_of_classes = 10):
    Y_new = []
    # Assuming that the items can be converted into ints.
    # Manual method for converting to one-hot.
    default_vector = np.zeros((amount_of_classes,), dtype=int)
    for index in Y_vector:
        default_vector_to_be_added = np.array(default_vector, copy=True)
        default_vector_to_be_added[int(index)] = 1
        Y_new.append(default_vector_to_be_added)
    return np.array(Y_new)


def load_data_batch(CIFAR_PATH, amount_of_batches=5):
    training_data = []
    training_labels = []
    for i in range(1,amount_of_batches+1):

        raw_data = unpickle(CIFAR_PATH + r'\data_batch_' + str(i))
        training_data.append(raw_data["data"])
        training_labels.append(raw_data["labels"])
        print(str(i/(amount_of_batches) * 100), '%')
    X_test = np.concatenate(training_data)
    X_train = X_test.astype('int32')
    Y_train = np.concatenate(training_labels)

    return X_train, Y_train


def print_accuracy(accuracy):
    accuracy = accuracy * 100
    return f"The classification accuracy is {accuracy:.2f} %"


def class_acc(pred, gt):
    num_correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            num_correct += 1
    accuracy = (num_correct/float(len(pred)))
    # Accuracy is in decimal form.
    print(print_accuracy(accuracy))
    return accuracy


def cifar10_classifier_random(X_test):
    random_label=[]
    for i in range(X_test.shape[0]):
        random_numbers = np.random.randint(9)
        random_label.append(random_numbers)
    return random_label


def cifar10_classifier_1nn(X_test, tr_data, tr_labels):
    x_label = []
    for x in range(0, 10000):
        dist = []
        for j in range(len(tr_data)):
            dist.append(np.sum(np.subtract(tr_data[j], X_test[x]) ** 2))
        dist = np.array(dist)
        test_label = tr_labels[dist.argmin()]
        x_label.append(test_label)
        print(str((len(x_label) / 10000) * 100) + ' %')
    return x_label


def test_learning_rate(X_train, Y_train, X_test, Y_test, BATCH_SIZE, EPOCHS, starting_rate = 0.00001):
    current_rate = starting_rate
    acc_list = np.array([])
    lr_list = np.array([])
    for idx in range(5):
        model = initialize_neuro_network_v1(lrate=current_rate)
        model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        loss, acc = model.evaluate(X_test, Y_test)
        acc_list = np.append(acc_list, acc)
        lr_list = np.append(lr_list, current_rate)
        current_rate = current_rate * 10
    plt.plot(lr_list, acc_list)
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.show()
    print(lr_list[np.argmax(acc_list)])
    return lr_list[np.argmax(acc_list)]


def test_epoch_size(X_train, Y_train, X_test, Y_test, BATCH_SIZE, LRATE):
    acc_list = np.array([])
    epoch_list = np.array([5, 10, 20, 40, 60, 80, 100])
    model = initialize_neuro_network_v1(lrate=LRATE)
    for epoch in epoch_list:
        model.fit(X_train, Y_train, epochs=epoch, batch_size=BATCH_SIZE, verbose=1)
        loss, acc = model.evaluate(X_test, Y_test)
        acc_list = np.append(acc_list, acc)
    plt.plot(epoch_list, acc_list)
    # plt.xscale('log')
    plt.xlabel('Epoch size')
    plt.ylabel('Accuracy')
    plt.show()


def initialize_neuro_network_v1(lrate = 0.01):
    # First iteration of NN 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, input_dim=3072, activation='sigmoid'))
    opt = keras.optimizers.RMSprop(learning_rate=lrate)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def initialize_neuro_network_v2(lrate = 0.00001):
    # Second iteration of NN 
    tfk = tf.keras
    model = tf.keras.models.Sequential([
        tfk.layers.Flatten(input_shape=(32,32,3)),
        tfk.layers.Dense(3000, activation='relu'),
        tfk.layers.Dense(1000, activation='relu'),
        tfk.layers.Dense(10, input_dim=3072, activation='sigmoid')
        # Dropout prevents overfitting, because it ignores now 25% of the layers.
        # tfk.layers.Dropout(0.25)
    ])

    '''
    Categorical crossentropy means the one-hots. 
    Sparse categorical crossentropy means basic int values.
    'SGD' stands for Gradient Descent and RMSprop for the RMSprop algoritm(?)
    RMSprop gives us 10% on 1/5 epoch run with lrate=0.1, meanwhile SGD givers 28%.
    When lrate = 0.00001, accuracy is over 35% (all the way to 47%) and not even that slower.
    '''
    opt = keras.optimizers.RMSprop(learning_rate=lrate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model


def initialize_neuro_network_v3(lrate = 0.00001):
    ## Third iteration of NN
    '''
    This version gives about 70% accuracy quite fast.

    In this model we changed the activation function from sigmoid to softmax, 
    which normalizes the activations differently (normalizes by divinding with all
    different weights.)

    Filters in convolution are for detecting 32 different features from images.
    Kernel size means the size of the filter.
    '''
    tfk = tf.keras
    model = tf.keras.models.Sequential([

        # Convolution layer 1
        tfk.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
        tfk.layers.MaxPooling2D((2,2)),

        # Convolution layer 2
        tfk.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
        tfk.layers.MaxPooling2D((2,2)),

        # This flatten makes 32,32,3 into 3072.
        # One additional layer with 32 units, the accuracy decreases a bit.
        tfk.layers.Flatten(),
        tfk.layers.Dense(64, activation='relu'),
        tfk.layers.Dense(10, input_dim=3072, activation='softmax')
        # Dropout prevents overfitting, because it ignores now 25% of the layers.
    ])

    '''
    Ive read that adam is the most used and best optimizer.
    SGD gave only 56% with 5 epoch, so we switch to adam.
    Also turns out, that if we modify the learning rate of adam to very low (0.00001), we get very low accuracies.
    '''
    # opt = keras.optimizers.Adam(learning_rate=lrate)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def prepare_data(X_train, X_test):
    # Lets convert 3072 -> 32x32x3
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    # Then we do the normalization (this actually increases the overall accuracy.)
    X_train = X_train / 255
    X_test = X_test / 255
    return X_train, X_test


def main():
    '''
    This implementation gives relatively good accuracies with relatively low amount of work 
    Only five epochs gives about 65% with small neural network (64 and 10) quite fast (only 5 minutes on my computer).
    '''
    print("Loading batches...")
    CIFAR_PATH = r'C:\Users\simos\Desktop\MRANDPR\cifar-10-batches-py'
    X_test,Y_test = load_test_data(CIFAR_PATH)
    X_train,Y_train = load_data_batch(CIFAR_PATH, amount_of_batches = 5 )
    # label_names = unpickle(CIFAR_PATH + r'\batches.meta')

    EPOCHS = 5
    # LRATE = 0.00001
    # BATCH_SIZE = 30
    # DECAY = None

    Y_train = tf.one_hot(Y_train.astype(np.int32), depth=10)
    Y_test = tf.one_hot(Y_test.astype(np.int32), depth=10)

    X_train, X_test = prepare_data(X_train, X_test)

    model = initialize_neuro_network_v3()
    model.fit(X_train, Y_train, epochs=EPOCHS, verbose=1)
    value, acc = model.evaluate(X_test, Y_test)
    print(print_accuracy(acc))

    # Test for learning rate in logarithmic scale.
    # test_learning_rate(X_train, Y_train, X_test, Y_test, BATCH_SIZE, EPOCHS)

    # Test for epoch size.
    # test_epoch_size(X_train, Y_train, X_test, Y_test, BATCH_SIZE, LRATE)


if __name__ == "__main__":
    main()
