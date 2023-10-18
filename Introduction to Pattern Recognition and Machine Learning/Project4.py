# School project: CIFAR-10 Color classifier
# Simo Sjögren

import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from numba import jit
import os
import skimage
import statistics
from scipy.stats import multivariate_normal

colors = ["red", "green", "blue"]
CIFAR_PATH = r'C:\Users\simos\Desktop\MRANDPR\cifar-10-batches-py'


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


def load_data_batch(CIFAR_PATH, range_num=6):
    training_data = []
    training_labels = []
    for i in range(1,range_num):

        raw_data = unpickle(CIFAR_PATH + r'\data_batch_' + str(i))
        training_data.append(raw_data["data"])
        training_labels.append(raw_data["labels"])
        print(str(i/(range_num-1) * 100), '%')
    X_test = np.concatenate(training_data)
    X_train = X_test.astype('int32')
    Y_train = np.concatenate(training_labels)
    return X_train, Y_train


def class_acc(pred, gt):
    num_correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            num_correct += 1
    accuracy = (num_correct/float(len(pred)))*100
    print(f"The classification accuracy is {accuracy:.2f} %")
    return accuracy


def cifar10_classifier_random(X_test):
    random_label=[]
    for i in range(X_test.shape[0]):
        random_numbers = np.random.randint(9)
        random_label.append(random_numbers)
    return random_label


@jit
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


def custom_maximum_resolver(list_of_multivariates):
    # Finds the highest given multivariate-index from a list and returns it.
        current_index_and_value = [0, 0]
        for index in range(0, len(list_of_multivariates)):
            if list_of_multivariates[index] >= current_index_and_value[1]:
                current_index_and_value = [index, list_of_multivariates[index]]
        return current_index_and_value[0]


def cifar10_color(X):
    # Rescales any given image to 1x1 image.
    size = X.shape[0]
    Xp = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
    return skimage.transform.resize(Xp, (size,3))


def cifar10_custom_size_color(X, custom_size):
    # Same as cifar10_color but with customized pixel amount.
    size = X.shape[0]
    print(size)
    custom_size = custom_size * 3
    print(custom_size)
    Xp = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
    print("Kusee varmaan skimageen?")
    print(Xp.shape)
    return skimage.transform.resize(Xp, (size, custom_size))


def cifar_10_naivebayes_learn(Xp, Y, return_only_classlist=False):

    def class_list_maker():
        '''
        Returns a dict, which includes all the color values for every color defined in colors-list.
        For example, class_key_values["1"] = {'red' : [], 'green' : [], 'blue' : []}
        '''
        if (Xp.shape[0] != Y.shape[0]):
            print("Xp and Y are different sizes: " + str(Xp.shape[0]) + ',' + str(Y.shape[0]))
            return
        class_key_values = {}
        for pic_index in range(0, len(Y)):
            try:
                for color_index in range(0,len(colors)):
                    # For every color, it adds its mean color code from the cifar10_color.
                    class_key_values[str(Y[pic_index])][colors[color_index]].append(Xp[pic_index][color_index])
            except KeyError:
                class_key_values[str(Y[pic_index])] = {}
                for temp_color in colors:
                    class_key_values[str(Y[pic_index])][temp_color] = []
                for color_index in range(0,len(colors)):
                    # For every color, it adds its mean color code from the cifar10_color.
                    class_key_values[str(Y[pic_index])][colors[color_index]].append(Xp[pic_index][color_index])
        return class_key_values
    
    # Start of the function.
    class_key_values = class_list_maker()
    if return_only_classlist:
        return class_key_values
    class_key_size = len(Y)
    means = []
    variances = []
    probabilities = []
    for index in range(0, len(class_key_values)):
        means_for_index = []
        variances_for_index = []
        for color in colors:
            means_for_index.append(sum(class_key_values[str(index)][color]) / len(class_key_values[str(index)][color]))
            variances_for_index.append(statistics.variance(class_key_values[str(index)][color]))
        means.append(means_for_index)
        variances.append(variances_for_index)
        # Next we count the probability of certain color by the amount of its first colors.
        probabilities.append(len(class_key_values[str(index)][colors[0]]) / class_key_size)
    return means, variances, probabilities


def cifar10_classifier_naivebayes(x,mu,sigma,p):
    # Bayesian classifier for cifar-10, but with only one variance dimension.
    list_of_multivariates = []
    # First we have only the nominators without probability.
    for index in range(0, len(p)):
        multivariate = multivariate_normal.pdf(x, mu[index], sigma[index])
        list_of_multivariates.append(multivariate)
    sum_of_multivariates = sum(list_of_multivariates)
    # Second loop adds the probability to the nominator and divides by the sum of all "multivariates".
    for index in range(0, len(p)):
        list_of_multivariates[index] = (list_of_multivariates[index] * p[index]) / sum_of_multivariates
    return custom_maximum_resolver(list_of_multivariates)


def cifar_10_bayes_learn(Xf,Y):
    # Bayesian learner for cifar-10 data.
    means = np.empty((0, Xf.shape[1]))
    variances = np.empty((0, Xf.shape[1], Xf.shape[1]))
    probabilities = []
    classes = []
    for i in range(0, 10):
        temp = np.where(Y == i)
        new_Xf = Xf[temp[0], :]
        classes.append(new_Xf)
    for j in classes:
        means = np.append(means, [np.mean(j, axis=0)], axis=0)
        variances = np.append(variances, [np.cov(np.transpose(j))], axis=0)
        probabilities.append(len(j) / len(Xf))
    return means, variances, probabilities


def cifar10_classifier_bayes(x,mu,sigma,p):
    # Bayesian classifier for 3D variances.
    list_of_multivariates = []
    for index in range(0, len(p)):
        multivariate = multivariate_normal.pdf(x, mu[index], sigma[index])
        list_of_multivariates.append(multivariate)
    sum_of_multivariates = sum(list_of_multivariates)
    # Second loop adds the probability to the nominator and divides by the sum of all "multivariates".
    for index in range(0, len(p)):
        list_of_multivariates[index] = (list_of_multivariates[index] * p[index]) / sum_of_multivariates
    return custom_maximum_resolver(list_of_multivariates)


def main():

    print("Loading batches...")
    X_test,Y_test = load_test_data(CIFAR_PATH)
    label_names = unpickle(CIFAR_PATH + r'\batches.meta')

    X_train,Y_train = load_data_batch(CIFAR_PATH, range_num=6)

    X = X_train.reshape(X_train.shape[0], 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32).transpose(0,2,3,1).astype("uint8")

    print("Using NAIVE bayes with 1x1 images:")
    X_custom = cifar10_custom_size_color(X, 1)
    X_test_custom = cifar10_custom_size_color(X_test, 1)
    means, variances, probabilities = cifar_10_naivebayes_learn(X_custom, Y_train)
    Y_guess = []
    for idx in range(0, len(Y_test)):
        Y_guess.append(cifar10_classifier_bayes(X_test_custom[idx], means, variances, probabilities))
    class_acc(Y_guess, Y_test)
    print()

    print("Using REGULAR bayes with 1x1 images:")
    means, variances, probabilities = cifar_10_bayes_learn(X_custom, Y_train)
    Y_guess = []
    for idx in range(0, len(Y_test)):
        Y_guess.append(cifar10_classifier_bayes(X_test_custom[idx], means, variances, probabilities))
    class_acc(Y_guess, Y_test)
    print()

    print("Using REGULAR bayes with scaled image sizes:")
    sizes = [1,2,4,6,8,16,24,32]
    values_for_sizes = []
    for custom_size in sizes:
        X_custom = cifar10_custom_size_color(X, custom_size)
        X_test_custom = cifar10_custom_size_color(X_test, custom_size)
        means, variances, probabilities = cifar_10_bayes_learn(X_custom, Y_train)
        Y_guess = []
        for idx in range(0, len(Y_test)):
            Y_guess.append(cifar10_classifier_bayes(X_test_custom[idx], means, variances, probabilities))
        print(f"Using bayes with {custom_size}x{custom_size} images:")
        values_for_sizes.append(class_acc(Y_guess, Y_test))
    plt.plot(sizes, values_for_sizes)
    plt.ylabel('Prediction accuracy percent')
    plt.xlabel('Number of color pixels')
    plt.show()

if __name__ == "__main__":
    main()