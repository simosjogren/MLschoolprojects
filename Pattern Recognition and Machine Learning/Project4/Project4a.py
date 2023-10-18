# School work - Recurrent neural networks 1/3
# Simo Sjögren

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def main():

    ground_truth = np.loadtxt('detector_groundtruth.dat')
    # Edit: Fixed the ground_truth to be inverse.
    ground_truth = 1-ground_truth
    detector_output = np.loadtxt('detector_output.dat')

    thresholds = np.linspace(0, 1, 101)
    true_positive_rates = []
    false_positive_rates = []
    for threshold in thresholds:
        tp = ((detector_output >= threshold) & (ground_truth == 1)).sum()
        fn = ((detector_output < threshold) & (ground_truth == 1)).sum()
        tn = ((detector_output < threshold) & (ground_truth == 0)).sum()
        fp = ((detector_output >= threshold) & (ground_truth == 0)).sum()
        true_positive_rate = tp / (tp + fn)
        false_positive_rate = fp / (tn + fp)
        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)

    plt.plot(false_positive_rates, true_positive_rates)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    ## Current ROC curve is very bad with given data.

main()