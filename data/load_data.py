#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import glob
import os
import numpy as np


def load_data(path):
    """
    Loads data from a specified path.
    Returns: numpy array to be used as training set

    """

    #load train files
    L1data = iter(sorted(glob.glob(os.path.expanduser(os.path.join(path, 'predictions_ensemble/*.npy'))),key=os.path.basename))

    X = np.asarray([np.load(array) for array in L1data])

    y = np.load((os.path.expanduser(os.path.join(path, 'true_targets_ensemble.npy'))))


    #load test files
    L1test = iter(sorted(glob.glob(os.path.expanduser(os.path.join(path,'predictions_test_2/*model')))))
    X_test_matrix = np.asarray([np.load(array) for array in L1test])

    X_test = []
    for i in range(X_test_matrix.shape[0]):
        X_test.append(np.column_stack((1 - X_test_matrix[i,:], X_test_matrix[i,:])))

    X_test = np.asarray(X_test)
    y_test = np.load(os.path.expanduser(os.path.join(path, 'y_test.npy')))
    return X, y, X_test, y_test






