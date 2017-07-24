#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import numpy as np
from metrics import calculate_score


def prune(X, y, X_test, task_type, metric, p=0.8):

    """
    Preprocessing method for ensemble methods. Takes instance of model and percentage
    of models to be pruned as input. It computes the performance of the base
    models and sorts them. And then takes a p% of those sorted models as
    input to the ensemble.

    Parameters
    ----------
    X : {array-like} of shape = [n_basemodels, n_datapoints, n_targets]
        Predicted class probabilites of base models trained on level-zero validation data
    y : array-like, shape = [n_datapoints]
        The predicted classes.
    X_test: array-like,  of shape = [n_basemodels, n_datapoints, n_targets]
        Predicted class probabilities on level-zero test data

    Returns
    -------
    pruned training set and pruned test set

    """
    n_basemodels = X.shape[0]
    N = int(p * n_basemodels)
    perf = np.zeros(n_basemodels)
    for idx, basemodel in enumerate(X):
        perf[idx] = calculate_score(y, basemodel, task_type, metric)


    indices = np.argsort(perf)[-N:]
    X_pruned = X[indices]
    X_test_pruned = X_test[indices]

    return X_pruned, X_test_pruned


