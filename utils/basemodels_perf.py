#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import numpy as np
from metrics import calculate_score, accuracy
from utils.constants import BINARY_CLASSIFICATION


def basemodels_perf(X, y):
    """TODO: Docstring for __basemodels_score__.

    Args:
        arg1 (TODO): TODO

    Returns: TODO

    """
    n_datapoints = X.shape[1]
    X = X.transpose()
    M1 = np.asarray([np.argmax(X[:, i, 0]) for i in
                    range(n_datapoints)])
    M2 = np.asarray([np.argmax(X[:, i, 1]) for i in
                    range(n_datapoints)])
    M3 = np.asarray([np.argmax(X[:, i, 2]) for i in
                    range(n_datapoints)])
    M4 = np.asarray([np.argmax(X[:, i, 3]) for i in
                    range(n_datapoints)])
    M5 = np.asarray([np.argmax(X[:, i, 4]) for i in
                    range(n_datapoints)])
    M6 = np.asarray([np.argmax(X[:, i, 5]) for i in
                    range(n_datapoints)])
    M7 = np.asarray([np.argmax(X[:, i, 6]) for i in
                    range(n_datapoints)])
    print("basemodels accuracy")
    print("######################################################")
    print("basemodel_1_Accuracy:", calculate_score(y, M1, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_2_Accuracy:", calculate_score(y, M2, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_3_Accuracy:", calculate_score(y, M3, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_4_Accuracy:", calculate_score(y, M4, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_5_Accuracy:", calculate_score(y, M5, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_6_Accuracy:", calculate_score(y, M6, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_7_Accuracy:", calculate_score(y, M7, BINARY_CLASSIFICATION, accuracy))
