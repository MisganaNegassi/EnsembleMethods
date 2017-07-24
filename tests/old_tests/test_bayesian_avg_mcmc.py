#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-

"""
Unit tests for Bayesian Model Averaging.
"""
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4


import sys
sys.path.append("..")
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import unittest
import numpy as np
from classifiers.stacking.stacking_bayesian_avg_mcmc import BayesianAverageMCMC
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from utils.level1_dataset import compute_level1_dataset

def setUpGlobal(cls):
    #dataset = datasets.load_breast_cancer()
    dataset = datasets.load_digits()
    X = dataset.data
    y = dataset.target
    a, b, c, d = compute_level1_dataset(X, y)
    cls.base_models_predictions = a
    cls.test_predictions = b
    cls.train_y = c
    cls.test_y = d


def tearDownGlobal(cls):
    cls.library = None
    return cls


class TestStacking_MLR(unittest.TestCase):

    def setUp(self):
        self = setUpGlobal(self)

    def tearDown(self):
        self = tearDownGlobal(self)

    #test input format
    def test_valid_input(self):
        ba = BayesianAverageMCMC()
        ba.fit(self.base_models_predictions, self.train_y)
        pred = ba.predict(self.test_predictions)
        #error metric is accuracy measure.

        n_datapoints = self.base_models_predictions.shape[1]
        M1 = [np.argmax(self.base_models_predictions[:, i, 0]) for i in
              range(n_datapoints)]
        M2 = [np.argmax(self.base_models_predictions[:, i, 1]) for i in
              range(n_datapoints)]
        M3 = [np.argmax(self.base_models_predictions[:, i, 2]) for i in
              range(n_datapoints)]
        M1 = np.asarray(M1)
        M2 = np.asarray(M2)
        M3 = np.asarray(M3)
        print(M2[:9], self.train_y[:9])
        print(M1[:9], self.train_y[:9])
        print("M1:", np.asarray(M2).shape, "Y:", self.train_y.shape)
        print("basemodel_1_Accuracy:", ba.error_metric(M1, self.train_y))
        print("basemodel_2_Accuracy:", ba.error_metric(M2, self.train_y))
        print("basemodel_3_Accuracy:", ba.error_metric(M3, self.train_y))
        print("Accuracy:", ba.error_metric(pred, self.test_y))
        self.assertGreaterEqual(ba.error_metric(pred, self.test_y), 0.5)
        #print self.test_y


def main():
    unittest.main()

if __name__ == "__main__":
    main()
