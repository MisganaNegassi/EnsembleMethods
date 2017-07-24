#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-

"""
Unit tests for stacking with Multi-response linear regression algorithm.
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
from classifiers.stacking.stacking_MLR import StackingMLR
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, KFold
from utils.level1_dataset import compute_level1_dataset


def setUpGlobal(cls):
    iris = datasets.load_digits()
    X = iris.data
    y = iris.target

    X = iris.data
    y = iris.target
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
        st = StackingMLR(cvxopt=False)
        st.fit(self.base_models_predictions, self.train_y)
        pred = st.predict(self.test_predictions)
        print("ARGMAX PREDICTIONS: ", pred)
        #error metric is accuracy measure.
        print("Accuracy:", st.error_metric(pred, self.test_y))
        #self.assertGreaterEqual(st.error_metric(pred, self.test_y), 0.5)
        print (self.test_y)


def main():
    unittest.main()

if __name__ == "__main__":
    main()
