#!/usr/bin/python3
# -*- encoding: utf-8 -*-

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import unittest
from sklearn import datasets
from metrics import calculate_score
from utils.level1_dataset import compute_level1_dataset


def setUpGlobal(cls):
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target
    a, b, c, d = compute_level1_dataset(X, y)
    cls.basemodels_preds = a
    cls.test_preds = b
    cls.train_y = c
    cls.test_y = d

def tearDownGlobal(cls):
    cls.library = None

class TestEnsembleSelectionSortedInitialization(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)

    def tearDown(self):
        tearDownGlobal(self)

    def test_invalid_inputs(self):
        """checks if valueError is raised when input is invalid"""
        calculate_score(self.train_y, self.basemodels_preds)

        self.assertRaises(ValueError, es.fit, self.validation_x,
                          self.validation_y)
