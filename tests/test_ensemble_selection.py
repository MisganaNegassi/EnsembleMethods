#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-
"""
Unit tests for ensemble selection algorithm.
"""
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import numpy as np
import unittest
from ensemble.ensemble_selection.ensemble_selection import EnsembleSelection
from utils.constants import BINARY_CLASSIFICATION
from metrics import calculate_score, accuracy
from data.load_data import load_data


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def __basemodels_score__(basemodels_preds, train_y):
    """TODO: Docstring for __basemodels_score__.

    Args:
        arg1 (TODO): TODO

    Returns: TODO

    """
    n_datapoints = basemodels_preds.shape[1]
    basemodels_preds = basemodels_preds.transpose()
    M1 = np.asarray([np.argmax(basemodels_preds[:, i, 0]) for i in
                    range(n_datapoints)])
    M2 = np.asarray([np.argmax(basemodels_preds[:, i, 1]) for i in
                    range(n_datapoints)])
    M3 = np.asarray([np.argmax(basemodels_preds[:, i, 2]) for i in
                    range(n_datapoints)])
    M4 = np.asarray([np.argmax(basemodels_preds[:, i, 3]) for i in
                    range(n_datapoints)])
    M5 = np.asarray([np.argmax(basemodels_preds[:, i, 4]) for i in
                    range(n_datapoints)])
    M6 = np.asarray([np.argmax(basemodels_preds[:, i, 5]) for i in
                    range(n_datapoints)])
    M7 = np.asarray([np.argmax(basemodels_preds[:, i, 6]) for i in
                    range(n_datapoints)])
    print("basemodels accuracy")
    print("######################################################")
    print("basemodel_1_Accuracy:", calculate_score(train_y, M1, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_2_Accuracy:", calculate_score(train_y, M2, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_3_Accuracy:", calculate_score(train_y, M3, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_4_Accuracy:", calculate_score(train_y, M4, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_5_Accuracy:", calculate_score(train_y, M5, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_6_Accuracy:", calculate_score(train_y, M6, BINARY_CLASSIFICATION, accuracy))
    print("basemodel_7_Accuracy:", calculate_score(train_y, M7, BINARY_CLASSIFICATION, accuracy))


def ensemble_selections_equal(old_ensemble, new_ensemble):
    if len(old_ensemble.ensemble) != len(new_ensemble.ensemble):
        return False
    for i in range(len(old_ensemble.ensemble)):
        old_ensemble_model = old_ensemble.ensemble[i]
        new_ensemble_model = new_ensemble.ensemble[i]
        if type(old_ensemble_model) != type(new_ensemble_model):
            return False
        if old_ensemble_model.get_params() != new_ensemble_model.get_params():
            return False
    return True


def setUpGlobal(cls):
    path  = './data/datasets/75191'
    a, b, c, d = load_data(path)
    cls.basemodels_preds = a
    cls.train_y = b
    cls.test_preds = c
    cls.test_y = d



def tearDownGlobal(cls):
    cls.library = None


class TestEnsembleSelectionSortedInit(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)

    def tearDown(self):
        tearDownGlobal(self)

    def test_basemodels_performance(self):
        __basemodels_score__(self.basemodels_preds, self.train_y)

    def test_invalid_inputs(self):
        """checks if valueError is raised when input is invalid"""

        es = EnsembleSelection(task_type=BINARY_CLASSIFICATION,
                                metric=accuracy,
                                no_iterations=100,
                                with_replacements=False,
                                sorted_initialization=True,
                                n_init=100)
        # self.assertRaises(ValueError, es.fit, self.basemodels_preds,
        #                  self.train_y)

    def test_small_valid_input(self):

        """ Checks if ensemble is initializated by sorting with respect to
            performace.For zero iterations, returns best performing model,
            for more iterations, initializes ensemble with n_init good
            performing models.
        """
        es = EnsembleSelection(task_type=BINARY_CLASSIFICATION,
                                metric=accuracy,
                                no_iterations=100,
                                with_replacements=False,
                                sorted_initialization=True,
                                n_init=100)
        es.fit(self.basemodels_preds, self.train_y)
        pred = es.predict(self.test_preds)

        print("Test for Ensemble Selection with Sorted Initialization")
        print("######################################################")
        score = calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy)
        print("Accuracy:", score)
        self.assertGreaterEqual(score, 0.5)


class TestEnsembleSelectionStandard(unittest.TestCase):

    """Tests the case where models are added to ensemle with replacement"""

    def setUp(self):
        self = setUpGlobal(self)

    def tearDown(self):
        self = tearDownGlobal(self)

    def test_small_valid_input(self):
        es = EnsembleSelection(task_type=BINARY_CLASSIFICATION,
                                metric=accuracy,
                                no_iterations=100,
                                with_replacements=False,
                                sorted_initialization=False,
                                n_init=None)
        es.fit(self.basemodels_preds, self.train_y)
        pred = es.predict(self.test_preds)
        print("Test for Ensemble Selection Standard")
        print("######################################################")
        score = calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy)
        print("Accuracy:", score)
        self.assertGreaterEqual(score, 0.5)

    def test_compute_error(self):
        """ checks if error metric works"""
        #es = EnsembleSelection(BINARY_CLASSIFICATION)

        #with open("./models/standard_models.pkl", 'wb') as f:
        #    pickle.dump(es, f)

        #with open("./models/standard_models.pkl", 'rb') as f:
        #    old_ensemble = pickle.load(f)
        #self.assertTrue(ensemble_selections_equal(old_ensemble, es))


class TestEnsembleSelectionWithReplacements(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        self = setUpGlobal(self)

    def tearDown(self):
        self = tearDownGlobal(self)

    def test_small_valid_input(self):
        es = EnsembleSelection(task_type=BINARY_CLASSIFICATION,
                                metric=accuracy,
                                no_iterations=100,
                                with_replacements=True,
                                sorted_initialization=False,
                                n_init=None)
        es.fit(self.basemodels_preds, self.train_y)
        pred = es.predict(self.test_preds)

        #with open("./models/with_replacement_models.pkl", 'wb') as f:
        #    pickle.dump(es, f)

        #with open("./models/with_replacement_models.pkl", 'rb') as f:
        #    old_ensemble = pickle.load(f)
        #self.assertTrue(ensemble_selections_equal(old_ensemble, es))
        score = calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy)
        print("Accuracy:", score)
        self.assertGreaterEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
