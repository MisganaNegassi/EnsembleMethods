#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import unittest
import numpy as np
from ensemble.ensemble_selection.ensemble_selection import EnsembleSelection
from utils.constants import BINARY_CLASSIFICATION
from utils.basemodels_perf import basemodels_perf
from metrics import calculate_score, accuracy
from wrappers.library_pruning import prune
#from wrappers.library_bagging import bag
from data.load_data import load_data
from metrics import CLASSIFICATION_METRICS, calculate_score
from metrics.constants import STRING_TO_TASK_TYPES


def setUpGlobal(cls):
    a, b, c, d = load_data("./data/datasets/75191")
    cls.basemodels_preds = a
    cls.train_y = b
    cls.test_preds = c
    cls.test_y = d




def tearDownGlobal(cls):
    cls.library = None


class TestPruning(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)


    def tearDown(self):
        tearDownGlobal(self)

    def test_basemodels_performance(self):
        #basemodels_perf(self.basemodels_preds, self.train_y)
        pass

    def test_invalid_inputs(self):
        """checks if valueError is raised when input is invalid"""

        # es = EnsembleSelection(BINARY_CLASSIFICATION, accuracy)

    def test_small_valid_input(self):

        """ Checks if ensemble is initializated by sorting with respect to
            performace.For zero iterations, returns best performing model,
            for more iterations, initializes ensemble with n_init good
            performing models.
        # TODO: automate this to call instances of all classes and test
        # for all of them at once

        es = EnsembleSelection(BINARY_CLASSIFICATION, accuracy, no_iterations=100,
                               sorted_initialization=True, n_init=2)
        lb = LibraryPruning(es, 0.8, BINARY_CLASSIFICATION, accuracy)
        lb.fit(self.basemodels_preds, self.train_y)
        pred = lb.predict(self.test_preds)

        print("Test for Wrapper: Library_Pruning")
        print("######################################################")
        Accuracy = calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy)
        print("Accuracy:", Accuracy)
        self.assertGreaterEqual(Accuracy, 0.5)


        """
        pass

    def test_number_kept_models(self):
        """TODO: Docstring for test_kept_models.
        Tests if the sorting in library pruning sorts correctly with respect to
        metric
        Returns


        """

        task = "binary.classification"
        task_type = STRING_TO_TASK_TYPES[task]
        metric  = accuracy

        X_pruned, X_test_pruned = prune(X=self.basemodels_preds, X_test=self.test_preds, y=self.train_y, task_type=task_type, metric=metric, p=0.8)
        # XXX: Check that really approximately 80 % are kept
        n_kept_train = int(0.8 * self.basemodels_preds.shape[0])
        n_kept_test = int(0.8 * self.test_preds.shape[0])
        self.assertEqual(n_kept_train, X_pruned.shape[0])
        self.assertEqual(n_kept_test, X_test_pruned.shape[0])

    def test_kept_models_are_best(self):
        # XXX: Check that the models we keep are actually the ones with best performance

        task = "binary.classification"
        task_type = STRING_TO_TASK_TYPES[task]
        metric  = accuracy

        lst_perf = []
        for model in self.basemodels_preds:
            perf = calculate_score(solution=self.train_y, prediction=model, task_type=task_type, metric=metric)
            lst_perf.append(perf)
        np.argsort(self.basemodels_preds, kind=lst_perf)





class TestBagging(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)

    def tearDown(self):
        tearDownGlobal(self)

    def test_basemodels_performance(self):
        basemodels_perf(self.basemodels_preds, self.train_y)

    def test_invalid_inputs(self):
        """checks if valueError is raised when input is invalid"""

        # es = EnsembleSelection(BINARY_CLASSIFICATION, accuracy)

    def test_small_valid_input(self):

        """ Checks if ensemble is initializated by sorting with respect to
            performace.For zero iterations, returns best performing model,
            for more iterations, initializes ensemble with n_init good
            performing models.
        """
        # TODO: automate this to call instances of all classes and test
        # for all of them at once

        es = EnsembleSelection(BINARY_CLASSIFICATION, accuracy)
        bg = LibraryBagging(es)
        #bag_gen = bg.generate_random_bag(self.basemodels_preds)
        #next(bag_gen)
        bg.fit(self.basemodels_preds, self.train_y, self.test_preds)
        pred = bg.predict()

        print("Test for Wrapper: Library_Pruning")
        print("######################################################")
        Accuracy = calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy)
        print("Accuracy:", Accuracy)
        self.assertGreaterEqual(Accuracy, 0.5)
