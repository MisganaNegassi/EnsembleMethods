#!/usr/bin/env/python3
# -*- coding: iso-8859-15 -*-


"""
Unit tests for stacking with unregularized logistic regression algorithm.
"""
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import unittest
import numpy as np
from timeit import default_timer
from ensemble.stacking.stacking_logit_reg import StackingLogitReg
from ensemble.stacking.stacking_logit import StackingLogit
from ensemble.stacking.stacking_MLR import StackingMLR
from ensemble.stacking.stacking_bayesian_avg import BayesianAverage
from ensemble.stacking.stacking_bayesian_avg_mcmc import BayesianAverageMCMC
from ensemble.stacking.stacking_agnostic_bayesian import AgnosticBayesian
from utils.tensor2matrix import tensor2matrix
from data.load_data import load_data
from metrics import accuracy, precision, roc_auc, recall, log_loss
from metrics import calculate_score
from utils.constants import BINARY_CLASSIFICATION
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


def setUpGlobal(cls):
    """
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target
    a, b, c, d = compute_level1_dataset(X, y)
    cls.basemodels_preds = a
    cls.test_preds = b
    cls.train_y = c
    cls.test_y = d
    """
    path = "./data/datasets/75103"
    print("data_loaded")
    a, b, c, d = load_data(path)
    cls.basemodels_preds = a
    cls.train_y = b
    cls.test_preds = c
    cls.test_y = d
    __basemodels_score__(cls.basemodels_preds, cls.train_y)


def tearDownGlobal(cls):
    cls.library = None

class TestStackingLogitReg(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)
        self.l1train = tensor2matrix(self.basemodels_preds)
        self.l1test = tensor2matrix(self.test_preds)



    def tearDown(self):
        tearDownGlobal(self)

    #test input format
    def test_valid_input(self):
        st = StackingLogitReg()
        st.fit(self.l1train, self.train_y)
        pred = st.predict(self.l1test)
        #basemodels predictions accuracy
        # error metric is accuracy measure.
        print("Test for Stacking with Regularized Logistic Regression")
        print("######################################################")
        print("Accuracy:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy))

        self.assertGreaterEqual(st.error_metric(np.argmax(pred, axis=1), self.test_y), 0.5)

class TestStackingLogit(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)

    def tearDown(self):
        tearDownGlobal(self)

    #test input format
    def test_valid_input(self):
        st = StackingLogit()
        self.l1train = tensor2matrix(self.basemodels_preds)
        self.l1test = tensor2matrix(self.test_preds)
        st.fit(self.l1train, self.train_y)
        pred = st.predict(self.l1test)
        print("Test for Stacking with Unregularized Logistic Regression")
        print("######################################################")
        print("Accuracy:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy))
        self.assertGreaterEqual(st.error_metric(pred, self.test_y), 0.5)


class TestStackingMLR(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)

    def tearDown(self):
        tearDownGlobal(self)

    #test input format
    def test_valid_input(self):
        st = StackingMLR(cvxopt=False)
        start = default_timer()
        print(self.basemodels_preds.shape, "test", self.test_preds.shape)
        st.fit(self.basemodels_preds, self.train_y)
        pred = st.predict(self.test_preds)
        print("pred", pred, "shape", pred.shape)
        print(default_timer() - start)
        n_datapoints = self.basemodels_preds.shape[1]
        print("Test for Stacking with Multi-Response Linear Regression")
        print("######################################################")
        print("precision:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, precision))
        print("roc_auc:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, roc_auc))
        print("log_loss:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, log_loss))
        print("recall:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, recall))
        print("accuracy:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy))
        #print (self.test_y)


class TestStackingBayesianAverage(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)

    def tearDown(self):
        tearDownGlobal(self)

    #test input format
    def test_valid_input(self):
        ba = BayesianAverage()
        ba.fit(self.basemodels_preds, self.train_y)
        pred = ba.predict(self.test_preds)
        print("preds", pred.shape, "test_y",self.test_y.shape, "test_x", self.test_preds.shape)
        #error metric is accuracy measure.
        print("Test for Stacking with Bayesian Averaging")
        print("######################################################")
        print("Accuracy:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy))
        print("roc_auc:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, roc_auc))


        #self.assertGreaterEqual(ba.error_metric(np.argmax(pred, axis=1), self.test_y), 0.5)


class TestStackingBayesianAverageMCMC(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)

    def tearDown(self):
        tearDownGlobal(self)

    #test input format
    def test_valid_input(self):
        bm = BayesianAverageMCMC()
        bm.fit(self.basemodels_preds.transpose(), self.train_y)
        pred = bm.predict(self.test_preds.transpose())
        print("pred", pred.shape)
        #error metric is accuracy measure.

        self.basemodels_preds = self.basemodels_preds.transpose()
        n_datapoints = self.basemodels_preds.shape[1]
        M1 = np.asarray([np.argmax(self.basemodels_preds[:, i, 0]) for i in
              range(n_datapoints)])
        M2 = np.asarray([np.argmax(self.basemodels_preds[:, i, 1]) for i in
              range(n_datapoints)])
        M3 = np.asarray([np.argmax(self.basemodels_preds[:, i, 2]) for i in
              range(n_datapoints)])
        M4 = np.asarray([np.argmax(self.basemodels_preds[:, i, 3]) for i in
              range(n_datapoints)])
        M5 = [np.argmax(self.basemodels_preds[:, i, 4]) for i in
              range(n_datapoints)]
        M6 = [np.argmax(self.basemodels_preds[:, i, 5]) for i in
              range(n_datapoints)]
        M7 = [np.argmax(self.basemodels_preds[:, i, 6]) for i in
              range(n_datapoints)]
        M5 = np.asarray(M5)
        M6 = np.asarray(M6)
        M7 = np.asarray(M7)
        print("Test for Stacking with Bayesian Averaging MCMC")
        print("######################################################")
        print("basemodel_1_Accuracy:", calculate_score(self.train_y, M1, BINARY_CLASSIFICATION, accuracy))
        print("basemodel_2_Accuracy:", calculate_score(self.train_y, M2, BINARY_CLASSIFICATION, accuracy))
        print("basemodel_3_Accuracy:", calculate_score(self.train_y, M3, BINARY_CLASSIFICATION, accuracy))
        print("basemodel_4_Accuracy:", calculate_score(self.train_y, M4, BINARY_CLASSIFICATION, accuracy))
        print("basemodel_5_Accuracy:", calculate_score(self.train_y, M5, BINARY_CLASSIFICATION, accuracy))
        print("basemodel_6_Accuracy:", calculate_score(self.train_y, M6, BINARY_CLASSIFICATION, accuracy))
        print("basemodel_7_Accuracy:", calculate_score(self.train_y, M7, BINARY_CLASSIFICATION, accuracy))
        print("Accuracy:", calculate_score(self.test_y, pred, BINARY_CLASSIFICATION, accuracy))


        self.assertGreaterEqual(bm.error_metric(np.argmax(pred, axis=1), self.test_y), 0.5)

        #self.assertGreaterEqual(bm.error_metric(pred, self.test_y), 0.5)
        #print self.test_y


class TestAgnosticBayesian(unittest.TestCase):

    def setUp(self):
        setUpGlobal(self)

    def tearDown(self):
        tearDownGlobal(self)

    #test input format
    def test_valid_input(self):
        ag = AgnosticBayesian()
        ag.fit(self.basemodels_preds.transpose(), self.train_y)
        ag = AgnosticBayesian(BINARY_CLASSIFICATION, accuracy)
        ag.fit(self.basemodels_preds.transpose(), self.train_y, n_bootstraps=1)
        pred = ag.predict(self.test_preds.transpose())
        print("Test for Stacking with Agnostic Bayesian Averaging")
        print("######################################################")
        #error metric is accuracy measure.
        print("Accuracy:", ag.error_metric(pred, self.test_y))
        self.assertGreaterEqual(ag.error_metric(pred, self.test_y), 0.5)



def main():
    unittest.main()

if __name__ == "__main__":
    main()
