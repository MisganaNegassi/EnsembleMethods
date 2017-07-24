#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-
from sklearn.linear_model import LogisticRegressionCV
from utils.tensor2matrix import tensor2matrix

class StackingLogitReg(object):
    """
    This class implements Stacking with regularized logistic regression
    """

    __name__ = "StackingLogitReg"

    def __init__(self, regularizer):

        if regularizer is None:
            regularizer = "l2"
        self.regularizer = regularizer

    def fit(self, X, true_targets):

        """ Build an ensemble from base_models_predictions

        Parameters
        ----------
        X: {array-like} of shape = [n_basemodels, n_datapoints, n_targets]
            Predicted class probabilites of base models trained on level-zero validation data
        true_targets : array-like, shape = [n_datapoints]
                      The target values.

        Returns
        --------
        self: object
             Returns self.
        """

        X = tensor2matrix(X)
        if self.regularizer == "l1":
            model = LogisticRegressionCV(penalty= self.regularizer, solver="liblinear")
        else:
            model = LogisticRegressionCV(penalty=self.regularizer)
        self.level1_model = model.fit(X, true_targets)

    def predict(self, X_test):

        """

        Parameters
        ----------
        X_test: array-like,  of shape = [n_basemodels, n_datapoints, n_targets]
                Predicted class probabilities on level-zero test data

        Returns
        -------
        predicted classes of level-one model(logistic regression)
        """
        X_test = tensor2matrix(X_test) #flatten 3-D vector to 2-D
        return self.level1_model.predict_proba(X_test)

