#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-


from utils.tensor2matrix import tensor2matrix
from statsmodels.api import MNLogit as log_reg

class StackingLogit(object):
    """
    This class implements Stacking with unregularized logistic regression
    """

    __name__ = "StackingLogit"


    def __init__(self):

        self.level1_model = None


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
        model = log_reg(true_targets, X)
        self.level1_model = model.fit(method='bfgs')


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
        X_test = tensor2matrix(X_test) #flatten 3-D to 2-D vector
        return self.level1_model.predict(X_test)
