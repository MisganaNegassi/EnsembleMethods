#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-

import numpy as np


class BayesianAverage(object):

    """  This class implements Bayesian model averaging

    """

    __name__ = "BayesianAverage"


    def __init__(self, multi=False):
        self.weights = None
        self.multi = multi
        self.normalizer = 0.0


    def fit(self, base_models_predictions, true_targets):

        """ Build an ensemble from base_models_predictions

        Parameters
        ----------
        base_models_predictions : {array-like} of shape = [n_basemodels, n_datapoints, n_targets]
            Predicted class probabilites of base models trained on level-zero validation data
        true_targets : array-like, shape = [n_datapoints]
                      The target values.

        Returns
        --------
        self.weights: array-like, shape[n_basemodels]
        """

        base_models_predictions = base_models_predictions.transpose()
        n_targets = base_models_predictions.shape[0]
        n_datapoints = base_models_predictions.shape[1]
        n_base_models = base_models_predictions.shape[2]

        # One hot encoding of true_targets
        def compute_y(n_targets, target_index):
            return np.array([0 if target != target_index else 1
                            for target in range(n_targets)])

        self.weights = np.zeros([n_base_models])
        for model in range(n_base_models):
            for datapoint in range(n_datapoints):
                pred = base_models_predictions[:, datapoint, model]
                target = true_targets[datapoint]

                if self.multi:
                    # binary classification
                    log_likelihood = np.log(target * pred[1] + (1. - target) *
                                            pred[0] + 1e-7)
                    if log_likelihood != log_likelihood:
                        log_likelihood = 0
                else:
                    #multi-class classification
                    encoded_target = compute_y(n_targets, target)
                    log_likelihood = np.dot(pred, encoded_target)
                    if log_likelihood != log_likelihood:
                        log_likelihood = 0
                    log_likelihood = np.log(log_likelihood)
            self.weights[model] += log_likelihood
            #take exponent of weights
        self.weights = np.exp(self.weights)
        self.normalizer = np.sum(self.weights)
        self.weights /= self.normalizer
        print("BayesianAverageWeights:", self.weights, "shape", self.weights.shape)
        return self.weights

    def predict(self, X_test):

        """
        Parameters
        ----------
        X_test: array-like,  of shape = [n_basemodels, n_datapoints, n_targets]
                Predicted class probabilities on level-zero test data

        Returns
        -------
        predicted classes of final ensemble

        """
        X_test = X_test.transpose()
        pred = np.dot(X_test, self.weights)
        return pred.transpose()
