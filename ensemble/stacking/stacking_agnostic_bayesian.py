#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-

import numpy as np
from sklearn.utils import resample
from metrics import calculate_score

class AgnosticBayesian(object):

    """ Class implements ideas from "Agnostic Bayesian Learning" by Lacoste et al


    Parameters
    ----------

    task_type : bool
               type of classification task
    metric : metrics from Auto-sklearn used for scoring base_models_predictions.
    n_bootstraps: int
                number of random samples

    """


    __name__ =  "AgnosticBayesian"

    def __init__(self, task_type, metric, n_bootstraps):
        self.metric = metric
        self.task_type = task_type
        self.n_bootstraps = n_bootstraps
        self.weight_vector = None


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
        self.weight_vector: array-like, shape[n_basemodels]
        """

        base_models_predictions = base_models_predictions.transpose()
        n_targets = base_models_predictions.shape[0]
        n_datapoints = base_models_predictions.shape[1]
        n_base_models = base_models_predictions.shape[2]

        self.weight_vector = np.zeros(n_base_models)

        index_list = np.arange(n_datapoints, dtype=int)
        for _ in range(self.n_bootstraps):
            risk_vector = np.zeros(n_base_models)

            for model in range(n_base_models):
                pred = base_models_predictions[:, index_list, model]
                target = true_targets[index_list]
                score = calculate_score(target, pred.transpose(), self.task_type, self.metric)
                risk_vector[model] = 1 - score
            self.weight_vector[np.argmin(risk_vector)] += 1

        def normalize_proba(probs):
            return probs / float(np.sum(probs))

        self.weight_vector = normalize_proba(self.weight_vector)
        print("AgnosticBayesianWeights:", self.weight_vector)

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
        pred = np.dot(X_test, self.weight_vector)
        return pred.transpose()
