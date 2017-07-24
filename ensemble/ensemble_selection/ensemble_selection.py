#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-
import numpy as np
from metrics import calculate_score
from metrics import accuracy


class EnsembleSelection(object):
    """
    Class for Ensemble selection. Idea from "Ensemble Selection from libraries
    of models" by Caruana et al(2004)

    Parameters
    ----------
    no_iterations: int
                  Number of selection iterations to perform.
    task_type : bool
               type of classification task
    metric : metrics from Auto-sklearn used for scoring base_models_predictions.
    with_replacements :bool
                      Does not remove models after they are added
    sorted_initialization : bool
                        Sorts models by performance and initilizes ensemble
    n_init: int
                  Number of sorted good models


    """
    __name__ = "EnsembleSelection"

    def __init__(self, task_type, metric,
                 no_iterations,
                 with_replacements,
                 sorted_initialization,
                 n_init):


        self.n_init = n_init
        self.weights = None
        if no_iterations is None:
            no_iterations = 1000
        self.no_iterations = no_iterations
        self.with_replacements = with_replacements
        self.sorted_initialization = sorted_initialization
        if metric is None:
            metric = accuracy
        self.metric = metric
        self.task_type = task_type

    def fit(self, base_models_predictions, true_targets, model_identifiers=None):

    """ Build an ensemble from base_models_predictions

    Parameters
    ----------
    base_models_predictions : {array-like} of shape = [n_basemodels, n_datapoints, n_targets]
        Predicted class probabilites of base models trained on level-zero validation data
    true_targets : array-like, shape = [n_datapoints]
                  The target values.

    Returns
    --------
    self : object
          Returns self.
    """

        ensemble = []
        n_basemodels = base_models_predictions.shape[0]
        self.no_iterations = min(n_basemodels, self.no_iterations)
        active = np.ones(n_basemodels, dtype=bool)

        if not self.sorted_initialization and self.n_init is not None:
            raise ValueError("You specified parameter for number of " +
                             "initial models 'N', but did not choose " +
                             "'sorted_initialization'. This parameter " +
                             "combination is not supported!")

        if self.sorted_initialization:
            #self.n_init = int(0.2 * n_base_models_predictions)
            if self.n_init is None:
                raise ValueError("Please specify the number of" +
                                 " models to initialize ensemble with n_init")

            perf = np.zeros(n_basemodels)
            for idx, pred in enumerate(base_models_predictions):
                perf[idx] = calculate_score(true_targets, pred, self.task_type, self.metric)
            indices = np.argsort(perf)[-self.n_init:]

            ensemble = [base_models_predictions[idx] for idx in indices]

            if not self.with_replacements:
                for idx in indices:
                    active[idx] = False

        self.weights = np.zeros((n_basemodels,))
        for _ in range(self.no_iterations):

            best_index = -1
            best_score = -np.inf
            temp = list(ensemble)

            # N:number of models active in the library
            for basemodel in range(n_basemodels):
                if active[basemodel]:
                    if not ensemble:
                        score = calculate_score(true_targets, base_models_predictions[basemodel, :, :], self.task_type, self.metric)
                    else:
                        avg_temp = np.mean(temp + [base_models_predictions[basemodel, :, :]], axis=0)
                        score = calculate_score(true_targets, avg_temp, self.task_type, self.metric)
                    if score >= best_score:
                        best_index = basemodel
                        best_score = score
            ensemble.append(base_models_predictions[best_index, :, :])
            self.weights[best_index] += 1
            # Should this be outside of loop?
            if not self.with_replacements:
                active[best_index] = False

        self.weights = self.weights / np.sum(self.weights)
        print("EnsembleSelection Weights: ", self.weights)

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
        pred = np.dot(X_test.transpose(), self.weights)
        return pred.transpose()
