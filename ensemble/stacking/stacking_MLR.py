#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-

import numpy as np


class StackingMLR(object):

    """ This class implements stacking with Multi-response Linear Regression
        from the paper "Issues in Stacked Generalization" by Ting and Witten

    """

    __name__ = "StackingMLR"


    def __init__(self, cvxopt=False):
        self.weight_matrix = None
        self.cvxopt = cvxopt



    def fit(self, base_models_predictions, true_targets,
            model_identifiers=None):
        """Fit an ensemble given predictions of base models and targets.
        Parameters
        ----------
        base_models_predictions :
                  array of shape = [n_base_models, n_data_points, n_targets]
                  n_targets is the number of classes in case of classification,
                  n_targets is 0 or 1 in case of regression

        true_targets : array of shape [n_targets]

        model_identifiers : identifier for each base model.
                         Can be used for practical text output of the ensemble.
        Returns
        -------
        self

        """

        n_base_models= base_models_predictions.shape[0]
        n_targets = base_models_predictions.shape[2]
        #n_data_points = base_models_predictions .shape[1]
        self.weight_matrix = np.zeros([n_base_models, n_targets])
        initial_guess = np.ones(n_targets) * (1.0 / n_targets)

        y = true_targets

        from scipy.optimize import minimize
        for model in range(n_base_models):
            X = base_models_predictions[model, :, :]

            def f(weights):
                return sum((y[n] - sum(weights[k] * X[n][k]
                                       for k in range(n_targets)))
                           for n in range(len(y)))**2

            def normalize_weights(weights):
                return [w / float(sum(weights)) for w in weights]

            def gen_constraint(i):
                def cons(x):
                    return x[i]
                return cons

            funcs = [gen_constraint(i) for i in range(n_targets)]

            constraints = [{"type": "ineq", "fun": func} for func in funcs]

            sol = minimize(f, initial_guess, constraints=constraints)
            self.weight_matrix[model, :] = normalize_weights(sol.x)
        print("MLRWeights:", self.weight_matrix)

    def predict(self, X_test):
        """Create ensemble predictions from weights learned


        Parameters
        ----------
        X_test: array-like,  of shape = [n_basemodels, n_datapoints, n_targets]
                Predicted class probabilities on level-zero test data

        Returns
        -------
        predicted classes of final ensemble
        """
        n_datapoints = X_test.shape[1]
        n_targets = X_test.shape[2]
        pred = np.zeros([n_targets, n_datapoints])
        tmp = np.zeros([n_targets, n_datapoints])


        weight_targets = self.weight_matrix.shape[1]
        for target in range(weight_targets):
            tmp = np.dot(test_set.transpose(), self.weight_matrix[:, target])
            pred = np.add(tmp, pred)
        return pred.transpose()
