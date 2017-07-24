#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-

import numpy as np


class StackingRegression(object):

    """Stacked Regressions

    This class implements method for linear combination of different models
    using least squares under non-negativity constraint to determine their
    weights as proposed in paper by Leo Breiman in "Stacked Regression",1995.

    Parameters
    ----------

    Returns
    -------

    """
    __ name__ = "StackingRegression"

    def __init__(self, base_models=None, error_metric=None, cvxopt=False):

        self.weights = None
        self.error_metric = error_metric
        self.cvxopt = cvxopt

        if self.error_metric is None:
            from sklearn.metrics import accuracy_score
            self.error_metric = accuracy_score

    def fit(self, X, y):
        """Build Stacking Classifier from training set(X, y)

        Parameters
        ----------

        X: {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. The features are basemodel predictions.

        y: array-like shape = [n_samples]
           The target values(class labels)



        Returns
        -------

        """
        no_weights = X.shape[1]

        if self.cvxopt:
            try:
                from cvxopt import matrix, solvers
                Q = matrix(np.dot(X.transpose(), X), tc="d")
                c = matrix(-(np.dot(X.transpose(), y)), tc="d")
                G = matrix(-np.eye(no_weights, dtype=int), tc="d")
                h = matrix(np.zeros(no_weights, dtype=int), tc="d")
                sol = solvers.qp(Q, c, G, h)
                self.weights = sol["x"]
                print(self.weights)
            except NameError:
                print("Importing cvxopt failed, fall back to scipy.minimize!")
                self.cvxopt = False

        if not self.cvxopt:
            from scipy.optimize import minimize

            def f(weights):
                #TODO: use ndarray
                return sum((y[n] - sum(weights[k] * X[n][k]
                                       for k in range(no_weights))
                            for n in range(len(y))))**2

            #initial_guess = np.zeros(no_weights)
            initial_guess = np.ones(no_weights) * (1.0 / X.shape[0])

            def gen_constraint(i):
                def cons(x):
                    return x[i]
                return cons
            funcs = [gen_constraint(i) for i in range(no_weights)]

            constraints = [{"type": "ineq", "fun": func} for func in funcs]

            sol = minimize(f, initial_guess, constraints=constraints)
            self.weights = sol.x
            print(self.weights)

    def predict(self, X):
        """Create ensemble predictions from predictions of base_models

        Parameters
        ----------


        Returns
        -------


        """
        if self.cvxopt:
            prediction_regression = np.dot(X, self.weights)
            return np.array([round(el) for sublst in prediction_regression
                            for el in sublst])
        else:
            return map(round, np.dot(X, self.weights))
