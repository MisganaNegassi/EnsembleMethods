#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-
import numpy as np
from copy import deepcopy



def bagging(X, y, X_test, model, task_type, metric, p=0.5, n_bags=20):
    """Implements bagging over datapoints from base model predictions
    Parameters
    ----------
    X : {array-like} of shape = [n_basemodels, n_datapoints, n_targets]
        Predicted class probabilites of base models trained on level-zero validation data
    y : array-like, shape = [n_datapoints]
        The predicted classes.
    X_test: array-like,  of shape = [n_basemodels, n_datapoints, n_targets]
        Predicted class probabilities on level-zero test data

    model : Instance of ensemble method
    task_type : type of classification task
    metric : metrics from Auto-sklearn used for scoring base_models_predictions.

    p : Fraction of models in the bag, optional
    n_bags : number of bags

    Returns
    -------
    bagged ensembles

    """
    bagged_ensembles = []
    n_models = X.shape[0]
    for _ in range(n_bags):
        model_cp = deepcopy(model)
        sample_size = p * X.shape[0]
        indices = sorted(np.random.choice(range(0, n_models), int(sample_size)))
        # random sample from library
        X_sample = X[indices, :, :]
        X_test_sample = X_test[indices, :, :]
        model_cp.fit(X_sample, y)
        bagged_ensembles.append(model_cp.predict(X_test_sample))

    return np.mean(bagged_ensembles, axis=0)
