#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-

import sys
from matplotlib import pyplot as plt
import numpy as np
from ensemble.stacking.stacking_bayesian_avg import BayesianAverage
from pymc3 import Model, HalfNormal, Metropolis, find_MAP, sample, Categorical
from pymc3 import traceplot, NUTS


class BayesianAverageMCMC(object):

    """ This class implements Bayesian Model Averaging with MCMC Sampling

    """
    __name__ = "BayesianAverageMCMC"

    def __init__(self, n_samples):
        self.n_samples = n_samples


    def fit(self, base_models_predictions, true_targets,
            model_identifiers=None):

        ba = BayesianAverage()
        weight_vector = ba.fit(base_models_predictions, true_targets)
        default = True

        base_models_predictions = base_models_predictions.transpose()
        n_basemodels = base_models_predictions.shape[2]
        with Model() as basic_model:
            #define prior
            HalfNormal('weights', sd=1, shape=n_basemodels)
            #define likelihood function
            ensemble_pred = np.dot(base_models_predictions, weight_vector)
            Categorical('likelihood', p=ensemble_pred.transpose(), observed=true_targets)

        with basic_model:
            start = find_MAP(model=basic_model)
            if not default:
                step = Metropolis()
            step = NUTS()
            trace = sample(self.n_samples, step=step, start=start)
        trace = trace[5000:]
        self.sampled_weights = trace["weights"]

    def predict(self, test_set):

        weight_vector = np.mean(self.sampled_weights, axis=0)
        pred = np.dot(test_set.transpose(), weight_vector)
        return pred.transpose()

