#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-
from logging import getLogger
import json
import argparse
import numpy as np
from timeit import default_timer
from metrics import CLASSIFICATION_METRICS, calculate_score
from metrics.constants import STRING_TO_TASK_TYPES
from wrappers.library_pruning import prune
from wrappers.library_bagging import bagging
from data.load_data import load_data
from utils.get_datasetname import get_datasetname



def main():
    parser = argparse.ArgumentParser(description="Benchmarking script for Ensemble Methods.")
    parser.add_argument("--output", help="Output 'JSON lines (jl)' filename.", default="output.jl")

    subparsers = parser.add_subparsers()

    ensemble_selection_parser = subparsers.add_parser("EnsembleSelection", help="Ensemble selection method")
    ensemble_selection_parser.add_argument("--n_init", help="Use sorted initialization with N_INIT models option for Ensemble Selection.")
    ensemble_selection_parser.add_argument("--replacements", help="Use with replacements option for Ensemble Selection.")
    ensemble_selection_parser.set_defaults(func=benchmark_ensemble_selection)

    stacking_logit_parser = subparsers.add_parser("StackingLogit", help="Stacking with unregularized Logistic Regression")
    stacking_logit_parser.set_defaults(func=benchmark_stacking_logit)


    stacking_logit_reg_parser = subparsers.add_parser("StackingLogitReg", help="Stacking with regularized Logistic Regression")
    stacking_logit_reg_parser.add_argument("--reg_l1", help="Regularizer used for logistic Regression")
    stacking_logit_reg_parser.set_defaults(func=benchmark_stacking_logit_reg)

    stacking_mlr_parser = subparsers.add_parser("StackingMLR", help="Stacking with Multi-response Linear Regression")
    stacking_mlr_parser.set_defaults(func=benchmark_stacking_mlr)

    stacking_bayesian_avg_parser = subparsers.add_parser("BayesianAverage", help="Stacking with Bayesian Model Averaging")
    stacking_bayesian_avg_parser.add_argument("--multi", help="Bayesian Averaging for multi or binary classification, if option True for multi-classification")
    stacking_bayesian_avg_parser.set_defaults(func=benchmark_stacking_bayesian_avg)

    stacking_bayesian_avg_mcmc_parser = subparsers.add_parser("BayesianAverageMCMC", help="Stacking with Bayesian Model Averaging MCMC")
    stacking_bayesian_avg_mcmc_parser.set_defaults(func=benchmark_stacking_bayesian_avg_mcmc)

    stacking_agnostic_bayesian_parser = subparsers.add_parser("AgnosticBayesian", help="Stacking with Agnostic Bayesian")
    stacking_agnostic_bayesian_parser.set_defaults(func=benchmark_stacking_agnostic_bayesian)

    best_model_parser = subparsers.add_parser("BestModel",help="Best performing base model")
    best_model_parser.set_defaults(func=benchmark_best_model)

    parser.add_argument("--prune", help="prunes basemodels to be ensembled")
    parser.add_argument("--bagging", help="bags multiple ensembles on random bags and takes their average")

    parser.add_argument("data_path", help="Path to file with dataset.")
    parser.add_argument("metric", help="Error metric from autosklearn.")
    parser.add_argument("task_type", help="Type of classification.")

    args = parser.parse_args()

    print("Performance:", args.func(args))
    encoder = json.JSONEncoder()
    append_json(filename=args.output, encoder=encoder, new_json=args.func(args))

def append_json(filename, encoder, new_json):
    with open(filename, "a") as f:
        f.write(encoder.encode(new_json) + "\n")



def apply_model(args, X, y, X_test, y_test, task_type, metric, model, name, dataset):

    print("X_train:", X.shape, "y_train", y.shape, "X_test:", X_test.shape, "y_test:", y_test.shape)

    if args.prune and args.bagging:
        start = default_timer()
        X_pruned, X_test_pruned = prune(X, y, X_test, task_type, metric) # prunes 20% away
        pred = bagging(X_pruned, y, X_test_pruned, model, task_type, metric) # predictions of bagged ensembles
        name = "pruned_bagged_" + name
    elif args.prune:
        start = default_timer()
        X_pruned, X_test_pruned = prune(X, y, X_test, task_type, metric)# prunes 20% away
        model.fit(X_pruned, y)
        pred  = model.predict(X_test_pruned)
        # do pruning
        name = "pruned_" + name
    elif args.bagging:
        start = default_timer()
        # do bagging
        pred = bagging(X, y, X_test, model, task_type, metric)#predictions of bagged ensembles
        name = "bagged_" + name
    else:
        start = default_timer()
        model.fit(X, y)
        pred  = model.predict(X_test)
    perf = calculate_score(y_test, pred, task_type, metric)
    runtime = default_timer() - start

    #logs runtime for each job
    with open("runtime.txt", "a") as f:
        f.write( name + ": " + str(runtime) + "\n")
    return {dataset: {name: {str(metric): perf}}}

def benchmark_best_model(args):

    task_type = STRING_TO_TASK_TYPES[args.task_type]
    path = args.data_path
    dataset = get_datasetname(path)
    metric = CLASSIFICATION_METRICS[args.metric]
    name = "SINGLE BEST"
    X, y, X_test, y_test = load_data(path)
    n_basemodels = X.shape[0]
    best_model = np.argmax(calculate_score(y, X[m,:,:], task_type, metric) for m in range(n_basemodels))
    perf = calculate_score(y_test, X_test[best_model, :, :], task_type, metric)

    return {dataset: {name: {str(metric): perf}}}


def benchmark_ensemble_selection(args):
    from ensemble.ensemble_selection.ensemble_selection import EnsembleSelection
    task_type = STRING_TO_TASK_TYPES[args.task_type]
    path = args.data_path

    metric = CLASSIFICATION_METRICS[args.metric]

    X, y, X_test, y_test = load_data(path)
    print(X.shape)

    if args.n_init and args.replacements:
        n_init = int(args.n_init)
        model = EnsembleSelection(task_type, metric, no_iterations=1000,
                                  sorted_initialization=True, with_replacements=True,
                                  n_init=n_init)
        name = "EnsembleSelectionSortedInitialization_WithReplacements"

    elif args.n_init:
        n_init = int(args.n_init)
        model = EnsembleSelection(task_type, metric, no_iterations=1000,
                                  sorted_initialization=True, n_init=n_init)
        name = "EnsembleSelectionSortedInitialization"
    elif args.replacements:
        model = EnsembleSelection(task_type, metric, no_iterations=1000,
                                  with_replacements=True)
        name = "EnsembleSelectionReplacements"
    else:
        model = EnsembleSelection(task_type, metric, no_iterations=1000)
        name = "EnsembleSelection"

    return apply_model(args, X=X, y=y, X_test=X_test, y_test=y_test, metric=metric, task_type=task_type, model=model, name=name, dataset=get_datasetname(path))

def benchmark_stacking_logit(args):
    from ensemble.stacking.stacking_logit import StackingLogit
    task_type = STRING_TO_TASK_TYPES[args.task_type]
    path = args.data_path
    name = "StackingLogit"

    metric = CLASSIFICATION_METRICS[args.metric]

    X, y, X_test, y_test = load_data(path)

    model = StackingLogit()
    return apply_model(args, X=X, y=y, X_test=X_test, y_test=y_test, metric=metric, task_type=task_type, model=model, name=name, dataset=get_datasetname(path))

def benchmark_stacking_logit_reg(args):
    from ensemble.stacking.stacking_logit_reg import StackingLogitReg
    task_type = STRING_TO_TASK_TYPES[args.task_type]
    path = args.data_path

    name = "StackingLogitReg"

    metric = CLASSIFICATION_METRICS[args.metric]

    X, y, X_test, y_test = load_data(path)

    if args.reg_l1:
        model = StackingLogitReg(regularizer="l1")
        name = "StackingLogitRegularized_l1"

    model = StackingLogitReg(regularizer="l2")
    name = "StackingLogitRegularized_l2"
    return apply_model(args, X=X, y=y, X_test=X_test, y_test=y_test, metric=metric, task_type=task_type, model=model, name=name, dataset=get_datasetname(path))

def benchmark_stacking_mlr(args):
    from ensemble.stacking.stacking_MLR import StackingMLR
    task_type = STRING_TO_TASK_TYPES[args.task_type]
    path = args.data_path
    name = "StackingMLR"

    metric = CLASSIFICATION_METRICS[args.metric]

    X, y, X_test, y_test = load_data(path)

    model = StackingMLR()
    return apply_model(args, X=X, y=y, X_test=X_test, y_test=y_test, metric=metric, task_type=task_type, model=model, name=name, dataset=get_datasetname(path))

def benchmark_stacking_bayesian_avg(args):
    from ensemble.stacking.stacking_bayesian_avg import BayesianAverage
    task_type = STRING_TO_TASK_TYPES[args.task_type]
    path = args.data_path

    name = "StackingBayesianAverage"
    metric = CLASSIFICATION_METRICS[args.metric]

    X, y, X_test, y_test = load_data(path)

    if args.multi:
        model = BayesianAverage(multi=True)
        name = "StackingBayesianAverageMultiClass"

    model = BayesianAverage()
    name = "StackingBayesianAverageBinaryClass"
    return apply_model(args, X=X, y=y, X_test=X_test, y_test=y_test, metric=metric, task_type=task_type, model=model, name=name, dataset=get_datasetname(path))

def benchmark_stacking_bayesian_avg_mcmc(args):
    from ensemble.stacking.stacking_bayesian_avg_mcmc import BayesianAverageMCMC
    task_type = STRING_TO_TASK_TYPES[args.task_type]
    path = args.data_path

    name = "StackingBayesianAverageMCMC"
    metric = CLASSIFICATION_METRICS[args.metric]

    X, y, X_test, y_test = load_data(path)

    model = BayesianAverageMCMC(n_samples=10000)
    return apply_model(args, X=X, y=y, X_test=X_test, y_test=y_test, metric=metric, task_type=task_type, model=model, name=name, dataset=get_datasetname(path))

def benchmark_stacking_agnostic_bayesian(args):
    from ensemble.stacking.stacking_agnostic_bayesian import AgnosticBayesian

    task_type = STRING_TO_TASK_TYPES[args.task_type]
    path = args.data_path
    name =  "StackingAgnosticBayesian"

    metric = CLASSIFICATION_METRICS[args.metric]

    X, y, X_test, y_test = load_data(path)

    model = AgnosticBayesian(task_type, metric, n_bootstraps=500)
    return apply_model(args, X=X, y=y, X_test=X_test, y_test=y_test, metric=metric, task_type=task_type, model=model, name=name, dataset=get_datasetname(path))



if __name__ == "__main__":
    main()
