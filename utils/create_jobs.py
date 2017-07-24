#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import glob
import os
from os import listdir

from metrics import CLASSIFICATION_METRICS
from ensemble.ensemble_selection.ensemble_selection import EnsembleSelection
from ensemble.stacking.stacking_logit_reg import StackingLogitReg
from ensemble.stacking.stacking_logit import StackingLogit
from ensemble.stacking.stacking_MLR import StackingMLR
from ensemble.stacking.stacking_bayesian_avg import BayesianAverage
from ensemble.stacking.stacking_bayesian_avg_mcmc import BayesianAverageMCMC
from ensemble.stacking.stacking_agnostic_bayesian import AgnosticBayesian


"""script to write to job.txt file where each line
       is a single job to be run in the cluster.
"""
#python3 benchmarking.py EnsembleSelection ./data/datasets/adult_2117 accuracy binary.classification

task_type = "binary.classification"

#get path where the datasets reside
base_dir = '/home/negassim/Workspace/mastersproject/code/'
data_dir = '/mhome/negassim/Workspace/mastersproject/code/data/datasets/'
server_dir = '/home/negassim/Workspace/mastersproject/code/data/datasets/'
def get_datapath(base_dir):
    files = [f for f in listdir(data_dir)]
    return [server_dir +  f  for f in files]

def get_benchmark_script_path(base_dir):
    return base_dir + "benchmarking.py"

DATA_PATH = get_datapath(base_dir)
benchmark_script = get_benchmark_script_path(base_dir)

# get ensemble methods to benchmark with
ENSEMBLE_METHODS = [EnsembleSelection, StackingLogitReg, StackingLogit, StackingMLR,
                    BayesianAverage, BayesianAverageMCMC, AgnosticBayesian]

for path in DATA_PATH:
    for method in ENSEMBLE_METHODS:
        for metric in CLASSIFICATION_METRICS:
            if any(map(lambda v: v in metric, ("micro", "macro", "samples", "weighted", "pac_score", "average_precision"))):
                continue

            if any(map(lambda v: v in path, ("75205", "75098", "75243", "75223", "236", "262", "75110", "2122", "75181"))):
                #task_type = "multiclass.classification"
                continue
            with open("mlr_jobs.txt", "a") as f:
                """
                if method.__name__ == "EnsembleSelection":
                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                        method.__name__,
                                                        "--n_init 100",
                                                        "",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "--n_init 100",
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "",
                                                        method.__name__,
                                                        "--n_init 100",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "--n_init 100",
                                                        path, metric, task_type + "\n"))

                    #replacements,bagged, pruned, both
                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                        method.__name__,
                                                        "",
                                                        "--replacements True",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "--replacements True",
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "",
                                                        method.__name__,
                                                        "--replacements True",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "--replacements True",
                                                        path, metric, task_type + "\n"))

                    #repalcements, sorted
                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                        method.__name__,
                                                        "--n_init 100",
                                                        "--replacements True",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "--n_init 100",
                                                        "--replacements True",
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "",
                                                        method.__name__,
                                                        "--n_init 100",
                                                        "--replacements True",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "--n_init 100",
                                                        "--replacements True",
                                                        path, metric, task_type + "\n"))

                    # no model options
                    f.write("{} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))
                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "",
                                                            "--bagging True",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "--bagging True",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))


                elif method.__name__ == "StackingLogitReg":


                    f.write("{} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                        method.__name__,
                                                        "--reg_l1 True",
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "--reg_l1 True",
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "",
                                                        method.__name__,
                                                        "--reg_l1 True",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "--reg_l1 True",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "",
                                                            "--bagging True",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "--bagging True",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))


                elif method.__name__ == "StackingBayesianAverage":

                    f.write("{} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                        method.__name__,
                                                        "-- multi True",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "-- multi True",
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "",
                                                        method.__name__,
                                                        "-- multi True",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "--bagging True",
                                                        method.__name__,
                                                        "-- multi True",
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "",
                                                            "--bagging True",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                            "--prune True",
                                                            "--bagging True",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))

                """
                if method.__name__ == "StackingMLR":

                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                           "--output mlr_output.jl",
                                                            "",
                                                            "--bagging True",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))


                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                           "--output mlr_output.jl",
                                                            "--prune True",
                                                            "",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                           "--output mlr_output.jl",
                                                            "--prune True",
                                                            "--bagging True",
                                                        method.__name__,
                                                        path, metric, task_type + "\n"))

                    f.write("{} {} {} {} {} {} {}".format("python3",
                                                           benchmark_script,
                                                           "--output mlr_output.jl",
                                                           method.__name__,
                                                           path, metric, task_type + "\n"))
                else:
                    continue


