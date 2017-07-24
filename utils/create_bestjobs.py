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
base_dir = '/mhome/negassim/Workspace/mastersproject/code/'
data_dir = '/mhome/negassim/Workspace/mastersproject/code/data/datasets/'
server_dir = '/mhome/negassim/Workspace/mastersproject/code/data/datasets/'
def get_datapath(base_dir):
    files = [f for f in listdir(data_dir)]
    return [server_dir +  f  for f in files]

def get_benchmark_script_path(base_dir):
    return base_dir + "benchmarking.py"

DATA_PATH = get_datapath(base_dir)
benchmark_script = get_benchmark_script_path(base_dir)
name = "BestModel"

for path in DATA_PATH:
    for metric in CLASSIFICATION_METRICS:
        if any(map(lambda v: v in metric, ("micro", "macro", "samples", "weighted", "pac_score", "average_precision"))):
            continue

        if any(map(lambda v: v in path, ("75205", "75098", "75243", "75223", "236", "262", "75110", "2122", "75181"))):
            #task_type = "multiclass.classification"
            continue
        with open("best_jobs.txt", "a") as f:
            f.write("{} {} {} {} {} {} {}".format("python3", benchmark_script,
                                                  "--output best.jl", name,
                                                  path, metric, task_type + "\n"))
