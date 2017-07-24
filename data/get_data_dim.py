#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

import os
import glob
import re
from os import listdir
from load_data import load_data


path = '/mhome/negassim/Workspace/mastersproject/code/data/datasets'
X, y, X_test, y_test = load_data(path)

def dataset_dimension(path, basename):
    X, y, X_test, y_test = load_data(path)
    with open("data_dimensions.txt", "a") as f:
        f.write(basename + " " +  "X_train:" + str(X.shape) + "," + "y_train" + str(y.shape) + "X_test:" + str(X_test.shape) + "y_test:" +  str(y_test.shape) + "\n")



target_dir = '/mhome/negassim/Workspace/mastersproject/code/data/datasets'
files = [f for f in listdir(target_dir)]

for f in files:
    path = target_dir + "/" + f
    dataset_dimension(path, f)
