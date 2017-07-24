#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-


import os
import glob
from subprocess import check_call
import re
from os import listdir
from load_data import load_data

"""
Script for automating copying of multiple files
"""

#base_dir = '/mhome/negassim/Workspace/mastersproject/code/data/2017_misgana_ensembles/level0/'
base_dir = '/home/feurerm/mhome/projects/2017_misgana_ensembles/level0/'
target_dir = '/mhome/negassim/Workspace/mastersproject/code/data/datasets'

"""
for path in glob.glob(base_dir):
    dirname, filename = os.path.split(path)
    stripped_filename = re.sub('[^0-9]','', filename)
    check_call(["mkdir", os.path.join(target_dir,stripped_filename)])

"""

def copy_if_matches(pattern, m):
   if pattern in m:
       check_call(["cp", "-r", m, target_dir + "/" + f + "/" + pattern])
   else:
        print(pattern + " not found")



def dataset_dimension(path, basename):
    X, y, X_test, y_test = load_data(path)
    with open("data_dimensions.txt", "a") as f:
        f.write(basename + " " +  "X_train:" + str(X.shape) + "," + "y_train" + str(y.shape) + "X_test:" + str(X_test.shape) + "y_test:" +  str(y_test.shape) + "\n")



files = [f for f in listdir(target_dir)]

for f in files:
    path = target_dir + "/" + f
    print(path)
    dataset_dimension(path, f)


"""
for f in files:
   misgana =  glob.glob(base_dir + f + "_tmp/*.npy")
   for m in misgana:
       copy_if_matches("y_test.npy", m)

   misgana2 =  [base_dir + f + "_tmp/.auto-sklearn/" + filename for filename in os.listdir(base_dir + f + "_tmp/.auto-sklearn/")]
   for m2 in misgana2:
       copy_if_matches("true_target_ensemble.npy", m2)
       copy_if_matches("predictions_test_2", m2)
       copy_if_matches("predictions_ensemble", m2)

   #print(misgana)
   #print(misgana2)
   print("\n")
"""










