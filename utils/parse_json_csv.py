#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import json
import glob
import os
import numpy as np
import csv

def parse_json():

    data = next(iter(glob.glob(os.path.expanduser('~/Workspace/masters/mastersproject/code/data/*.json'))))
    dicts = []
    with open(data, "r") as f:
        for line in f:
            dicts.append(json.loads(line))
    X = np.asarray([np.asarray(list(line.values())) for line in dicts])
    print("X", X)
    return X


print("=======================================================================")


def parse_csv():
    data = next(iter(glob.glob(os.path.expanduser('~/Workspace/masters/mastersproject/code/data/*.csv'))))
    dicts = []
    with open(data, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        X = np.asarray([np.asarray(line) for line in reader])
        print("X", X)
    return X

