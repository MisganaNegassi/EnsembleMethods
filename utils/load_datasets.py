#!/usr/bin/env/python
# -*- coding: utf-8 -*-
import numpy as np
import os


from sklearn.datasets import load_iris,load_digits, load_breast_cancer

BIN_DATASETS = [load_iris(), load_breast_cancer()]
MULTI_DATASETS = [load_digits()]

