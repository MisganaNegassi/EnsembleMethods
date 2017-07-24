#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-

""" Take an error_metric from a list and iterate each algorithm and return
results for each metric
"""
from metrics import accuracy

METRICS = (accuracy)


def gen_model_with_metrics(cls):
    for metric in METRICS:
        instance = cls(error_metric=metric)
        yield instance


