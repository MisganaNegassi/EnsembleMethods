#!/usr/bin/env/python
# -*- coding: iso-8859-15 -*-


def bayesian_averaging(m, C, x):
    return (C*m + sum(x)) / float(C + len(x))
