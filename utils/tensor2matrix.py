#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import numpy as np

def tensor2matrix(tensor):
    """TODO: Docstring for tensor2marix.

    Args:
        tensor (TODO): TODO

    Returns: TODO

    """
    matrix = np.append(tensor[0, :, :], tensor[1, :, :], axis=1)
    for class_matrix in tensor[2:, :, :]:
        matrix = np.append(matrix, class_matrix, axis=1)

    return matrix


