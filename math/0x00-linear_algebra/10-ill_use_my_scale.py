#!/usr/bin/env python3
"""calculates the shape"""
import numpy as np


def np_shape(matrix):
    """np_shape : calculates the shape of a numpy.ndarray

    Args:
        matrix: matrix to calculate the shape
    """
    _shape = []
    if len(matrix.tolist()) == 0:
        _shape.append(0)
    elif isinstance(matrix[0], np.int64):
        _shape.append(len(matrix))
    else:
        _shape.append(len(matrix))
        _shape.append(len(matrix[0]))
        _shape.append(len(matrix[0][0]))

    return tuple(_shape)
