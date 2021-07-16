#!/usr/bin/env python3
import numpy as np
"""Concatenate two matrices"""


def np_cat(mat1, mat2, axis=0):
    """np_cat: concatenates two matrices along a specific axis

    Args:
        mat1 ([type]): [description]
        mat2 ([type]): [description]
        axis (int, optional): [description]. Defaults to 0.
    """
    return np.concatenate((mat1, mat2), axis=axis)