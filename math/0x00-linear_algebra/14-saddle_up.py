#!/usr/bin/env python3
"""Matrix multiplication"""
import numpy as np


def np_matmul(mat1, mat2):
    """np_matmul: performs matrix multiplication

    Args:
        mat1 : First matrix
        mat2 : Second matrix
    """
    return np.matrix(mat1) * np.matrix(mat2)
