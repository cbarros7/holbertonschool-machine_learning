#!/usr/bin/env python3

""" Calculate the shape of a matrix, returned as a list of integers """


def matrix_shape(matrix):
    """matrix_shape :  that calculates the shape of a matrix

    Args:
        matrix: matrix to be dimensioned
    """
    if not matrix:
        return None

    row = len(matrix)
    col = len(matrix[0])

    if isinstance(matrix[0][0], int):
        return [row, col]
    else:
        elements = len(matrix[0][0])
        return [row, col, elements]
