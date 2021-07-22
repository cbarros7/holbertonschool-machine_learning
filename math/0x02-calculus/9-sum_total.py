#!/usr/bin/env python3
"""Summation"""


def summation_i_squared(n):
    """summation_i_squared: calculate the squared

    Args:
        n: stopping condition
    """
    if not isinstance(n, int) or n <= 0:
        return None
    elif n == 1:
        return n
    return n**2 + summation_i_squared(n - 1)
