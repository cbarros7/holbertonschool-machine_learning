#!/usr/bin/env python3
""" Normal"""
import math

class Normal():
    def __init__(self, data=None, mean=0., stddev=1.):
        """Constructor

        Args:
            data : list of the data to be used to estimate the distribution. 
                    Defaults to None.
            mean : mean of the distribution. Defaults to 0..
            stddev :  standard deviation of the distribution Defaults to 1.
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) <= 1:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)
                self.stddev = (sum([((x - self.mean) ** 2) for x in data]) \
                                / len(data))** 0.5