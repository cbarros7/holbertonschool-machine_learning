#!/usr/bin/env python3
""" Normal"""


class Normal():
    """Normal class
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """Constructor

        Args:
            data : list of the data to be used to estimate the distribution.
                    Defaults to None.
            mean : mean of the distribution. Defaults to 0.
            stddev :  standard deviation of the distribution Defaults to 1.
        """
        self.e = 2.7182818285
        self.pi = 3.1415926536
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = mean
            self.stddev = stddev
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            var = sum([((x - self.mean) ** 2) for x in data])
            self.stddev = var / len(data) ** 0.5
