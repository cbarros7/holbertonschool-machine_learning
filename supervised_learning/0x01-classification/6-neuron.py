#!/usr/bin/env python3
"""Neuron"""
import numpy as np


class Neuron():
    """Single neuron performing binary classification"""

    def __init__(self, nx):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1: 
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """forward propagation function"""
        Z = np.matmul(self.W, X) + self.b
        self.__A = self.sigmoid(Z)
        return self.A

    def sigmoid(self, Y):
        """define the sigmoid activation function"""
        return 1 / (1 + np.exp(-1 * Y))

    def cost(self, Y, A):
        """defnine the cost function"""
        # classification pb: use the cross-entropy function
        m = Y.shape[1]
        return (-1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """function that evaluates the neuron's predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """function that calculates one pass of gradient descent"""
        dZ = A - Y
        m = Y.shape[1]
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        self.__W -= alpha * dW
        self.__b -= (alpha * db)[0][0]

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """function that trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        A, cost = self.evaluate(X, Y)
        return A, cost
