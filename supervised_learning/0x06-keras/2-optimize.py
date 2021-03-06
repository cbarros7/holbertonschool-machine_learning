#!/usr/bin/env python3
"""Optimize"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """optimize_model: sets up Adam optimization for
                        a keras model with categorical
                        crossentropy loss and accuracy metrics

    Args:
        network : is the model to optimize
        alpha : is the learning rate
        beta1 : is the first Adam optimization parameter
        beta2 : is the second Adam optimization parameter

    Returns:
        None
    """
    optimizer = K.optimizers.Adam(lr=alpha,
                                  beta_1=beta1, beta_2=beta2)
    loss = 'categorical_crossentropy'
    network.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return None
