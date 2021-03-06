#!/usr/bin/env python3
"""Momemtum"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """create_momentum_op: creates the training operation for a
                            neural network in tensorflow using the gradient
                            descent with momentum optimization algorithm

    Args:
        loss is the loss of the network
        alpha is the learning rate
        beta1 is the momentum weight

    Returns:
        the momentum optimization operation
    """
    return tf.train.MomentumOptimizer(
        learning_rate=alpha, momentum=beta1, use_locking=False,
        name='Momentum', use_nesterov=False
    ).minimize(loss)
