#!/usr/bin/env python3
"""Adam"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """create_Adam_op: creates the training operation for a neural network
                    in tensorflow using the Adam optimization algorithm
    Args:
        loss is the loss of the network
        alpha is the learning rate
        beta1 is the weight used for the first moment
        beta2 is the weight used for the second moment
        epsilon is a small number to avoid division by zero

    Returns: the Adam optimization operation
    """
    return tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1, beta2=beta2,
        epsilon=epsilon, use_locking=False,
        name='Adam')
