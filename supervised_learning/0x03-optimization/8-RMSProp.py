#!/usr/bin/env python3
"""RMSProp"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """create_RMSProp_op: training operation for a neural network in
                            tensorflow using the RMSProp optimization algorithm

    Args:
        loss is the loss of the network
        alpha is the learning rate
        beta2 is the RMSProp weight
        epsilon is a small number to avoid division by zero

    Returns: the RMSProp optimization operation
    """
    return tf.train.RMSPropOptimizer(learning_rate=alpha,
                                     decay=beta2,
                                     momentum=0.0,
                                     epsilon=epsilon).minimize(loss)
