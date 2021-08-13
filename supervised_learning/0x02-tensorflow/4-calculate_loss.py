#!/usr/bin/env python3
"""Loss"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """def calculate_loss: calculates the softmax cross-entropy
                            loss of a prediction

    Args:
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing the network’s predictions

    Returns:
        a tensor containing the loss of the prediction
    """
    return tf.compat.v1.losses.softmax_cross_entropy(
        y, y_pred)
