#!/usr/bin/env python3
"""Placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """create_placeholders : returns two placeholders,
                            x and y, for the neural network

    Args:
        nx : the number of feature columns in our data
        classes : the number of classes in our classifier

    Returns:
        x is the placeholder for the input data to the neural network
        y is the placeholder for the one-hot labels for the input data

    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    return x, y
