#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
"""Placeholders"""

def create_placeholders(nx, classes):
    # tf.compat.v1.disable_eager_execution()
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    return x, y
