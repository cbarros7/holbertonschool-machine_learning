#!/usr/bin/env python3
"""Momemtum"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    return tf.train.MomentumOptimizer(
        learning_rate=alpha, momentum=beta1, use_locking=False,
        name='Momentum', use_nesterov=False
    ).minimize(loss)
