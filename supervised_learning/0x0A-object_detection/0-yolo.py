#!/usr/bin/env python3
"""Initialize Yolo """
import numpy as np
import tensorflow as tf


class Yolo():
    """Class YOLO"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """define and initialize attributes and variables"""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
