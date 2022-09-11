import tensorflow as tf
from LTLOperator import LTLOperator
import numpy as np
from flloat.parser.ltlf import LTLfParser
import random
import math
from copy import deepcopy

batch_size = 100
num_train = 500

class Anneal(tf.keras.callbacks.Callback):
    """
    Callback that anneals sigmoid and relu
    """
    def on_epoch_end(self, epoch, logs=None):
        for l in self.model.layers:
            if isinstance(l, LTLOperator):
                l.slope.assign(l.slope + 0.01)
                l.alpha.assign(tf.maximum(l.alpha - 7e-5, 0))

class DiscreteAcc(tf.keras.callbacks.Callback):
    """
    Callback that checks discretized accuracy
    """
    def __init__(self, train_traces, train_labels):
        self.train_traces = train_traces
        self.train_labels = train_labels

    def on_epoch_end(self, epoch, logs=None):
        output = self.model.predict(self.train_traces, batch_size=batch_size)
        output = output.reshape(-1)
        labels = self.train_labels.numpy()
        metrics = float(tf.reduce_mean(tf.cast(output == labels, tf.float32)))
        if metrics == 1.0:
            self.model.stop_training = True
