import tensorflow as tf


class Losses(object):
    def __init__(self, logits, labels):
        self.loss = tf.reduce_mean(tf.abs(tf.subtract(logits, labels)))
