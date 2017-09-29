import tensorflow as tf


class Losses(object):
    def __init__(self, Autoencoder_output, raw_wav):
        # self.loss = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(Autoencoder_output, raw_wav), 2))
        self.loss = tf.reduce_mean(tf.abs(tf.subtract(Autoencoder_output, raw_wav)))

    def get_loss(self):
        return self.loss
