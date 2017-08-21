import tensorflow as tf


class Losses(object):
    def __init__(self, logits, labels):
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='losses')
        self.losses = tf.reduce_mean(xentropy)
        tf.summary.scalar('total_losses', self.losses)

    def get_losses(self):
        return self.losses


XXX = ''
labels = ''
biases = ''


logits = tf.matmul(XXX, XXX) + biases
loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

