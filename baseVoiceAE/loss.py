import tensorflow as tf


def loss(logits, labels):
    labels = tf.reshape(labels, [-1])
    tf.add_to_collection('losses', tf.reduce_mean(tf.abs(tf.subtract(logits, labels))))
