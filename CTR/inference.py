import tensorflow as tf


class Inference(object):
    def __init__(self, data_input, h1_size, h2_size, num_classes, is_training):
        with tf.variable_scope('hidden_1'):
            weight = tf.Variable(tf.truncated_normal([data_input.shape.as_list()[-1], h1_size], stddev=1e-4),
                                 name='hidden_1_weight')
            biase = tf.constant(0.1, shape=[h1_size])
            h1_layer = tf.nn.sigmoid(tf.matmul(data_input, weight) + biase, name='hidden_1')
            self._summary_helper(h1_layer)
            if is_training:
                h1_layer = tf.nn.dropout(h1_layer, 0.5)

        with tf.variable_scope('hidden_2'):
            weight = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=1e-4), name='hidden_2_weight')
            biase = tf.constant(0.1, shape=[h2_size])
            h2_layer = tf.nn.sigmoid(tf.matmul(h1_layer, weight) + biase, name='hidden_2')
            self._summary_helper(h2_layer)
            if is_training:
                h2_layer = tf.nn.dropout(h2_layer, 0.5)

        with tf.variable_scope('logits'):
            weight = tf.Variable(tf.truncated_normal([h2_size, num_classes], stddev=1e-4), name='logits_layer_weight')
            biase = tf.constant(0.1, shape=[num_classes])
            self.logits = tf.matmul(h2_layer, weight) + biase

    def _summary_helper(self, variable):
        tf.summary.histogram(variable.op.name + '/activation', variable)
        tf.summary.scalar(variable.op.name + '/mean', tf.reduce_mean(variable))

    def get_inference(self):
        return self.logits

    def get_softmax(self):
        return tf.nn.softmax(self.logits)