import tensorflow as tf


# def downconv(x, output_dim, kwidth=5, stride=2, init=None, uniform=False, name='downconv'):
#     if init is None:
#         init = tf.contrib.layers.xavier_initializer()
#     with tf.variable_scope(name):
#         w = tf.get_variable('w', [kwidth, x.shape[-1], output_dim],
#                             initializer=init)
#         conv = tf.nn.conv1d(x, w, padding='SAME', stride=stride)
#         b = tf.get_variable('b', [output_dim],
#                             initializer=tf.constant_initializer([0.1]))
#         conv = tf.nn.bias_add(conv, b)
#     return conv


class Inference(object):
    def __init__(self, input_tensor, kwidth=5, stride=2):
        self.input_tensor = input_tensor
        self.window_size = input_tensor.shape[-1]
        self.num_kernel = [16, 32, 64, 128]
        self.kwidth = kwidth
        self.stride = stride
        self.w_init = tf.contrib.layers.xavier_initializer()
        self.b_init = tf.constant_initializer([0.1])

    def build_model(self):
        with tf.variable_scope('input_layer') as scope:
            input_tensor = tf.cast(self.input_tensor, tf.float32)
            input_tensor = tf.expand_dims(input_tensor, 2)
            w = tf.get_variable('w', [self.kwidth, 1, 16], initializer=self.w_init)
            b = tf.get_variable('b', [16], initializer=self.b_init)
            conv = tf.nn.conv1d(input_tensor, w, stride=self.stride, padding='SAME')
            input_layer = tf.nn.relu(conv + b, name=scope.name)
            tf.summary.histogram('input_layer', input_layer)
        with tf.variable_scope('hidden_1') as scope:
            w = tf.get_variable('w', [self.kwidth, 16, 32], initializer=self.w_init)
            b = tf.get_variable('b', [32], initializer=self.b_init)
            conv = tf.nn.conv1d(input_layer, w, stride=self.stride, padding='SAME')
            hidden_1 = tf.nn.relu(conv + b, name=scope.name)
            tf.summary.histogram('hidden_1', hidden_1)
        with tf.variable_scope('hidden_2') as scope:
            w = tf.get_variable('w', [self.kwidth, 32, 64], initializer=self.w_init)
            b = tf.get_variable('b', [64], initializer=self.b_init)
            conv = tf.nn.conv1d(hidden_1, w, stride=self.stride, padding='SAME')
            hidden_2 = tf.nn.relu(conv + b, name=scope.name)
            print hidden_2
            tf.summary.histogram('hidden_2', hidden_2)
        with tf.variable_scope('output_layer') as scope:
            dim = 1
            for d in hidden_2.get_shape().as_list()[1:]:
                dim *= d
            flat = tf.reshape(hidden_2, [-1, dim])
            w = tf.get_variable('w', [dim, 1], initializer=self.w_init)
            b = tf.get_variable('b', [1], initializer=self.b_init)
            output_layer = tf.matmul(flat, w) + b
            tf.summary.scalar('logits', tf.reduce_mean(output_layer))
        return output_layer


if __name__ == '__main__':
    import read

    sess = tf.InteractiveSession()
    reader = read.Reader('wavFile_train_frame_60.tfr', 32, 266)
    tf.train.start_queue_runners()
    inference = Inference(reader.read(), 18, 2)
    print inference.build_model()
