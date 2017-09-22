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
        with tf.variable_scope('encoder'):
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
                tf.summary.histogram('hidden_2', hidden_2)
            with tf.variable_scope('hidden_3') as scope:
                w = tf.get_variable('w', [self.kwidth, 64, 128], initializer=self.w_init)
                b = tf.get_variable('b', [128], initializer=self.b_init)
                conv = tf.nn.conv1d(hidden_2, w, stride=self.stride, padding='SAME')
                hidden_3 = tf.nn.relu(conv + b, name=scope.name)
                tf.summary.histogram('hidden_3', hidden_3)
            with tf.variable_scope('hidden_4') as scope:
                w = tf.get_variable('w', [self.kwidth, 128, 256], initializer=self.w_init)
                b = tf.get_variable('b', [256], initializer=self.b_init)
                conv = tf.nn.conv1d(hidden_3, w, stride=self.stride, padding='SAME')
                hidden_4 = tf.nn.relu(conv + b, name=scope.name)
                tf.summary.histogram('hidden_4', hidden_4)
        with tf.variable_scope('decoder'):
            hidden_4 = tf.expand_dims(hidden_4, 1)
            with tf.variable_scope('deconv_1') as scope:
                w = tf.get_variable('w', [self.kwidth, 1, 128, 256], initializer=self.w_init)
                b = tf.get_variable('b', [128], initializer=self.b_init)
                prev_shape = hidden_3.shape.as_list()
                output_shape = [prev_shape[0], 1, prev_shape[1], prev_shape[2]]
                deconv = tf.nn.conv2d_transpose(hidden_4, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                deconv_1 = tf.nn.relu(deconv + b, name=scope.name)
                tf.summary.histogram('deconv_1', deconv_1)
            with tf.variable_scope('deconv_2') as scope:
                w = tf.get_variable('w', [self.kwidth, 1, 64, 128], initializer=self.w_init)
                b = tf.get_variable('b', [64], initializer=self.b_init)
                prev_shape = hidden_2.shape.as_list()
                output_shape = [prev_shape[0], 1, prev_shape[1], prev_shape[2]]
                deconv = tf.nn.conv2d_transpose(deconv_1, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                deconv_2 = tf.nn.relu(deconv + b, name=scope.name)
                tf.summary.histogram('deconv_2', deconv_2)
            with tf.variable_scope('deconv_3') as scope:
                w = tf.get_variable('w', [self.kwidth, 1, 32, 64], initializer=self.w_init)
                b = tf.get_variable('b', [32], initializer=self.b_init)
                prev_shape = hidden_1.shape.as_list()
                output_shape = [prev_shape[0], 1, prev_shape[1], prev_shape[2]]
                deconv = tf.nn.conv2d_transpose(deconv_2, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                deconv_3 = tf.nn.relu(deconv + b, name=scope.name)
                tf.summary.histogram('deconv_3', deconv_3)
            with tf.variable_scope('deconv_4') as scope:
                w = tf.get_variable('w', [self.kwidth, 1, 16, 32], initializer=self.w_init)
                b = tf.get_variable('b', [16], initializer=self.b_init)
                prev_shape = input_layer.shape.as_list()
                output_shape = [prev_shape[0], 1, prev_shape[1], prev_shape[2]]
                deconv = tf.nn.conv2d_transpose(deconv_3, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                deconv_4 = tf.nn.relu(deconv + b, name=scope.name)
                tf.summary.histogram('deconv_4', deconv_4)
            with tf.variable_scope('output_layer') as scope:
                w = tf.get_variable('w', [self.kwidth, 1, 1, 16], initializer=self.w_init)
                b = tf.get_variable('b', [1], initializer=self.b_init)
                prev_shape = input_tensor.shape.as_list()
                output_shape = [prev_shape[0], 1, prev_shape[1], prev_shape[2]]
                deconv = tf.nn.conv2d_transpose(deconv_4, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                output_layer = tf.nn.relu(deconv + b, name=scope.name)
                tf.summary.histogram('output_layer', output_layer)
        with tf.variable_scope('logits') as scope:
            output_layer = tf.squeeze(output_layer)
            w = tf.get_variable('w', [output_layer.shape[-1], 1], initializer=self.w_init)
            b = tf.get_variable('b', [1], initializer=self.b_init)
            logits = tf.matmul(output_layer, w) + b
            tf.summary.histogram(scope.name, logits)
        return logits


if __name__ == '__main__':
    import read

    sess = tf.InteractiveSession()
    reader = read.Reader('wavFile_train_frame_60.tfr', 32, 266)
    tf.train.start_queue_runners()
    inference = Inference(reader.read(), 18, 2)

    model = inference.build_model()
    tf.global_variables_initializer().run()
    print model
    print model.eval()
