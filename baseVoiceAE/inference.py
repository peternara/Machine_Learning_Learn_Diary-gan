import tensorflow as tf


class Inference(object):
    def __init__(self, input_tensor, kwidth=5, stride=2, is_train=True):
        self.input_tensor = input_tensor
        self.window_size = input_tensor.shape[-1]
        self.num_kernel = [64, 128, 256, 512]
        self.kwidth = kwidth
        self.stride = stride
        self.w_init = tf.contrib.layers.xavier_initializer()
        self.b_init = tf.constant_initializer([0.1])
        self.is_train = is_train

    def build_model(self):
        with tf.variable_scope('batch_norm') as scope:
            input_tensor = tf.contrib.layers.batch_norm(x=self.input_tensor,
                                                        is_training=self.is_train,
                                                        updates_collections=None,
                                                        scale=False,
                                                        reuse=False,
                                                        scope=scope.name)
        with tf.variable_scope('encoder'):
            with tf.variable_scope('input_layer') as scope:
                input_tensor = tf.cast(input_tensor, tf.float32)
                input_tensor = tf.expand_dims(input_tensor, 2)
                w = tf.get_variable('w', [self.kwidth, 1, self.num_kernel[0]], initializer=self.w_init)
                b = tf.get_variable('b', [self.num_kernel[0]], initializer=self.b_init)
                conv = tf.nn.conv1d(input_tensor, w, stride=self.stride, padding='SAME')
                input_layer = tf.nn.relu(conv + b, name=scope.name)
            with tf.variable_scope('hidden_1') as scope:
                w = tf.get_variable('w', [self.kwidth, self.num_kernel[0], self.num_kernel[1]], initializer=self.w_init)
                b = tf.get_variable('b', [self.num_kernel[1]], initializer=self.b_init)
                conv = tf.nn.conv1d(input_layer, w, stride=self.stride, padding='SAME')
                hidden_1 = tf.nn.relu(conv + b, name=scope.name)
            with tf.variable_scope('hidden_2') as scope:
                w = tf.get_variable('w', [self.kwidth, self.num_kernel[1], self.num_kernel[2]], initializer=self.w_init)
                b = tf.get_variable('b', [self.num_kernel[2]], initializer=self.b_init)
                conv = tf.nn.conv1d(hidden_1, w, stride=self.stride, padding='SAME')
                hidden_2 = tf.nn.relu(conv + b, name=scope.name)
            with tf.variable_scope('hidden_3') as scope:
                w = tf.get_variable('w', [self.kwidth, self.num_kernel[2], self.num_kernel[3]], initializer=self.w_init)
                b = tf.get_variable('b', [self.num_kernel[3]], initializer=self.b_init)
                conv = tf.nn.conv1d(hidden_2, w, stride=self.stride, padding='SAME')
                hidden_3 = tf.nn.relu(conv + b, name=scope.name)
        with tf.variable_scope('decoder'):
            hidden_3 = tf.expand_dims(hidden_3, 1)
            with tf.variable_scope('deconv_1') as scope:
                w = tf.get_variable('w', [self.kwidth, 1, self.num_kernel[2], self.num_kernel[3]],
                                    initializer=self.w_init)
                b = tf.get_variable('b', [self.num_kernel[2]], initializer=self.b_init)
                prev_shape = hidden_2.shape.as_list()
                output_shape = [prev_shape[0], 1, prev_shape[1], prev_shape[2]]
                deconv = tf.nn.conv2d_transpose(hidden_3, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                deconv_1 = tf.nn.relu(deconv + b, name=scope.name)
            with tf.variable_scope('deconv_2') as scope:
                w = tf.get_variable('w', [self.kwidth, 1, self.num_kernel[1], self.num_kernel[2]],
                                    initializer=self.w_init)
                b = tf.get_variable('b', [self.num_kernel[1]], initializer=self.b_init)
                prev_shape = hidden_1.shape.as_list()
                output_shape = [prev_shape[0], 1, prev_shape[1], prev_shape[2]]
                deconv = tf.nn.conv2d_transpose(deconv_1, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                deconv_2 = tf.nn.relu(deconv + b, name=scope.name)
            with tf.variable_scope('deconv_3') as scope:
                w = tf.get_variable('w', [self.kwidth, 1, self.num_kernel[0], self.num_kernel[1]],
                                    initializer=self.w_init)
                b = tf.get_variable('b', [self.num_kernel[0]], initializer=self.b_init)
                prev_shape = input_layer.shape.as_list()
                output_shape = [prev_shape[0], 1, prev_shape[1], prev_shape[2]]
                deconv = tf.nn.conv2d_transpose(deconv_2, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                deconv_3 = tf.nn.relu(deconv + b, name=scope.name)
            with tf.variable_scope('output_layer') as scope:
                w = tf.get_variable('w', [self.kwidth, 1, 1, self.num_kernel[0]], initializer=self.w_init)
                b = tf.get_variable('b', [1], initializer=self.b_init)
                prev_shape = input_tensor.shape.as_list()
                output_shape = [prev_shape[0], 1, prev_shape[1], prev_shape[2]]
                deconv = tf.nn.conv2d_transpose(deconv_3, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                output_layer = tf.nn.relu(deconv + b, name=scope.name)
        with tf.variable_scope('logits') as scope:
            output_layer = tf.squeeze(output_layer)
            w = tf.get_variable('w', [output_layer.shape[-1], 1], initializer=self.w_init)
            b = tf.get_variable('b', [1], initializer=self.b_init)
            logits = tf.matmul(output_layer, w) + b
        return logits


if __name__ == '__main__':
    import read

    sess = tf.InteractiveSession()
    reader = read.Reader(sess, 'wavFile_train_frame_60.tfr', 2, 266, 18)
    tf.train.start_queue_runners()
    input_tensor = tf.placeholder(tf.int16, [494, 18])
    inference = Inference(input_tensor, 18, 2)
    logits = inference.build_model()
    tf.global_variables_initializer().run()
    print sess.run(logits, feed_dict={
        input_tensor: reader.read()[0]
    })
