# coding=utf-8
import tensorflow as tf


def add_l2_loss(w, decay):
    l2loss = decay * tf.nn.l2_loss(w)
    tf.add_to_collection('l2_loss', l2loss)


class Inference(object):
    def __init__(self, input_tensor, kwidth=31, stride=2, isTrain=True, scope=None):
        self.input_tensor = input_tensor
        self.window_size = input_tensor.shape[-1]
        self.num_kernel = [16, 64, 256, 512]
        self.kwidth = kwidth
        self.stride = stride
        self.w_init = tf.contrib.layers.xavier_initializer()
        self.b_init = tf.constant_initializer([0.1])
        self.isTrain = isTrain
        self.scope = scope

    def batch_norm(
            self, x, is_train, scope="batch_norm"):
        """
        披标准化
        :param x: Tensor
        :param is_train:  是否是训练, 训练和测试必须指定
        :param scope: 操作名字
        :return: op
        """
        return tf.contrib.layers.batch_norm(
            x,
            is_training=is_train,
            updates_collections=None,
            scale=False,  # 如果下一个操作是线性的, 比如 Relu scale可以为False
            reuse=True,
            scope=scope
        )

    def build_model(self):
        layer_shape = []
        with tf.variable_scope('encoder'):
            input_tensor = tf.reshape(self.input_tensor, [-1, self.kwidth])
            # with tf.variable_scope('batch_norm') as batch_norm_scope:
            #     input_tensor = self.batch_norm(input_tensor, is_train=self.isTrain, scope=batch_norm_scope)
            layer = tf.expand_dims(input_tensor, 1)
            layer = tf.expand_dims(layer, 3)
            layer_shape.append(layer.get_shape().as_list())
            for index, num_kernel in enumerate(self.num_kernel):
                with tf.variable_scope('encode_{}'.format(index)) as scope:
                    w = tf.get_variable('w', [1, self.kwidth, layer.shape[-1], num_kernel], initializer=self.w_init)
                    b = tf.get_variable('b', [num_kernel], initializer=self.b_init)
                    conv = tf.nn.conv2d(layer, w, strides=[1, 1, self.stride, 1], padding='SAME')
                    layer_prew = layer.shape
                    layer = tf.nn.relu(tf.nn.bias_add(conv, b), name=scope.name)
                    print "conv {} -> {}".format(layer_prew, layer.shape)
                    layer_shape.append(layer.get_shape().as_list())

        with tf.variable_scope('decoder'):
            layer_shape_reversed = layer_shape[::-1]
            for index in xrange(0, len(self.num_kernel)):
                with tf.variable_scope('deconv_{}'.format(index)) as scope:
                    input_channels = layer_shape_reversed[index][-1]
                    output_channels = layer_shape_reversed[index + 1][-1]
                    output_shape = layer_shape_reversed[index + 1]

                    w = tf.get_variable('w', [1, self.kwidth, output_channels, input_channels],
                                        initializer=self.w_init)
                    b = tf.get_variable('b', [output_channels], initializer=self.b_init)
                    deconv = tf.nn.conv2d_transpose(layer, w, output_shape=output_shape, strides=[1, 1, self.stride, 1])
                    layer_prew = layer.shape
                    layer = tf.nn.relu(tf.nn.bias_add(deconv, b), name=scope.name)
                    print "deconv {} -> {}".format(layer_prew, layer.shape)

        layer = tf.squeeze(layer)
        with tf.variable_scope('full_connect'):
            w = tf.get_variable('w', [layer.shape[-1], 1], initializer=self.w_init)
            b = tf.get_variable('b', [1], initializer=self.b_init)
            logits = tf.matmul(layer, w) + b

        return logits


if __name__ == '__main__':
    import read

    sess = tf.InteractiveSession()
    reader = read.Reader('wavFile_train_frame_60.tfr', 1, 266, 32)
    tf.train.start_queue_runners()
    inference = Inference(reader.wav_raw, 32, 2)
    logits = inference.build_model()
    tf.global_variables_initializer().run()
    print logits
    print reader.wav_raw
    print reader.label
