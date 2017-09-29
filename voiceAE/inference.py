# coding=utf-8
import tensorflow as tf


class Inference(object):
    def __init__(self, input_tensor, kwidth=31, stride=2, isTrain=True):
        self.input_tensor = input_tensor
        self.window_size = input_tensor.shape[-1]
        self.num_kernel = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.kwidth = kwidth
        self.stride = stride
        self.w_init = tf.contrib.layers.xavier_initializer()
        self.b_init = tf.constant_initializer([0.1])
        self.isTrain = isTrain

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
            reuse=False,
            scope=scope
        )

    def build_model(self):
        layer_shape = []
        with tf.variable_scope('encoder'):
            input_tensor = tf.cast(self.input_tensor, tf.float32)
            input_tensor = self.batch_norm(input_tensor, is_train=self.isTrain)
            layer = tf.expand_dims(input_tensor, 2)
            layer = tf.expand_dims(layer, 2)
            layer_shape.append(layer.get_shape().as_list())
            for index, num_kernel in enumerate(self.num_kernel):
                with tf.variable_scope('encode_{}'.format(index)) as scope:
                    w = tf.get_variable('w', [1, self.kwidth, layer.shape[-1], num_kernel], initializer=self.w_init)
                    b = tf.get_variable('b', [num_kernel], initializer=self.b_init)
                    conv = tf.nn.conv2d(layer, w, strides=[1, self.stride, 1, 1], padding='SAME')
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
                    deconv = tf.nn.conv2d_transpose(layer, w, output_shape=output_shape, strides=[1, self.stride, 1, 1])
                    layer_prew = layer.shape
                    layer = tf.nn.relu(tf.nn.bias_add(deconv, b), name=scope.name)
                    print "deconv {} -> {}".format(layer_prew, layer.shape)

        layer = tf.squeeze(layer)
        return layer


if __name__ == '__main__':
    import read
    import loss

    sess = tf.InteractiveSession()
    reader = read.Reader(sess, 'data', 2, 2 ** 14)
    tf.train.start_queue_runners()

    raw, noise = reader.read()

    inference = Inference(noise)
    inference = inference.build_model()

    # loss_inf = loss.Losses(inference, raw).get_loss()
    # loss_raw = loss.Losses(noise, raw).get_loss()
    #
    # op = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(loss_inf)
    #
    # tf.global_variables_initializer().run()
    #
    # print loss_inf.eval()
    # print loss_raw.eval()
    #
    # for i in xrange(100):
    #     op.run()
    #
    # print loss_inf.eval()
