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
        self.num_full_connect_weight = [256, 1]
        self.kwidth = kwidth
        self.stride = stride
        self.w_init = tf.contrib.layers.xavier_initializer()
        self.b_init = tf.constant_initializer([0.1])
        self.isTrain = isTrain
        self.scope = scope

    def batch_norm(
            self, x, is_train, scope="batch_norm", reuse=False):
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
            reuse=reuse,
            scope=scope
        )

    def build_model(self):
        with tf.variable_scope('encoder'):
            input_tensor = tf.reshape(self.input_tensor, [-1, self.kwidth])
            layer = tf.expand_dims(input_tensor, 2)
            for index, num_kernel in enumerate(self.num_kernel):
                with tf.variable_scope('encode_{}'.format(index)) as scope:
                    w = tf.get_variable('w', [self.kwidth, layer.shape[-1], num_kernel], initializer=self.w_init)
                    b = tf.get_variable('b', [num_kernel], initializer=self.b_init)
                    conv = tf.nn.conv1d(layer, w, padding='SAME', stride=self.stride)
                    layer_prew = layer.shape
                    layer = tf.nn.relu(tf.nn.bias_add(conv, b), name=scope.name)
                    print "conv {} -> {}".format(layer_prew, layer.shape)

        with tf.name_scope('flat'):
            dim = 1
            for each in layer.get_shape().as_list()[1:]:
                dim *= each
            layer_prew = layer.shape
            layer = tf.reshape(layer, [-1, dim])
            print "flat conv_layer: {} -> {}".format(layer_prew, layer.shape)

        with tf.variable_scope('full_connect'):
            for index, num_weight in enumerate(self.num_full_connect_weight):
                with tf.variable_scope('full_connect_{}'.format(index)):
                    w = tf.get_variable('w', [layer.shape[-1], num_weight], initializer=self.w_init)
                    b = tf.get_variable('b', [num_weight], initializer=self.b_init)
                    layer_prew = layer.shape
                    layer = tf.matmul(layer, w) + b
                    print "full connect {} -> {}".format(layer_prew, layer.shape)

        return layer


if __name__ == '__main__':
    import read

    sess = tf.InteractiveSession()
    reader = read.Reader('wavFile_train_frame_60.tfr', 1, 266, 32)
    tf.train.start_queue_runners()
    inference = Inference(reader.wav_raw, 32, 2)
    logits = inference.build_model()
    tf.global_variables_initializer().run()

    print reader.wav_raw
