# coding=utf-8
import tensorflow as tf
import numpy as np


def variable_with_weight_decay(name, shape, stddev, wd):
    """
    初始化权重, 如果有wd, name会进行L2规范化
    :param name: 别名
    :param shape: 形状
    :param stddev: 高斯标准差
    :param wd: 权重衰减参数
    :return: Tensor
    """
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def summary_helper(x):
    """
    tensorbord 助手函数
    :param x: 需要可视化的操作
    """
    name = x.op.name
    tf.summary.histogram(name + '/activation', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))


def inference(image):
    with tf.variable_scope('conv1') as scope:
        """
        第一层卷积
        50 * 50 * 3
        |
        V
        50 * 50 * 64
        """
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        baises = tf.constant(0.1, shape=[64])
        bais = tf.add(conv, baises)
        conv1 = tf.nn.relu(bais, name=scope.name)

    with tf.variable_scope('pool_and_norm_1'):
        """
        第一次池化和归一化
        50 * 50 * 64
        |
        V
        25 * 25 * 64
        """
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        """
        第二次卷积
        25 * 25 * 64
        |
        V
        25 * 25 * 64
        """
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        baises = tf.constant(0.1, shape=[64])
        bais = tf.add(conv, baises)
        conv2 = tf.nn.relu(bais, name=scope.name)

    with tf.variable_scope('pool_and_norm_2'):
        """
        第二次池化和归一化
        25 * 25 * 64
        |
        V
        13 * 13 * 64
        """
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('full_connect_1') as scope:
        """
        全连接1层
        step1: 展平
        step2: 权重初始化
        step3: 偏置初始化
        13 * 13 * 64
        |
        V
        384
        """
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        flat = tf.reshape(pool2, [-1, dim])
        weights = variable_with_weight_decay('weights', [dim, 256], 4e-2, 4e-3)
        biases = tf.constant(0.1, shape=[256], name='biases')
        fc_1 = tf.nn.relu(tf.add(tf.matmul(flat, weights), biases), name=scope.name)

    with tf.variable_scope('full_connect_2') as scope:
        """
        全连接2层
        384
        |
        V
        192
        """
        weights = variable_with_weight_decay('weights', shape=[256, 128], stddev=0.04, wd=0.004)
        biases = tf.constant(0.1, shape=[128], name='biases')
        fc_2 = tf.nn.relu(tf.matmul(fc_1, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        """
        soft_max_线性全连接层
        """
        weights = variable_with_weight_decay('weights', [128, 2], stddev=1 / 192.0, wd=0.0)
        biases = tf.constant(0.0, shape=[2], name='biases')
        softmax_linear = tf.add(tf.matmul(fc_2, weights), biases, name=scope.name)
        summary_helper(softmax_linear)

    return softmax_linear


def main(arg=None):
    image = tf.Variable(np.ones([15, 50, 50, 3], dtype=float))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    image = tf.cast(image, tf.float32)
    a = inference(image)
    shape = tf.shape(a)
    print sess.run(shape)


if __name__ == '__main__':
    tf.app.run()
