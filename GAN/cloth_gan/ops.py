# coding=utf-8
import tensorflow as tf
import numpy as np


def batch_norm(
        x, is_Train, scope="batch_norm"):
    """
    披标准化
    :param x: Tensor
    :param is_Train:  是否是训练, 训练和测试必须指定
    :param scope: 操作名字
    :return: op
    """
    return tf.contrib.layers.batch_norm(
        x,
        is_Train=is_Train,
        scale=True,  # 如果下一个操作是线性的, 比如 Relu scale可以为False
        scope=scope
    )


def weight_init(shape, decay=0.0):
    """
    权重初始化, 如果有decay, 会进行l2_loss
    :param shape: 张量形状
    :param decay: 如果存在, 会进行l2_loss
    :return: Tensor
    """
    weight = tf.get_variable('weight', tf.truncated_normal(shape, stddev=1e-4))
    if not decay == 0.0:
        l2_loss = tf.multiply(tf.nn.l2_loss(weight), decay, name='weight_loss')
        tf.add_to_collection('l2_loss', l2_loss)
    return weight


def biase_init(shape, value=0.1):
    """
    偏置初始化
    :param shape: 张量形状
    :param value: 初始值大小
    :return: Tensor
    """
    biase = tf.get_variable('biase', tf.constant(value, shape=shape))
    return biase


def full_connect(x, shape_out, scope='full_connect'):
    """
    全连接层
    :param x: 全连接输入
    :param shape_out: 全连接输出大小
    :param scope: scope名
    :return: Option
    """
    with tf.variable_scope(scope):
        weight = weight_init(shape=[x.get_shape()[-1], shape_out], decay=4e-3)
        biase = biase_init([shape_out])
        output = tf.nn.bias_add(tf.matmul(x, weight), biase)
        return output


def conv_2d(x, num_filter, kernel_size=5, stride=2, scope='conv_2d'):
    """
    卷积
    :param x: 卷积输入
    :param num_filter: 卷积核数
    :param kernel_size: 卷积核大小
    :param stride: 卷积步长
    :param scope: scope名
    :return: Option
    """
    with tf.variable_scope(scope):
        weight = weight_init([kernel_size, kernel_size, x.get_shape()[-1], num_filter])
        biase = biase_init([num_filter])
        conv = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding="SAME")
        output = tf.nn.bias_add(conv, biase)
        return output


def conv_2d_transpose(x, out_shape, kernel_size=5, stride=2, scope='conv_2d_transpose'):
    """
    逆卷积
    :param x: 卷积输入
    :param out_shape: 卷积输出
    :param kernel_size: 卷积核大小
    :param stride: 卷积步长
    :param scope: scope名
    :return: Option
    """
    with tf.variable_scope(scope):
        weight = weight_init([kernel_size, kernel_size, out_shape[-1], x.get_shape()[-1]])
        biase = biase_init(out_shape[-1])
        deconv = tf.nn.conv2d_transpose(x, weight, out_shape, strides=[1, stride, stride], padding="SAME")
        output = tf.nn.bias_add(deconv, biase)
        return output


def leaky_relu(x, leakey=0.2):
    return tf.maximum(x, x * leakey)
