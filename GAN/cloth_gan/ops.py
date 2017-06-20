# coding=utf-8
import tensorflow as tf
import numpy as np


def batch_norm(
        x, is_Train, scale=False, scope="batch_norm"):
    """
    披标准化
    :param x: Tensor
    :param is_Train:  是否是训练, 训练和测试必须指定
    :param scale: 下一个操作是否是线性
    :param scope: 操作名字
    :return: op
    """
    return tf.contrib.layers.batch_norm(
        x,
        is_Train=is_Train,
        scale=scale,  # 如果下一个操作是线性的, 比如 Relu scale可以为False
        scope=scope
    )
