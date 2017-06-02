# coding=utf-8
import tensorflow as tf


def loss(logits, labels):
    """
    构建损失函数
    :param logits: 从inference传过来的全连接输出
    :param labels: 从loader传来的标签值
    :return: Tensor
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                            name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('loss/loss', loss)
    return loss


def get_loss():
    return tf.add_n(tf.get_collection('losses'), name='loss_collect')
