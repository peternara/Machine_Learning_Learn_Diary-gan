# coding=utf-8
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import reader
import utils

tf.flags.DEFINE_string('buckets', './data', "数据源")
tf.flags.DEFINE_string('loss_model', 'inception_resnet_v2', "需要使用到的模型")
tf.flags.DEFINE_string('checkpoint_exclude_scopes', 'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits', '剔除的部分')
tf.flags.DEFINE_string('trainable_scopes', 'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits', '需要训练的部分')
tf.flags.DEFINE_integer('num_classes', 80, "最后一层输出")
tf.flags.DEFINE_integer('num_samples', 100000, "数据集数据数")
tf.flags.DEFINE_integer('height', 299, "图片高度")
tf.flags.DEFINE_integer('width', 299, "图片宽度")
tf.flags.DEFINE_integer('num_preprocessing_threads', 8, "运行线程")
tf.flags.DEFINE_boolean('is_training', True, "是否在训练")

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

network_fn = nets_factory.get_network_fn(
    FLAGS.loss_model,
    num_classes=FLAGS.num_classes,
    is_training=FLAGS.is_training)

image_preprocessing_fn, sec = preprocessing_factory.get_preprocessing(
    FLAGS.loss_model,
    is_training=FLAGS.is_training)

provider = utils.get_provider(FLAGS)
image, label = provider.get(['image', 'label'])

image = image_preprocessing_fn(image, FLAGS.height, FLAGS.width)

images, labels = tf.train.batch(
    [image, label],
    batch_size=FLAGS.batch_size,
    num_threads=FLAGS.num_preprocessing_threads,
    capacity=5 * FLAGS.batch_size)
labels = slim.one_hot_encoding(
    labels, FLAGS.num_classes)
batch_queue = slim.prefetch_queue.prefetch_queue(
    [images, labels], capacity=2 * FLAGS.num_classes)

network_fn(image)
