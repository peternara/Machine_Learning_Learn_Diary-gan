# coding=utf-8
import os

import numpy as np
import tensorflow as tf

from model import DCGAN
from utils import pp, show_all_variables, visualize
import scipy.misc as plt

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "训练次数")
flags.DEFINE_float("learning_rate", 0.003, "学习速率")
flags.DEFINE_float("beta1", 0.5, "Adam 动量")
flags.DEFINE_integer("train_size", np.inf, "每个epoch的训练的次数")
flags.DEFINE_integer("batch_size", 64, "批大小")
flags.DEFINE_integer("input_height", 256, "图片输入高度")
flags.DEFINE_integer("input_width", None, "图片输入宽度, 如果空, 和高度一致")
flags.DEFINE_integer("output_height", 128, "输出图片高度")
flags.DEFINE_integer("output_width", None, "图片输出宽度, 如果空, 和高度一致")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "图片名")
flags.DEFINE_string("checkpointDir", "checkpoint_dir/", "模型保存路径")
flags.DEFINE_string("summaryDir", "logs/", "TensorBoard路径")
flags.DEFINE_string("buckets", "data/cat", "数据源路径")
flags.DEFINE_string("dataset", "cat", "数据集名称")
flags.DEFINE_boolean("train", False, "是否是训练, 否则将进行可视化")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not tf.gfile.Exists(FLAGS.checkpointDir):
        tf.gfile.MakeDirs(FLAGS.checkpointDir)
    if not tf.gfile.Exists(FLAGS.buckets):
        tf.gfile.MakeDirs(FLAGS.buckets)

    #
    # 针对PAI IO 优化:
    # 把OSS文件拷贝到运行时目录
    # 如果在本地运行请跳过这一步

    # if not tf.gfile.Exists('./cope_data'):
    #     tf.gfile.MakeDirs('./cope_data')
    # for file_path in tf.gfile.Glob(os.path.join(FLAGS.buckets, '*')):
    #     tf.gfile.Copy(file_path, os.path.join('cope_data', os.path.basename(file_path)), overwrite=True)
    # FLAGS.buckets = './cope_data/'

    # 注意, 如果不是在PAI上可以省去上面这一步
    # 请注释掉上面5行代码
    #


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        dcgan = DCGAN(
            sess,
            coord,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpointDir,
            sample_dir=FLAGS.buckets,
            config=FLAGS)

        show_all_variables()

        if FLAGS.train:
            dcgan.train()
        else:
            if not dcgan.load(FLAGS.checkpointDir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            option = 0
            # 0: 生成1 * batch_size 张图片
            # 1: 生成100 * batch_size 张图片
            # 2 - 4: 使用不同的模式生成GIF
            visualize(sess, dcgan, FLAGS, option)


if __name__ == '__main__':
    tf.app.run()
