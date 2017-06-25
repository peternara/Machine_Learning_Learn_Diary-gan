# coding=utf-8
import os

import numpy as np
import tensorflow as tf

from model import DCGAN
from utils import pp, show_all_variables

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "训练次数")
flags.DEFINE_float("learning_rate", 0.0002, "学习速率")
flags.DEFINE_float("beta1", 0.5, "Adam 动量")
flags.DEFINE_integer("train_size", np.inf, "每次训练的次数")
flags.DEFINE_integer("batch_size", 64, "图片批数")
flags.DEFINE_integer("input_height", 256, "图片输入大小")
flags.DEFINE_integer("input_width", None, "图片输入宽度, 如果空, 和高度一致")
flags.DEFINE_integer("output_height", 256, "输出图片大小")
flags.DEFINE_integer("output_width", None, "图片输出宽度, 如果空, 和高度一致")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "图片名")
flags.DEFINE_string("checkpointDir", "logs/", "模型保存路径")
flags.DEFINE_string("summaryDir", "logs/", "TensorBoard路径")
flags.DEFINE_string("local", "data/sample_dir/", "图片储存路径")
flags.DEFINE_string("dataset", "sample_dir", "数据集名称")
flags.DEFINE_boolean("train", True, "是否是训练")
flags.DEFINE_boolean("crop", False, "是否裁剪, 如果训练的时候, 建议为True, 如果是测试的时候, 建议为False")
flags.DEFINE_boolean("visualize", False, "是否进行可视化")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpointDir):
        os.makedirs(FLAGS.checkpointDir)
    if not os.path.exists(FLAGS.local):
        os.makedirs(FLAGS.local)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    #  这里意思是 缓慢的增加显存需求, 不是一开始就占用所有的显存

    with tf.Session(config=run_config) as sess:

        dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            dataset_name='sample_dir',
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpointDir,
            sample_dir=FLAGS.local)

        show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpointDir)[0]:
                raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
