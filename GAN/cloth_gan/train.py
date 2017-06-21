# coding=utf-8
import tensorflow as tf
import numpy as np

flags = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 20, """批大小""")
tf.flags.DEFINE_integer('image_height', 200, """图片高度""")
tf.flags.DEFINE_integer('image_width', 200, """图片宽度""")
