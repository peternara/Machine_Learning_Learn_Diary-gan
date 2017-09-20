# coding=utf-8
import tensorflow as tf
import read

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('buckets', 'wavFile_train_frame_60.tfr', '数据源地址')
tf.flags.DEFINE_string('checkpointDir', 'saves', "模型保存路径")
tf.flags.DEFINE_string('summaryDir', 'logs', "tensorboard保存路径")
tf.flags.DEFINE_integer('frame_count', 60, "帧数")
tf.flags.DEFINE_integer('frequency', 16000, "采样率")

sess = tf.InteractiveSession()
reader = read.Reader(FLAGS.buckets, 64, 16000 // 60)

batch = reader.read()

print batch
