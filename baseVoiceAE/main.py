# coding=utf-8
import tensorflow as tf
import read
import inference
import loss
import os
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS
print os.getcwd()
tf.flags.DEFINE_string('test_file', 'wavFile_test_frame_60.tfr', '数据源地址')
tf.flags.DEFINE_string('checkpointDir', 'saves/', "模型保存路径")
tf.flags.DEFINE_string('summaryDir', 'logs', "tensorboard保存路径")
tf.flags.DEFINE_integer('batch_size', 1, '批大小')
tf.flags.DEFINE_integer('frame_count', 60, "帧数")
tf.flags.DEFINE_integer('frequency', 16000, "采样率")
tf.flags.DEFINE_integer('kwidth', 18, '窗格大小')
tf.flags.DEFINE_integer('num_train', 1000, "训练次数")
tf.flags.DEFINE_float('learning_rate', 3e-4, "学习速率")
tf.flags.DEFINE_float('beta1', 0.5, "Adam动量")

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
reader = read.Reader(path=FLAGS.test_file, batch_size=FLAGS.batch_size,
                     window_size=FLAGS.frequency // FLAGS.frame_count, kwidth=FLAGS.kwidth)
tf.train.start_queue_runners(sess=sess, coord=coord)

logits = inference.Inference(reader.wav_raw, FLAGS.kwidth, 2, isTrain=False).build_model()
loss_val = loss.loss(logits=logits, labels=reader.label)

saver = tf.train.Saver()

tf.global_variables_initializer().run()

saver.restore(sess, os.path.join(FLAGS.checkpointDir))

tf.train.start_queue_runners(sess=sess, coord=coord)
labels = tf.reshape(reader.label, [-1])
logits_predict, ground_truth = sess.run([logits, labels])

plt.figure(1, [20, 12])
plt.subplot(411)
plt.title('predict')
plt.plot(logits_predict)
plt.subplot(412)
plt.title('truth')
plt.plot(ground_truth)
plt.subplot(413)
plt.title('both')
plt.plot(logits_predict, 'r')
plt.plot(ground_truth, 'b')
plt.subplot(414)
plt.title('loss')
plt.plot(abs(logits_predict - ground_truth))
plt.show()
