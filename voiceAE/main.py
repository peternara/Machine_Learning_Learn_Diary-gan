# coding=utf-8
import tensorflow as tf
import read
import inference
import loss
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('buckets', './data/', '数据源地址')
tf.flags.DEFINE_string('checkpointDir', 'saves/', "模型保存路径")
tf.flags.DEFINE_string('summaryDir', 'logs', "tensorboard保存路径")
tf.flags.DEFINE_string('test_wav', 'data/noisy.wav', "测试音频")
tf.flags.DEFINE_integer('batch_size', 150, '批大小')
tf.flags.DEFINE_integer('frequency', 16000, "采样率")
tf.flags.DEFINE_integer('kwidth', 18, '窗格大小')
tf.flags.DEFINE_integer('num_train', 10000, "训练次数")
tf.flags.DEFINE_float('learning_rate', 2e-4, "学习速率")
tf.flags.DEFINE_float('beta1', 0.5, "Adam动量")
tf.flags.DEFINE_float('canvas_size', 2 ** 14, "每条音频数据大小")
tf.flags.DEFINE_boolean('test', True, '是否测试')

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()

reader = read.Reader(sess=sess, path=FLAGS.buckets, batch_size=FLAGS.batch_size, canvas_size=FLAGS.canvas_size)
tf.train.start_queue_runners(sess=sess, coord=coord)

raw_wav, noisy_wav = reader.read()
if not FLAGS.test:
    output_wav = inference.Inference(noisy_wav, FLAGS.kwidth, 2, isTrain=True).build_model()
else:
    x = tf.placeholder(tf.float32, [1, FLAGS.canvas_size])
    output_wav = inference.Inference(x, FLAGS.kwidth, 2, isTrain=False).build_model()

losses = loss.Losses(output_wav, raw_wav).get_loss()
tf.summary.scalar('loss', losses)
tf.summary.audio('output_wav', output_wav, 16000)
tf.summary.audio('raw_wav', noisy_wav, 16000)

optimize = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1)
# optimize = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)

optimize_op = optimize.minimize(losses, var_list=tf.trainable_variables())

saver = tf.train.Saver(var_list=tf.trainable_variables())

tf.global_variables_initializer().run()


def clean(X):
    c_res = None
    for beg_i in range(0, X.shape[0], FLAGS.canvas_size):
        if X.shape[0] - beg_i < FLAGS.canvas_size:
            length = X.shape[0] - beg_i
            pad = FLAGS.canvas_size - length
        else:
            length = FLAGS.canvas_size
            pad = 0
        if pad > 0:
            x_ = np.concatenate((X[beg_i:beg_i + length], np.zeros(pad)))
        else:
            x_ = X[beg_i:beg_i + length]
        print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
        canvas_w = output_wav.eval(feed_dict={x: [x_]})
        canvas_w = canvas_w.reshape(FLAGS.canvas_size)
        print('canvas w shape: ', canvas_w.shape)
        if pad > 0:
            print('Removing padding of {} samples'.format(pad))
            # get rid of last padded samples
            canvas_w = canvas_w[:-pad]
        if c_res is None:
            c_res = canvas_w
        else:
            c_res = np.concatenate((c_res, canvas_w))
    return c_res


if not FLAGS.test:
    summary = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)
    summary_op = tf.summary.merge_all()
    num_train = FLAGS.num_train
    for i in xrange(0, num_train):
        optimize_op.run()
        print losses.eval()
        if i % 10 == 0 or i == num_train - 1:
            summary_data = summary_op.eval()
            summary.add_summary(summary_data, i)

        if i % 1000 == 0 or i == num_train - 1:
            saver.save(sess, os.path.join(FLAGS.checkpointDir, 'ae.model'))
else:
    import scipy.io.wavfile as wavfile
    import numpy as np

    saver.restore(sess, os.path.join(FLAGS.checkpointDir, 'ae.model'))
    fm, wav_data = wavfile.read(FLAGS.test_wav)
    wave = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
    wavfile.write('data/clear.wav', 16000, clean(wave))
