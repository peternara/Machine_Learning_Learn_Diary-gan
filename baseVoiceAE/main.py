# coding=utf-8
import tensorflow as tf
import read
import inference
import loss
import os

os.chdir(os.path.dirname(__file__))

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('train_file', 'wavFile_train_frame_60.tfr', '数据源地址')
tf.flags.DEFINE_string('checkpointDir', 'saves/model.ckpt', "模型保存路径")
tf.flags.DEFINE_string('summaryDir', 'logs', "tensorboard保存路径")
tf.flags.DEFINE_integer('batch_size', 1, '批大小')
tf.flags.DEFINE_integer('frame_count', 60, "帧数")
tf.flags.DEFINE_integer('frequency', 16000, "采样率")
tf.flags.DEFINE_integer('kwidth', 18, '窗格大小')
tf.flags.DEFINE_integer('num_train', 1000, "训练次数")
tf.flags.DEFINE_float('learning_rate', 3e-4, "学习速率")
tf.flags.DEFINE_float('beta1', 0.5, "Adam动量")
tf.flags.DEFINE_boolean('test', False, '是否测试')

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
reader = read.Reader(sess=sess, path=FLAGS.train_file, batch_size=FLAGS.batch_size,
                     window_size=FLAGS.frequency // FLAGS.frame_count, kwidth=FLAGS.kwidth)
tf.train.start_queue_runners(sess=sess, coord=coord)

num_batch_data_count = len(reader.read()[0])

input_tensor = tf.placeholder(tf.int16, [num_batch_data_count, FLAGS.kwidth])
label_tensor = tf.placeholder(tf.int16, [num_batch_data_count, 1])
input_tensor = tf.cast(input_tensor, tf.float32)
label_tensor = tf.cast(label_tensor, tf.float32)

logits = inference.Inference(input_tensor, FLAGS.kwidth, 2, not FLAGS.test).build_model()
loss_val = loss.Losses(logits=logits, labels=label_tensor).loss
tf.summary.scalar('loss', loss_val)

optimize = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1)
optimize_op = optimize.minimize(loss_val)

saver = tf.train.Saver()

tf.global_variables_initializer().run()

if not FLAGS.test:
    summary = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)
    summary_op = tf.summary.merge_all()
    for i in xrange(FLAGS.num_train):
        wavs, labels = reader.read()
        if i % 10 == 0:
            # print "loss: {}".format(loss_val.eval(feed_dict={
            #     input_tensor: wavs,
            #     label_tensor: labels
            # }))
            summary_data = summary_op.eval(feed_dict={
                input_tensor: wavs,
                label_tensor: labels
            })
            summary.add_summary(summary_data, i)
        if i % 1000 == 0:
            saver.save(sess, save_path=FLAGS.checkpointDir)

        optimize_op.run(feed_dict={
            input_tensor: wavs,
            label_tensor: labels
        })
else:
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wavfile

    saver.restore(sess, FLAGS.checkpointDir)
    reader = read.Reader(sess=sess, path='./wavFile_test_frame_60.tfr', batch_size=FLAGS.batch_size,
                         window_size=FLAGS.frequency // FLAGS.frame_count, kwidth=FLAGS.kwidth)
    tf.train.start_queue_runners(sess=sess, coord=coord)
    wavs, labels = reader.read()
    logits_predict, ground_truth = sess.run([logits, label_tensor], feed_dict={
        input_tensor: wavs,
        label_tensor: labels
    })
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

    # print "loss: {}".format(loss_val.eval(feed_dict={
    #     input_tensor: wavs,
    #     label_tensor: labels
    # }))
