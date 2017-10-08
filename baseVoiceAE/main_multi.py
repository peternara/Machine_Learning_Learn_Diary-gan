# coding=utf-8
import datetime
import tensorflow as tf
import time

import read
import inference
import loss
import os

import numpy as np

os.chdir(os.path.dirname(__file__))

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('train_file', 'wavFile_train_frame_60.tfr', '数据源地址')
tf.flags.DEFINE_string('checkpointDir', 'saves/model.ckpt', "模型保存路径")
tf.flags.DEFINE_string('summaryDir', 'logs', "tensorboard保存路径")
tf.flags.DEFINE_integer('batch_size', 64, '批大小')
tf.flags.DEFINE_integer('frame_count', 60, "帧数")
tf.flags.DEFINE_integer('frequency', 16000, "采样率")
tf.flags.DEFINE_integer('kwidth', 32, '窗格大小')
tf.flags.DEFINE_integer('num_train', 20000, "训练次数")
tf.flags.DEFINE_integer('num_gpus', 2, "显卡数")
tf.flags.DEFINE_float('learning_rate', 3e-4, "学习速率")
tf.flags.DEFINE_float('beta1', 0.5, "Adam动量")
tf.flags.DEFINE_boolean('isTrain', True, '是否测试')
tf.flags.DEFINE_boolean('log_device_placement', True, '是否显示在哪块进行')


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)

            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tower_loss(scope):
    reader = read.Reader(path=FLAGS.train_file, batch_size=FLAGS.batch_size,
                         window_size=FLAGS.frequency // FLAGS.frame_count, kwidth=FLAGS.kwidth)
    logits = inference.Inference(reader.wav_raw, FLAGS.kwidth, 2, FLAGS.isTrain, scope=scope).build_model()
    loss.loss(logits=logits, labels=reader.label)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    for l in losses + [total_loss]:
        tf.summary.scalar(scope + ' (raw)', l)
    return total_loss


with tf.Graph().as_default(), tf.device('/cpu:0'):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1)
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gpu_%d' % i) as scope:
                    print "building Graph on {}".format(scope)
                    loss_total = tower_loss(scope)
                    tf.get_variable_scope().reuse_variables()
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = optimizer.compute_gradients(loss_total)
                    tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    for grad, var in grads:
        if grad is not None:
            summaries.append(
                tf.summary.histogram(var.op.name + '/gradients', grad))
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    saver = tf.train.Saver()
    summary_op = tf.summary.merge(summaries)
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.summary.FileWriter(FLAGS.summaryDir,
                                           graph=sess.graph)
    for step in xrange(FLAGS.num_train):
        start_time = time.time()
        _, loss_value = sess.run([apply_gradient_op, loss_total])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / FLAGS.num_gpus

            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value,
                                examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.checkpointDir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
