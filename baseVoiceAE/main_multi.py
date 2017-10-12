# coding=utf-8
import tensorflow as tf

import read
import inference
import loss
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('buckets', './', '数据源地址')
tf.flags.DEFINE_string('checkpointDir', 'saves/', "模型保存路径")
tf.flags.DEFINE_string('summaryDir', 'logs', "tensorboard保存路径")
tf.flags.DEFINE_integer('batch_size', 64, '批大小')
tf.flags.DEFINE_integer('frame_count', 60, "帧数")
tf.flags.DEFINE_integer('frequency', 16000, "采样率")
tf.flags.DEFINE_integer('kwidth', 32, '窗格大小')
tf.flags.DEFINE_integer('num_train', 20000, "训练次数")
tf.flags.DEFINE_integer('num_gpus', 2, "显卡数")
tf.flags.DEFINE_float('learning_rate', 3e-4, "学习速率")
tf.flags.DEFINE_float('beta1', 0.5, "Adam动量")
tf.flags.DEFINE_boolean('isTrain', True, '是否训练')
tf.flags.DEFINE_boolean('log_device_placement', False, '是否显示在哪块进行')


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
    reader = read.Reader(path=FLAGS.buckets + 'wavFile_train_frame_60.tfr', batch_size=FLAGS.batch_size,
                         window_size=FLAGS.frequency // FLAGS.frame_count, kwidth=FLAGS.kwidth)
    logits = inference.Inference(reader.wav_raw, FLAGS.kwidth, 2, FLAGS.isTrain, scope=scope).build_model()
    loss.loss(logits=logits, labels=reader.label)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    tf.add_to_collection('summary', tf.summary.scalar(scope + 'loss', losses[0]))
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
                    grads = optimizer.compute_gradients(loss_total)
                    tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    for grad, var in grads:
        if grad is not None:
            tf.add_to_collection('summary', tf.summary.histogram(var.op.name + '/gradients', grad))
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        tf.add_to_collection('summary', tf.summary.histogram(var.op.name, var))

    saver = tf.train.Saver()
    summaries = tf.get_collection('summary')
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
        sess.run(apply_gradient_op)

        if step % 10 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
        if step % 1000 == 0 or (step + 1) == FLAGS.num_train:
            checkpoint_path = os.path.join(FLAGS.checkpointDir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
