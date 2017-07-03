# coding=utf-8
import os

import numpy as np
import tensorflow as tf
import ac_gan

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("summaryDir", "./summary/", "TensorBoard路径")
tf.app.flags.DEFINE_string('buckets', './Records/', '图片文件夹')
tf.app.flags.DEFINE_string("checkpointDir", "./checkpoint_dir/", "模型保存路径")
tf.app.flags.DEFINE_integer('train_steps', 100000, '训练次数')
tf.app.flags.DEFINE_float('learning_rate', 0.01, '学习速率')
tf.app.flags.DEFINE_float('beta1', 0.5, 'Adam动量')
tf.app.flags.DEFINE_integer('num_classes', 133, '类型数')


def train():
    # placeholder for z
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name='z')
    sess = tf.Session()

    # get images and labels
    reader = ac_gan.Reader(path=FLAGS.buckets, pattem='*.tfrecords', batch_size=FLAGS.batch_size,
                           num_classes=FLAGS.num_classes)
    labels, images = reader.read()

    # logits
    [
        source_logits_real,
        class_logits_real,
        source_logits_fake,
        class_logits_fake,
        generated_images
    ] = ac_gan.inference(images, labels, z)

    # loss
    d_loss, g_loss, dc_loss = ac_gan.loss(labels,
                                          source_logits_real,
                                          class_logits_real,
                                          source_logits_fake,
                                          class_logits_fake,
                                          generated_images)

    # train the model
    train_d_op, train_g_op = ac_gan.train(d_loss, g_loss)

    # summary
    summary_image = tf.summary.image('generate_image', generated_images)
    summary_gloss = tf.summary.scalar('g_loss', g_loss)
    summary_dloss = tf.summary.scalar('d_loss', d_loss)
    summary_dcloss = tf.summary.scalar('dc_loss', dc_loss)

    with sess.as_default():
        init = tf.global_variables_initializer()

        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        saver = tf.train.Saver()

        ac_gan.load(sess, saver, checkpointDir=FLAGS.checkpointDir)

        summary_writer = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)

        for step in xrange(FLAGS.train_steps):

            random_z = np.random.uniform(
                -1, 1, size=(FLAGS.batch_size, FLAGS.z_dim)).astype(np.float32)

            sess.run(train_d_op, feed_dict={z: random_z})
            sess.run(train_g_op, feed_dict={z: random_z})
            sess.run(train_g_op, feed_dict={z: random_z})
            print "step: {}".format(step)
            if step % 10 == 0:
                merge_op = tf.summary.merge([summary_image, summary_gloss, summary_dloss, summary_dcloss])
            else:
                merge_op = tf.summary.merge([summary_gloss, summary_dloss, summary_dcloss])
            merge_data = sess.run(merge_op, feed_dict={z: random_z})
            summary_writer.add_summary(merge_data, step)
            if step % 100 == 0:
                saver.save(sess, os.path.join(FLAGS.checkpointDir, 'baiduJS.model'), global_step=step)
        summary_writer.close()


def main(argv=None):
    # check folder exist
    if tf.gfile.Exists(FLAGS.summaryDir):
        tf.gfile.DeleteRecursively(FLAGS.summaryDir)
        tf.gfile.MkDir(FLAGS.summaryDir)
    else:
        tf.gfile.MkDir(FLAGS.summaryDir)
    if not tf.gfile.Exists(FLAGS.checkpointDir):
        tf.gfile.MkDir(FLAGS.checkpointDir)
    for i in FLAGS.__flags:
        print "{}: {}".format(i, FLAGS.__flags[i])
    train()


if __name__ == '__main__':
    tf.app.run()
